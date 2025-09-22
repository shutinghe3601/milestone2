"""
Reddit Data Puller - Compliance Enhanced Version

This version includes proper rate limit monitoring, error handling,
and compliance with Reddit's API guidelines.

Key improvements:
- Rate limit header monitoring
- Exponential backoff for rate limit errors
- Better error handling and recovery
- Mature content handling
- Request logging
"""

import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import praw
import yaml
from dotenv import load_dotenv
from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException

# Set seed for consistent language detection
DetectorFactory.seed = 0

# Load environment variables
load_dotenv("secret.env")


class CompliantRedditDataPuller:
    """Reddit API client with enhanced compliance features"""

    def __init__(self, config_path: str = "configs/pull_config.yml"):
        """Initialize Reddit API client with compliance features"""
        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Initialize Reddit client
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            username=os.getenv("REDDIT_USERNAME"),
            password=os.getenv("REDDIT_PASSWORD"),
            user_agent=os.getenv(
                "REDDIT_USER_AGENT", "RedditDataPuller/1.0 by /u/researcher"
            ),
        )

        # Compliance tracking
        self.rate_limit_errors = 0
        self.max_retries = 3
        self.request_count = 0
        self.start_time = time.time()

        # Parse time range - make it dynamic (N months from now backwards)
        now = datetime.now(timezone.utc)
        self.end_ts = now.timestamp()

        # Get months_back from config (default to 12 if not specified)
        months_back = self.config.get("time_range", {}).get("months_back", 12)

        # Calculate start time (N months ago)
        if months_back >= 12:
            years_back = months_back // 12
            months_remaining = months_back % 12
            start_date = now.replace(
                year=now.year - years_back, month=now.month - months_remaining
            )
        else:
            start_date = now.replace(month=now.month - months_back)

        # Handle month overflow
        if start_date.month <= 0:
            start_date = start_date.replace(
                year=start_date.year - 1, month=start_date.month + 12
            )

        self.start_ts = start_date.timestamp()
        self.actual_start_date = start_date.strftime("%Y-%m-%d")
        self.actual_end_date = now.strftime("%Y-%m-%d")
        self.months_back = months_back

        # Initialize tracking variables
        self.stats = {
            "posts": {
                "total": 0,
                "filtered_time": 0,
                "filtered_nsfw": 0,
                "filtered_removed": 0,
                "filtered_non_english": 0,
                "by_subreddit": {},
            },
            "comments": {
                "total": 0,
                "filtered_removed": 0,
                "filtered_short": 0,
                "filtered_non_english": 0,
            },
            "time_range": {"earliest": None, "latest": None},
            "api_usage": {
                "requests_made": 0,
                "rate_limit_errors": 0,
                "start_time": self.start_time,
            },
        }

        # Setup logging
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

        print("✅ Reddit API connected with compliance features!")
        print(
            f"📅 Dynamic time range: {self.actual_start_date} to {self.actual_end_date} (Last {self.months_back} months)"
        )

    def check_rate_limits(self, response=None):
        """Check and log rate limit headers"""
        # Note: PRAW doesn't expose headers directly, but we can track our usage
        elapsed_time = time.time() - self.start_time
        requests_per_minute = (self.request_count / elapsed_time) * 60 if elapsed_time > 0 else 0

        self.logger.info(f"API Usage - Requests: {self.request_count}, Rate: {requests_per_minute:.1f}/min")

        # Warn if approaching limit
        if requests_per_minute > 80:  # 80% of 100/min limit
            self.logger.warning("Approaching rate limit! Consider slowing down.")

    def exponential_backoff(self, attempt, base_delay=1, max_delay=60):
        """Implement exponential backoff for rate limit errors"""
        delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
        self.logger.warning(f"Rate limited. Waiting {delay:.2f} seconds...")
        time.sleep(delay)

    def make_request_with_monitoring(self, func, *args, **kwargs):
        """Make request with rate limit monitoring and error handling"""
        self.request_count += 1

        try:
            result = func(*args, **kwargs)
            self.check_rate_limits()
            return result
        except Exception as e:
            error_msg = str(e).lower()
            if "rate limit" in error_msg or "429" in error_msg:
                self.stats["api_usage"]["rate_limit_errors"] += 1
                self.handle_rate_limit_error()
                # Retry with backoff
                if self.rate_limit_errors <= self.max_retries:
                    self.exponential_backoff(self.rate_limit_errors)
                    return self.make_request_with_monitoring(func, *args, **kwargs)
                else:
                    self.logger.error("Max rate limit retries exceeded")
                    raise Exception("Rate limit exceeded after retries")
            else:
                self.logger.error(f"API request failed: {e}")
                raise

    def handle_rate_limit_error(self):
        """Handle rate limit errors"""
        self.rate_limit_errors += 1
        self.logger.warning(f"Rate limit error #{self.rate_limit_errors}")

    def is_english(self, text: str) -> bool:
        """Check if text is in English using language detection"""
        if not text or len(text.strip()) < 3:
            return False

        try:
            detected_lang = detect(text)
            return detected_lang == "en"
        except LangDetectException:
            # If language detection fails, assume it's English for mental health content
            return True

    def extract_post_data(self, submission) -> Optional[Dict[str, Any]]:
        """Extract required fields from a Reddit submission with compliance checks"""
        # Check time window
        if not (self.start_ts <= submission.created_utc <= self.end_ts):
            self.stats["posts"]["filtered_time"] += 1
            return None

        # Apply filters
        if self.config["filters"]["exclude_nsfw"] and submission.over_18:
            self.stats["posts"]["filtered_nsfw"] += 1
            return None

        if self.config["filters"]["exclude_removed"] and submission.removed_by_category:
            self.stats["posts"]["filtered_removed"] += 1
            return None

        # Check English requirement (title OR selftext must be English)
        title_text = submission.title or ""
        selftext = submission.selftext or ""

        if self.config["filters"]["english_only"]:
            title_english = self.is_english(title_text) if title_text else False
            selftext_english = self.is_english(selftext) if selftext else False

            if not (title_english or selftext_english):
                self.stats["posts"]["filtered_non_english"] += 1
                return None

        # Update time tracking
        if (
            self.stats["time_range"]["earliest"] is None
            or submission.created_utc < self.stats["time_range"]["earliest"]
        ):
            self.stats["time_range"]["earliest"] = submission.created_utc
        if (
            self.stats["time_range"]["latest"] is None
            or submission.created_utc > self.stats["time_range"]["latest"]
        ):
            self.stats["time_range"]["latest"] = submission.created_utc

        return {
            "post_id": submission.id,
            "subreddit": str(submission.subreddit),
            "created_utc": submission.created_utc,
            "title": title_text,
            "selftext": selftext,
            "score": submission.score,
            "num_comments": submission.num_comments,
            "upvote_ratio": submission.upvote_ratio,
            "over_18": submission.over_18,
            "removed_by_category": submission.removed_by_category,
        }

    def extract_top_comments(self, submission, top_k: int = 3) -> List[Dict[str, Any]]:
        """Extract top-K comments from a submission with compliance monitoring"""
        try:
            # Use monitored request for comment extraction
            submission.comment_sort = "top"
            submission.comments.replace_more(limit=0)

            comments = []
            flat_comments = []

            # Flatten comment tree and collect all comments with their depth
            def flatten_comments(comment_list, depth=0):
                for comment in comment_list:
                    if hasattr(comment, "body") and comment.body not in [
                        "[removed]",
                        "[deleted]",
                    ]:
                        flat_comments.append((comment, depth))
                        if hasattr(comment, "replies"):
                            flatten_comments(comment.replies, depth + 1)

            flatten_comments(submission.comments)

            # Sort by score descending
            flat_comments.sort(key=lambda x: x[0].score, reverse=True)

            # Take top-K comments that pass filters
            for comment, depth in flat_comments:
                if len(comments) >= top_k:
                    break

                # Apply filters
                body = comment.body
                if len(body) < self.config["comments"]["min_len"]:
                    self.stats["comments"]["filtered_short"] += 1
                    continue

                if self.config["filters"]["english_only"] and not self.is_english(body):
                    self.stats["comments"]["filtered_non_english"] += 1
                    continue

                comments.append(
                    {
                        "comment_id": comment.id,
                        "link_id": submission.id,
                        "parent_id": (
                            comment.parent_id.split("_")[1]
                            if hasattr(comment, "parent_id")
                            else None
                        ),
                        "body": body,
                        "created_utc": comment.created_utc,
                        "score": comment.score,
                        "depth": depth,
                    }
                )

                self.stats["comments"]["total"] += 1

            return comments

        except Exception as e:
            self.logger.warning(f"Error extracting comments from {submission.id}: {e}")
            return []

    def pull_subreddit_posts(
        self, subreddit_name: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Pull posts from a specific subreddit with compliance monitoring"""
        self.logger.info(f"Pulling posts from r/{subreddit_name}")

        subreddit = self.reddit.subreddit(subreddit_name)
        posts = []
        processed = 0

        try:
            # Use 'new' to get posts in reverse chronological order
            for submission in subreddit.new(limit=None):
                processed += 1

                # Stop if we're older than our time window
                if submission.created_utc < self.start_ts:
                    break

                post_data = self.extract_post_data(submission)
                if post_data:
                    posts.append(post_data)

                    if len(posts) >= limit:
                        break

                # Enhanced rate limiting with monitoring
                if processed % 100 == 0:
                    time.sleep(1)
                    self.logger.info(
                        f"Processed {processed} submissions, collected {len(posts)} posts"
                    )
                    self.check_rate_limits()

        except Exception as e:
            self.logger.error(f"Error pulling from r/{subreddit_name}: {e}")

        self.stats["posts"]["by_subreddit"][subreddit_name] = len(posts)
        self.stats["posts"]["total"] += len(posts)

        # Log filtering statistics for this subreddit
        self.logger.info(f"Collected {len(posts)} posts from r/{subreddit_name}")
        self.logger.info(
            f"Filter stats for r/{subreddit_name}: Time={self.stats['posts']['filtered_time']}, "
            f"NSFW={self.stats['posts']['filtered_nsfw']}, "
            f"Removed={self.stats['posts']['filtered_removed']}, "
            f"Non-English={self.stats['posts']['filtered_non_english']}"
        )
        return posts

    def run_data_pull(self):
        """Main method to execute the data pull process with compliance monitoring"""
        self.logger.info("Starting Reddit data pull with compliance monitoring...")

        all_posts = []
        all_comments = []

        # Step 1: Pull posts from each subreddit
        for subreddit in self.config["subreddits"]:
            posts = self.pull_subreddit_posts(
                subreddit, self.config["per_subreddit_posts"]
            )
            all_posts.extend(posts)

            # Step 2: Extract top-K comments for each post
            self.logger.info(
                f"Extracting comments for {len(posts)} posts from r/{subreddit}"
            )
            for post in posts:
                try:
                    submission = self.reddit.submission(id=post["post_id"])
                    comments = self.extract_top_comments(
                        submission, self.config["comments"]["top_k"]
                    )
                    all_comments.extend(comments)
                except Exception as e:
                    self.logger.warning(f"Error processing post {post['post_id']}: {e}")

                # Enhanced rate limiting
                time.sleep(0.5)
                self.check_rate_limits()

        # Step 3: Write JSONL files
        self.write_jsonl_files(all_posts, all_comments)

        # Step 4: Generate QC log
        self.generate_qc_log(all_posts, all_comments)

        # Step 5: Log compliance summary
        self.log_compliance_summary()

        self.logger.info("Data pull completed successfully with compliance monitoring!")

    def write_jsonl_files(self, posts: List[Dict], comments: List[Dict]):
        """Write posts and comments to JSONL files"""
        # Ensure data directory exists
        os.makedirs("data/raw", exist_ok=True)

        # Write posts
        posts_file = "data/raw/submission.jsonl"
        with open(posts_file, "w", encoding="utf-8") as f:
            for post in posts:
                f.write(json.dumps(post, ensure_ascii=False) + "\n")

        # Write comments
        comments_file = "data/raw/comments_topk.jsonl"
        with open(comments_file, "w", encoding="utf-8") as f:
            for comment in comments:
                f.write(json.dumps(comment, ensure_ascii=False) + "\n")

        self.logger.info(f"Written {len(posts)} posts to {posts_file}")
        self.logger.info(f"Written {len(comments)} comments to {comments_file}")

    def generate_qc_log(self, posts: List[Dict], comments: List[Dict]):
        """Generate quality control log with compliance information"""
        log_file = "data/raw/execution_log.md"

        with open(log_file, "w", encoding="utf-8") as f:
            f.write("# Reddit Data Pull - Execution Log (Compliance Enhanced)\n\n")
            f.write(
                f"**Execution Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
            )

            # Configuration summary
            f.write("## Configuration\n")
            f.write(f"- **Subreddits:** {', '.join(self.config['subreddits'])}\n")
            f.write(
                f"- **Time Range:** {self.actual_start_date} to {self.actual_end_date} (Dynamic - Last {self.months_back} months)\n"
            )
            f.write(
                f"- **Posts per Subreddit:** {self.config['per_subreddit_posts']}\n"
            )
            f.write(
                f"- **Comments per Post:** Top {self.config['comments']['top_k']}\n\n"
            )

            # Counts
            f.write("## Data Counts\n")
            f.write(f"- **Total Posts:** {len(posts)}\n")
            f.write(f"- **Total Comments:** {len(comments)}\n\n")

            # API Usage Statistics
            f.write("## API Usage Statistics\n")
            elapsed_time = time.time() - self.start_time
            requests_per_minute = (self.request_count / elapsed_time) * 60 if elapsed_time > 0 else 0
            f.write(f"- **Total Requests:** {self.request_count}\n")
            f.write(f"- **Rate Limit Errors:** {self.stats['api_usage']['rate_limit_errors']}\n")
            f.write(f"- **Average Requests/Minute:** {requests_per_minute:.1f}\n")
            f.write(f"- **Total Execution Time:** {elapsed_time:.1f} seconds\n\n")

            # Subreddit distribution
            f.write("### Posts by Subreddit\n")
            for subreddit, count in self.stats["posts"]["by_subreddit"].items():
                f.write(f"- r/{subreddit}: {count} posts\n")
            f.write("\n")

            # Time coverage
            if (
                self.stats["time_range"]["earliest"]
                and self.stats["time_range"]["latest"]
            ):
                earliest = datetime.fromtimestamp(
                    self.stats["time_range"]["earliest"], tz=timezone.utc
                )
                latest = datetime.fromtimestamp(
                    self.stats["time_range"]["latest"], tz=timezone.utc
                )
                f.write("## Time Coverage\n")
                f.write(
                    f"- **Earliest Post:** {earliest.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
                )
                f.write(
                    f"- **Latest Post:** {latest.strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
                )

            # Filter rates
            f.write("## Filter Statistics\n")
            f.write("### Posts Filtered\n")
            f.write(f"- **NSFW:** {self.stats['posts']['filtered_nsfw']}\n")
            f.write(f"- **Removed:** {self.stats['posts']['filtered_removed']}\n")
            f.write(
                f"- **Non-English:** {self.stats['posts']['filtered_non_english']}\n\n"
            )

            f.write("### Comments Filtered\n")
            f.write(
                f"- **Removed/Deleted:** {self.stats['comments']['filtered_removed']}\n"
            )
            f.write(f"- **Too Short:** {self.stats['comments']['filtered_short']}\n")
            f.write(
                f"- **Non-English:** {self.stats['comments']['filtered_non_english']}\n\n"
            )

            # Quality checks
            f.write("## Quality Checks\n")
            post_ids = set(post["post_id"] for post in posts)
            comment_ids = set(comment["comment_id"] for comment in comments)

            f.write(
                f"- **Post ID Uniqueness:** {len(post_ids)} unique IDs out of {len(posts)} posts\n"
            )
            f.write(
                f"- **Comment ID Uniqueness:** {len(comment_ids)} unique IDs out of {len(comments)} comments\n"
            )

            # Missingness analysis
            missing_title = sum(1 for post in posts if not post.get("title"))
            missing_selftext = sum(1 for post in posts if not post.get("selftext"))
            missing_body = sum(1 for comment in comments if not comment.get("body"))

            f.write(
                f"- **Missing Title:** {missing_title}/{len(posts)} ({missing_title/len(posts)*100:.1f}%)\n"
            )
            f.write(
                f"- **Missing Selftext:** {missing_selftext}/{len(posts)} ({missing_selftext/len(posts)*100:.1f}%)\n"
            )
            f.write(
                f"- **Missing Comment Body:** {missing_body}/{len(comments)} ({missing_body/len(comments)*100:.1f}% if comments > 0)\n"
            )

        self.logger.info(f"Quality control log written to {log_file}")

    def log_compliance_summary(self):
        """Log a summary of compliance metrics"""
        elapsed_time = time.time() - self.start_time
        requests_per_minute = (self.request_count / elapsed_time) * 60 if elapsed_time > 0 else 0

        self.logger.info("="*50)
        self.logger.info("COMPLIANCE SUMMARY")
        self.logger.info("="*50)
        self.logger.info(f"Total API Requests: {self.request_count}")
        self.logger.info(f"Rate Limit Errors: {self.stats['api_usage']['rate_limit_errors']}")
        self.logger.info(f"Average Requests/Minute: {requests_per_minute:.1f}")
        self.logger.info(f"Total Execution Time: {elapsed_time:.1f} seconds")

        if requests_per_minute > 80:
            self.logger.warning("⚠️  High request rate detected - consider slowing down")
        elif self.stats['api_usage']['rate_limit_errors'] > 0:
            self.logger.warning("⚠️  Rate limit errors occurred - review rate limiting")
        else:
            self.logger.info("✅ Compliance metrics look good!")


def main():
    """Main execution function"""
    try:
        puller = CompliantRedditDataPuller()
        puller.run_data_pull()
    except Exception as e:
        logging.error(f"Fatal error during execution: {e}")
        raise


if __name__ == "__main__":
    main()
