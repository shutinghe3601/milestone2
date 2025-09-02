import os
import argparse
import json
from datetime import datetime
from pathlib import Path

import praw
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class NewsRedditAPI:
    """Reddit API client focused on news with comments functionality"""

    def __init__(self):
        """Initialize Reddit API client"""
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            username=os.getenv("REDDIT_USERNAME"),
            password=os.getenv("REDDIT_PASSWORD"),
            user_agent=os.getenv("REDDIT_USER_AGENT"),
        )
        print("✅ Reddit API connected!")

    def get_news_posts(self, limit=5):
        """Get hot posts from r/news"""
        subreddit = self.reddit.subreddit("news")
        posts = []

        for submission in subreddit.hot(limit=limit):
            post = {
                "title": submission.title,
                "author": str(submission.author) if submission.author else "[deleted]",
                "score": submission.score,
                "url": submission.url,
                "comments_count": submission.num_comments,
                "created_time": submission.created_utc,
                "permalink": f"https://reddit.com{submission.permalink}",
            }
            posts.append(post)

        return posts

    def get_post_comments(self, post_url, limit=10):
        """Get comments from a specific post"""
        # Extract submission ID from URL
        if "reddit.com" in post_url:
            submission_id = post_url.split("/comments/")[1].split("/")[0]
        else:
            submission_id = post_url

        submission = self.reddit.submission(id=submission_id)
        submission.comment_sort = "top"  # Sort by top comments
        submission.comments.replace_more(limit=0)  # Remove "load more comments" links

        comments = []
        for comment in submission.comments[:limit]:
            comment_info = {
                "body": comment.body,
                "author": str(comment.author) if comment.author else "[deleted]",
                "score": comment.score,
                "created_time": comment.created_utc,
                "replies_count": len(comment.replies),
            }
            comments.append(comment_info)

        return comments

    def get_user_info(self, username):
        """Get information about a Reddit user"""
        try:
            user = self.reddit.redditor(username)
            user_info = {
                "username": user.name,
                "comment_karma": user.comment_karma,
                "link_karma": user.link_karma,
                "created_time": user.created_utc,
                "is_gold": user.is_gold,
                "is_mod": user.is_mod,
            }
            return user_info
        except:
            return None


# Usage examples
def example_1_get_news_posts():
    """Example 1: Get hot news posts"""
    print("=== Example 1: Get hot posts from r/news ===")
    api = NewsRedditAPI()
    posts = api.get_news_posts(3)

    for i, post in enumerate(posts, 1):
        print(f"{i}. {post['title']}")
        print(
            f"   Author: {post['author']} | Score: {post['score']} | Comments: {post['comments_count']}"
        )
        print(f"   URL: {post['permalink']}")
        print()


def example_2_get_comments():
    """Example 2: Get comments from a news post"""
    print("=== Example 2: Get comments from a news post ===")
    api = NewsRedditAPI()

    # First get a news post
    posts = api.get_news_posts(1)
    if posts:
        post = posts[0]
        print(f"Post: {post['title']}")
        print(f"URL: {post['permalink']}")
        print()

        # Get comments
        comments = api.get_post_comments(post["permalink"], 5)
        print("Top comments:")
        for i, comment in enumerate(comments, 1):
            print(f"{i}. {comment['body'][:100]}...")
            print(
                f"   Author: {comment['author']} | Score: {comment['score']} | Replies: {comment['replies_count']}"
            )
            print()


def example_3_get_user_info():
    """Example 3: Get user information from comments"""
    print("=== Example 3: Get user information from comments ===")
    api = NewsRedditAPI()

    # Get a post and its comments
    posts = api.get_news_posts(1)
    if posts:
        comments = api.get_post_comments(posts[0]["permalink"], 3)

        for i, comment in enumerate(comments, 1):
            print(f"Comment {i}: {comment['body'][:50]}...")
            print(f"Author: {comment['author']}")

            # Get user info (skip deleted users)
            if comment["author"] != "[deleted]":
                user_info = api.get_user_info(comment["author"])
                if user_info:
                    print(
                        f"  Karma: {user_info['comment_karma']} comment, {user_info['link_karma']} link"
                    )
                    print(
                        f"  Gold: {user_info['is_gold']} | Mod: {user_info['is_mod']}"
                    )
                else:
                    print("  User info not available")
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch Reddit posts and save to data/raw as JSONL"
    )
    parser.add_argument(
        "--subreddits",
        type=str,
        default="news",
        help="Comma-separated list of subreddits (default: news)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Number of posts to fetch per subreddit (default: 50)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="data/raw",
        help="Output directory for JSONL files (default: data/raw)",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default=None,
        help="Explicit output JSONL filename (overrides autogenerated)",
    )
    parser.add_argument(
        "--comments-limit",
        type=int,
        default=20,
        help="Number of top comments to include per post (default: 20)",
    )
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Run built-in examples instead of fetching/saving",
    )

    args = parser.parse_args()

    if args.examples:
        # Run examples
        example_1_get_news_posts()
        example_2_get_comments()
        example_3_get_user_info()
    else:
        api = NewsRedditAPI()
        out_path = Path(args.outdir)
        out_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        subreddits = [s.strip() for s in args.subreddits.split(",") if s.strip()]

        for sub in subreddits:
            subreddit = api.reddit.subreddit(sub)
            posts = []
            for submission in subreddit.hot(limit=args.limit):
                post = {
                    "id": submission.id,
                    "subreddit": str(submission.subreddit),
                    "title": submission.title,
                    "selftext": submission.selftext,
                    "author": str(submission.author) if submission.author else "[deleted]",
                    "score": submission.score,
                    "upvote_ratio": getattr(submission, "upvote_ratio", None),
                    "url": submission.url,
                    "num_comments": submission.num_comments,
                    "created_utc": submission.created_utc,
                    "permalink": f"https://reddit.com{submission.permalink}",
                    "link_flair_text": getattr(submission, "link_flair_text", None),
                    "is_self": getattr(submission, "is_self", None),
                }

                # Include top comments content up to comments_limit
                try:
                    submission.comment_sort = "top"
                    submission.comments.replace_more(limit=0)
                    top_comments = []
                    for c in submission.comments[: args.comments_limit]:
                        top_comments.append(
                            {
                                "id": c.id,
                                "body": c.body,
                                "author": str(c.author) if c.author else "[deleted]",
                                "score": c.score,
                                "created_utc": c.created_utc,
                                "replies_count": len(c.replies),
                            }
                        )
                    post["top_comments"] = top_comments
                except Exception as e:
                    post["top_comments_error"] = str(e)

                posts.append(post)

            outfile = (
                out_path / args.outfile if args.outfile else out_path / f"{sub}_{timestamp}.jsonl"
            )
            with outfile.open("w", encoding="utf-8") as f:
                for p in posts:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")

            print(f"Saved {len(posts)} posts from r/{sub} to {outfile}")
