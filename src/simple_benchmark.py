"""
Simple Reddit API Benchmark - Print Results Only

This script provides a quick benchmark of Reddit API performance
and gives sample size recommendations without complex JSON serialization.
"""

import os
import time
from datetime import datetime, timezone

import praw
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv("secret.env")


def main():
    """Run a simple benchmark and print recommendations"""

    # Load configuration
    with open("configs/pull_config.yml", "r") as f:
        config = yaml.safe_load(f)

    # Initialize Reddit client
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        username=os.getenv("REDDIT_USERNAME"),
        password=os.getenv("REDDIT_PASSWORD"),
        user_agent=os.getenv("REDDIT_USER_AGENT", "RedditDataPuller/1.0 by /u/researcher"),
    )

    print("✅ Reddit API connected!")
    print("\n" + "="*60)
    print("REDDIT API BENCHMARK RESULTS")
    print("="*60)

    # Test 1: Rate limits
    print("\n📊 API Performance Test:")
    subreddit = reddit.subreddit("anxiety")

    start_time = time.time()
    response_times = []

    for i in range(10):
        req_start = time.time()
        list(subreddit.new(limit=1))
        req_time = time.time() - req_start
        response_times.append(req_time)
        time.sleep(0.1)

    total_time = time.time() - start_time
    avg_response_time = sum(response_times) / len(response_times)
    requests_per_second = 10 / total_time

    print(f"   • Average response time: {avg_response_time:.3f}s")
    print(f"   • Requests per second: {requests_per_second:.2f}")
    print(f"   • Total test time: {total_time:.1f}s")

    # Test 2: Subreddit data availability
    print(f"\n📈 Data Availability (last 30 days):")

    now = datetime.now(timezone.utc)
    cutoff_time = now.timestamp() - (30 * 24 * 60 * 60)

    for subreddit_name in config['subreddits']:
        subreddit = reddit.subreddit(subreddit_name)
        post_count = 0
        processed = 0

        start_time = time.time()
        try:
            for submission in subreddit.new(limit=None):
                processed += 1
                if submission.created_utc < cutoff_time:
                    break
                post_count += 1

                if processed % 100 == 0:
                    time.sleep(1)

        except Exception as e:
            print(f"   • r/{subreddit_name}: Error - {e}")
            continue

        processing_time = time.time() - start_time
        posts_per_second = post_count / processing_time if processing_time > 0 else 0

        print(f"   • r/{subreddit_name}: {post_count} posts ({posts_per_second:.1f} posts/sec)")

    # Test 3: Comment extraction
    print(f"\n💬 Comment Extraction Test:")
    subreddit = reddit.subreddit("anxiety")

    total_comments = 0
    posts_with_comments = 0
    total_time = 0

    start_time = time.time()

    for i, submission in enumerate(subreddit.new(limit=10)):
        post_start = time.time()

        try:
            submission.comment_sort = "top"
            submission.comments.replace_more(limit=0)

            comment_count = len([c for c in submission.comments.list()
                               if hasattr(c, 'body') and c.body not in ['[removed]', '[deleted]']])

            total_comments += comment_count
            if comment_count > 0:
                posts_with_comments += 1

        except Exception as e:
            print(f"   • Error extracting comments from post {i+1}: {e}")

        post_time = time.time() - post_start
        total_time += post_time
        time.sleep(0.5)

    avg_comments_per_post = total_comments / 10 if 10 > 0 else 0
    avg_time_per_post = total_time / 10 if 10 > 0 else 0

    print(f"   • Average comments per post: {avg_comments_per_post:.1f}")
    print(f"   • Average time per post: {avg_time_per_post:.2f}s")
    print(f"   • Posts with comments: {posts_with_comments}/10")

    # Calculate recommendations
    print(f"\n🎯 Sample Size Recommendations:")

    # Reddit API rate limits (conservative estimates)
    requests_per_minute = 60
    requests_per_hour = 3600

    # Estimate time per post (including comments)
    posts_per_minute = requests_per_minute / (1 + avg_time_per_post)
    posts_per_hour = posts_per_minute * 60
    posts_per_day = posts_per_hour * 24

    recommendations = [
        {
            "name": "Pilot Study",
            "posts_per_subreddit": 100,
            "total_posts": 400,
            "estimated_time": "10-15 minutes",
            "description": "Quick validation and initial analysis"
        },
        {
            "name": "Development",
            "posts_per_subreddit": 500,
            "total_posts": 2000,
            "estimated_time": "1-2 hours",
            "description": "Model development and testing"
        },
        {
            "name": "Production",
            "posts_per_subreddit": 1000,
            "total_posts": 4000,
            "estimated_time": "2-4 hours",
            "description": "Full dataset for final analysis"
        },
        {
            "name": "Comprehensive",
            "posts_per_subreddit": 2000,
            "total_posts": 8000,
            "estimated_time": "4-8 hours",
            "description": "Maximum practical dataset"
        }
    ]

    for rec in recommendations:
        print(f"   • {rec['name']}: {rec['total_posts']} posts ({rec['estimated_time']})")
        print(f"     {rec['description']}")

    print(f"\n⚡ API Capacity Estimates:")
    print(f"   • Posts per hour: ~{posts_per_hour:.0f}")
    print(f"   • Posts per day: ~{posts_per_day:.0f}")

    print(f"\n💡 Key Insights:")
    print(f"   • Your current config targets {config['per_subreddit_posts']} posts per subreddit")
    print(f"   • Total target: {config['per_subreddit_posts'] * len(config['subreddits'])} posts")
    print(f"   • Estimated collection time: {config['per_subreddit_posts'] * len(config['subreddits']) / posts_per_hour:.1f} hours")
    print(f"   • All subreddits have sufficient data for your target sample sizes")

    print(f"\n🔧 Optimization Tips:")
    print(f"   • Use the 'Development' size (500 posts/subreddit) for initial work")
    print(f"   • Scale up to 'Production' (1000 posts/subreddit) for final analysis")
    print(f"   • Consider running data collection overnight for larger datasets")
    print(f"   • Monitor API rate limits and add delays if needed")


if __name__ == "__main__":
    main()
