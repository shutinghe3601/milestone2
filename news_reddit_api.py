import os

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
    # Run examples
    example_1_get_news_posts()
    example_2_get_comments()
    example_3_get_user_info()
