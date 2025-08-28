# News Reddit API Usage Guide

A simple API client for fetching posts, comments, and user information from r/news.

## Installation

```bash
# Install all dependencies
pip install -r requirements.txt
```

## Configuration

Ask the author for the `.env` file and place it in the project root directory.

## Usage

```python
from news_reddit_api import NewsRedditAPI

api = NewsRedditAPI()

# Get news posts
posts = api.get_news_posts(5)

# Get comments
comments = api.get_post_comments(post_url, 10)

# Get user information
user_info = api.get_user_info('username')
```

## Main Features

- `get_news_posts(limit=5)` - Get hot posts
- `get_post_comments(post_url, limit=10)` - Get comments
- `get_user_info(username)` - Get user information

## Run

```bash
python news_reddit_api.py
```
