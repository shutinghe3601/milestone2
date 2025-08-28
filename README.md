# Reddit API Spike

Testing branch for connecting to the Reddit API and pulling posts, comments, and user info.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Config

- Copy .env.example to .env and add your Reddit credentials.

- .env is ignored by git, keep secrets local.

Example:

```ini
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=windows:siads-696-research:v0.1 (by u/mads696_research)
```

## Usage

Run examples:

```bash
python news_reddit_api.py
```


This will:

- Print hot posts from r/news
- Show top comments
- Display basic user info

---
