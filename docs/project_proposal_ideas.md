# Project Introduction

# Similar project introduction

# Part A (Supervised learning)

## What is the dataset you propose to use? Provide a link to the dataset.
The dataset we will be using the pulled content from Reddit through Reddit API. the pulled content is save as json format, including features like id, subtitle...

## What are the question(s) about the dataset’s structure you want to answer, or goal to achieve?
Here we are trying to identify the patterns of posts that trigger anxiety (we may need to define anxiety).

## What data manipulation will be necessary for this dataset to prepare it?
TBD (will be defined after unsupervised exploration informs features/labels)

## Specify unsupervised learning approaches and feature representations that are appropriate for this problem.

## Are there external datasets or tools that you might incorporate to help with the problem?

## How will you evaluate the quality of your results?

## Describe visualizations that would be appropriate as part of evaluating the effectiveness of your methods or characterizing the structure in the dataset.


# Part B (Unsupervised learning) 

## What is the dataset you propose to use? Provide a link to the dataset.
We will use Reddit content pulled via the Reddit API, saved as JSON Lines. Each line represents a post with fields such as `id`, `subreddit`, `title`, `selftext`, `author`, `score`, `upvote_ratio`, `url`, `num_comments`, `created_utc`, `permalink`, `link_flair_text`, `is_self`, and an optional `top_comments` list (each with `id`, `body`, `author`, `score`, `created_utc`). Example file: `data/raw/news_20250829_210623.jsonl`.

## What are the question(s) about the dataset’s structure you want to answer, or goal to achieve?
Identify recurring patterns in Reddit posts about AI that are associated with anxiety, and characterize those themes (e.g., job loss, deepfakes/privacy, existential risk). The goal is to surface interpretable topics/clusters and rank them by their anxiety intensity.

## What data manipulation will be necessary for this dataset to prepare it?
- Load JSONL line-by-line; keep metadata (subreddit, created_utc, scores).
- Filter to English posts about AI (keyword list), and combine `title + selftext + top comments` (skip `[removed]/[deleted]`).
- Clean text: lowercase, remove URLs/usernames, normalize whitespace, and keep useful bigrams (e.g., “job loss”).
- Deduplicate: exact hash of cleaned text and near-duplicate removal via 3‑gram TF–IDF cosine similarity (>0.9).
- Quality filters: drop very short/spammy docs; convert `created_utc` to datetime for time-based evaluation.

LDA/NMF-specific
- Tokenize + lemmatize; remove stopwords and very rare/common terms.
- Features: term counts for LDA; TF–IDF (unigrams+bigrams, L2-normalized) for NMF with `min_df` and `max_df` set.
- Persist artifacts: save vectorizer, vocabulary, and sparse matrices for reuse.

BERTopic-specific
- Compute sentence embeddings with a compact model (e.g., MiniLM) and cache them.
- Optionally truncate long docs (first 512–1024 tokens) for speed.
- Apply UMAP reduction and HDBSCAN clustering; keep mapping of post → topic and mark noise/outliers.

## Specify unsupervised learning approaches and feature representations that are appropriate for this problem.
- Approaches: LDA/NMF (word-based topic modeling) and BERTopic (embedding-based clustering with c‑TF‑IDF labeling).
- Features: term counts (for LDA), TF–IDF with unigrams+bigrams (for NMF), sentence embeddings (for BERTopic), and optional keyphrases to aid interpretation.

## Are there external datasets or tools that you might incorporate to help with the problem?
- Great Lakes GPU/large RAM to speed up embeddings/BERTopic and larger runs.
- Optional emotion lexicons (e.g., fear/anxiety lists) for unsupervised anxiety scoring.

## How will you evaluate the quality of your results?
- Topic coherence (e.g., NPMI/coherence) and human readability of top words.
- Stability across seeds/subsamples and reasonable coverage (few “misc/noise” topics).
- Alignment with anxiety: clusters/topics with high average anxiety score should read as anxious on inspection.

## Describe visualizations that would be appropriate as part of evaluating the effectiveness of your methods or characterizing the structure in the dataset.
- Top-words bar charts per topic to verify interpretability.
- Topic × anxiety heatmap to highlight anxiety-heavy themes.
- UMAP scatter colored by topic/cluster to assess separability and noise.

# Team Planning:

## Indicate the specific contributions that each team member will make to the project. (Include a rough timeline.)
