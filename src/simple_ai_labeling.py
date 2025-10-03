"""Simple AI labeling script - Using GPT to classify Reddit posts for anxiety

Usage:
    python src/simple_ai_labeling.py --limit 1000

Configuration:
    - Set OPENAI_API_KEY in secret.env
    - Modify PROMPT below to adjust labeling requirements
    - Modify API_CALLS_LIMIT to control processing quantity
"""

import json
import os
import time
from pathlib import Path

import openai
import pandas as pd
from dotenv import load_dotenv

# =============================================================================
# Configuration Section - You can directly modify settings here
# =============================================================================

# GPT Prompt - Based on improved GAD-7 version
PROMPT = """You are an expert mental health researcher analyzing Reddit posts for anxiety-related content. 

Analyze this Reddit post and classify the anxiety level using the adapted GAD-7 scale:

POST TITLE: {title}
POST CONTENT: {text}
SUBREDDIT: r/{subreddit}

ANXIETY SEVERITY LEVELS (adapted from GAD-7):
0 - None: No anxiety content/tone detected
1 - Minimal: Neutral/coping tone; brief concern, no functional impairment
2 - Mild: Occasional/situational worry; hedged language; functioning intact
3 - Moderate: Frequent worry/rumination; some impact on focus/sleep/daily activities
4 - Severe: Strong fear/anticipation, catastrophizing, panic-like language; clear functional impairment
5 - Crisis-level: Major impairment (can't work/leave house, constant panic attacks) or safety risk

CLASSIFICATION CATEGORIES:
- ANXIETY: Post contains anxiety-related content (levels 1-5)
- NOT_ANXIETY: Post does not contain anxiety-related content (level 0)

Return ONLY a JSON object in this exact format:
{{
    "category": "ANXIETY" or "NOT_ANXIETY",
    "severity": 0-5 (based on GAD-7 adapted scale above),
    "confidence": 0.0-1.0 (your confidence in this classification)
}}"""

# API call limit - Control how many data to process
API_CALLS_LIMIT = 100  # Process all 1000 balanced samples

# Text length control - Avoid token limit and save cost
MAX_TITLE_LENGTH = 200  # Maximum title characters
MAX_TEXT_LENGTH = 1000  # Maximum text characters
MAX_TOTAL_TOKENS = 3000  # Estimated maximum tokens (including prompt + response)

# Data source file path
DATA_SOURCE = "data/processed/balanced_sample_1000.csv"

# Output file path
OUTPUT_FILE = "data/processed/simple_ai_labels.csv"

# API call interval (seconds)
API_DELAY = 1.0

# =============================================================================
# Main Function Code
# =============================================================================


def load_existing_labels():
    """Load existing labeling results"""
    output_path = Path(OUTPUT_FILE)
    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        existing_post_ids = set(existing_df["post_id"].tolist())
        print(f"Found existing label file: {len(existing_post_ids)} records")
        return existing_df, existing_post_ids
    else:
        print("No existing label file found, will create new file")
        return pd.DataFrame(), set()


def load_data(limit=None):
    """Load data, maintain original order"""
    print(f"Loading data: {DATA_SOURCE}")

    # Check file extension and load accordingly
    if DATA_SOURCE.endswith(".parquet"):
        df = pd.read_parquet(DATA_SOURCE)
    elif DATA_SOURCE.endswith(".csv"):
        df = pd.read_csv(DATA_SOURCE)
    else:
        raise ValueError(f"Unsupported file format: {DATA_SOURCE}")

    if limit:
        df = df.head(limit)
        print(f"Limited to first {limit} records")

    print(f"Loaded {len(df)} records")
    return df


def truncate_text_smart(text, max_length):
    """Smart text truncation, prioritize keeping beginning and end"""
    if len(text) <= max_length:
        return text

    # If text is too long, keep 70% from beginning and 30% from end
    separator = "...[truncated]..."
    separator_length = len(separator)

    if max_length <= separator_length:
        return text[: max_length - 3] + "..."

    available_length = max_length - separator_length
    start_length = int(available_length * 0.7)
    end_length = available_length - start_length

    if end_length > 0:
        return text[:start_length] + separator + text[-end_length:]
    else:
        return text[: max_length - 3] + "..."


def estimate_tokens(text):
    """Rough token estimation (1 token ≈ 4 characters for English, 1-2 for Chinese)"""
    return len(text) // 3  # Conservative estimate


def call_gpt_api(title, text, subreddit, api_key):
    """Call GPT API for labeling"""
    client = openai.OpenAI(api_key=api_key)

    # Smart text truncation
    title_truncated = (
        title[:MAX_TITLE_LENGTH] if len(title) > MAX_TITLE_LENGTH else title
    )
    text_truncated = truncate_text_smart(text, MAX_TEXT_LENGTH)

    # Prepare prompt
    user_prompt = PROMPT.format(
        title=title_truncated,
        text=text_truncated,
        subreddit=subreddit,
    )

    # Check total token count
    estimated_tokens = estimate_tokens(user_prompt)
    if estimated_tokens > MAX_TOTAL_TOKENS:
        print(
            f"Warning: Estimated tokens {estimated_tokens} exceeds limit {MAX_TOTAL_TOKENS}"
        )
        # Further shorten text
        text_truncated = truncate_text_smart(text, MAX_TEXT_LENGTH // 2)
        user_prompt = PROMPT.format(
            title=title_truncated,
            text=text_truncated,
            subreddit=subreddit,
        )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.1,
            max_tokens=150,
        )

        # Parse response
        content = response.choices[0].message.content.strip()

        # Clean JSON format
        if content.startswith("```json"):
            content = content.split("```json")[1].split("```")[0].strip()
        elif content.startswith("```"):
            content = content.split("```")[1].strip()

        result = json.loads(content)
        return result

    except Exception as e:
        print(f"API call failed: {e}")
        return None


def process_posts(df, api_key, limit, existing_labels_df, existing_post_ids):
    """Process post data"""
    results = []
    truncation_stats = {"title_truncated": 0, "text_truncated": 0, "total_processed": 0}
    skip_stats = {"already_labeled": 0, "new_processed": 0}

    # Limit processing count
    process_count = min(len(df), limit) if limit else len(df)

    print(f"Starting to process {process_count} posts...")
    print(
        f"Text length limits: Title {MAX_TITLE_LENGTH} chars, Text {MAX_TEXT_LENGTH} chars"
    )

    # If there are existing labels, add them to results first
    if not existing_labels_df.empty:
        results.extend(existing_labels_df.to_dict("records"))
        print(f"Loaded {len(existing_labels_df)} existing labels")

    for i, row in df.head(process_count).iterrows():
        post_id = row["post_id"]
        print(f"Processing {i+1}/{process_count}: {post_id}")

        # Check if already labeled
        if post_id in existing_post_ids:
            print(f"  Skip - Already labeled")
            skip_stats["already_labeled"] += 1
            continue

        # Extract data
        title = str(row.get("title_clean", row.get("title", "")))
        text = str(row.get("text_all", row.get("text_main", "")))
        subreddit = str(row["subreddit"])

        # Track truncation stats
        truncation_stats["total_processed"] += 1
        if len(title) > MAX_TITLE_LENGTH:
            truncation_stats["title_truncated"] += 1
        if len(text) > MAX_TEXT_LENGTH:
            truncation_stats["text_truncated"] += 1

        # Call API
        print(f"  Calling API for labeling...")
        ai_result = call_gpt_api(title, text, subreddit, api_key)
        skip_stats["new_processed"] += 1

        # Save result
        result_row = {
            "post_id": post_id,
            "subreddit": subreddit,
            "title": title,
            "text_preview": text[:200] + "..." if len(text) > 200 else text,
        }

        if ai_result:
            result_row.update(
                {
                    "ai_category": ai_result.get("category", "UNKNOWN"),
                    "ai_severity": ai_result.get("severity", 0),
                    "ai_confidence": ai_result.get("confidence", 0.0),
                }
            )
        else:
            result_row.update(
                {"ai_category": "ERROR", "ai_severity": 0, "ai_confidence": 0.0}
            )

        results.append(result_row)

        # Save progress every 10 records
        if skip_stats["new_processed"] % 10 == 0 and skip_stats["new_processed"] > 0:
            temp_df = pd.DataFrame(results)
            temp_df.to_csv(f"temp_progress_{len(results)}.csv", index=False)
            print(f"Progress saved: {len(results)} records")

        # API call interval
        time.sleep(API_DELAY)

    # Display processing statistics
    print(f"\n=== Processing Statistics ===")
    print(f"Skipped (already labeled): {skip_stats['already_labeled']} records")
    print(f"New processed (API calls): {skip_stats['new_processed']} records")
    print(f"Total results: {len(results)} records")

    # Display truncation statistics
    if truncation_stats["total_processed"] > 0:
        print(f"\n=== Text Truncation Statistics ===")
        print(f"New processed total: {truncation_stats['total_processed']} records")
        print(
            f"Title truncated: {truncation_stats['title_truncated']} records ({truncation_stats['title_truncated']/truncation_stats['total_processed']*100:.1f}%)"
        )
        print(
            f"Text truncated: {truncation_stats['text_truncated']} records ({truncation_stats['text_truncated']/truncation_stats['total_processed']*100:.1f}%)"
        )

    return pd.DataFrame(results)


def main():
    """Main function"""
    # Load environment variables
    load_dotenv("secret.env")
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key or api_key == "your_openai_api_key_here":
        print("Error: Please set OPENAI_API_KEY in secret.env")
        return

    print("=== Simple AI Labeling Tool ===")
    print(f"Data source: {DATA_SOURCE}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"API call limit: {API_CALLS_LIMIT}")
    print(f"API call interval: {API_DELAY} seconds")
    print()

    # Load existing labels
    existing_labels_df, existing_post_ids = load_existing_labels()

    # Load data
    df = load_data()

    # Process data
    results_df = process_posts(
        df, api_key, API_CALLS_LIMIT, existing_labels_df, existing_post_ids
    )

    # Save results
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print(f"\n=== Processing Complete ===")
    print(f"Processed {len(results_df)} records")
    print(f"Results saved to: {output_path}")

    # Display statistics
    if len(results_df) > 0:
        print(f"\nLabeling Results Statistics:")
        print(results_df["ai_category"].value_counts())
        print(f"Average confidence: {results_df['ai_confidence'].mean():.3f}")


if __name__ == "__main__":
    main()
