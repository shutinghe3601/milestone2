"""
Data analysis utilities for Reddit mental health posts analysis.

This module provides helper functions for data loading, analysis, and visualization
to keep notebooks clean and functions reusable across different analyses.
"""

# TODO
#clean and document better

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load raw data from JSONL file for quality check.

    Args:
        path: Path to JSONL file

    Returns:
        List of dictionaries containing the loaded data
    """
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def analyze_dataframe(df: pd.DataFrame, name: str = "DATASET") -> None:
    """Analyze any dataframe with pretty formatting.

    Args:
        df: DataFrame to analyze
        name: Name to display in the analysis header
    """
    print(f"\n{'='*50}")
    print(f" {name} ANALYSIS")
    print(f"{'='*50}")

    print(f" Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    print(f"\n Columns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")

    print(f"\n Data Types Summary:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  • {dtype}: {count} columns")

    print(f"\n Missing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100)

    has_missing = missing[missing > 0]
    if len(has_missing) == 0:
        print("   No missing values found!")
    else:
        print(f"   {len(has_missing)} columns have missing values:")
        for col in has_missing.index:
            count = missing[col]
            pct = missing_pct[col]
            print(f"     • {col:<25}: {count:>6,} ({pct:>5.1f}%)")

    print(f"{'='*50}")


def analyze_subreddit_distribution(posts_df: pd.DataFrame) -> None:
    """Analyze and visualize subreddit distribution with pretty formatting.

    Args:
        posts_df: DataFrame containing posts with 'subreddit' column
    """
    subreddit_counts = posts_df['subreddit'].value_counts()
    subreddit_pct = (subreddit_counts / len(posts_df) * 100)

    print(f"\n{'='*50}")
    print(f" SUBREDDIT DISTRIBUTION")
    print(f"{'='*50}")
    print(f" Total posts across {len(subreddit_counts)} subreddits:")

    for subreddit, count in subreddit_counts.items():
        pct = subreddit_pct[subreddit]
        bar_length = int(pct / 2)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        print(f"  {subreddit:<20} │{bar}│ {count:>6,} ({pct:>5.1f}%)")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Bar plot
    subreddit_counts.plot(kind='bar', ax=ax1, color='skyblue', edgecolor='navy')
    ax1.set_title('Posts per Subreddit', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Subreddit')
    ax1.set_ylabel('Number of Posts')
    ax1.tick_params(axis='x', rotation=45)

    # Pie chart
    colors = plt.cm.Set3(np.linspace(0, 1, len(subreddit_counts)))
    wedges, texts, autotexts = ax2.pie(subreddit_counts.values,
                                       labels=subreddit_counts.index,
                                       autopct='%1.1f%%',
                                       colors=colors,
                                       startangle=90)
    ax2.set_title('Subreddit Distribution', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()
    print(f"{'='*50}")


def analyze_text_lengths(posts_df: pd.DataFrame) -> None:
    """Analyze text length distributions and create visualizations.

    Args:
        posts_df: DataFrame containing text columns (title, selftext, fulltext)
    """
    # Ensure a unified fulltext column exists
    CANDIDATE_TEXT_COLS = ["fulltext", "processed_full_text", "cleaned_text", "selftext", "text"]

    if "fulltext" not in posts_df.columns:
        # Prefer explicit title+selftext if available
        if ("title" in posts_df.columns) or ("selftext" in posts_df.columns):
            posts_df["fulltext"] = (
                posts_df.get("title", "").fillna("").astype(str) + " " +
                posts_df.get("selftext", "").fillna("").astype(str)
            ).str.strip()
        else:
            # Otherwise fall back to the first available candidate
            src = next((c for c in CANDIDATE_TEXT_COLS if c in posts_df.columns), None)
            posts_df["fulltext"] = posts_df[src].fillna("").astype(str) if src else ""

    # Safe length helper
    def safe_len(x):
        if pd.isna(x):
            return 0
        if isinstance(x, list):
            return len(" ".join(str(t) for t in x))
        return len(str(x))

    # Compute lengths only for columns that exist
    for base in ["title", "selftext", "fulltext"]:
        if base in posts_df.columns:
            posts_df[f"{base}_len"] = posts_df[base].apply(safe_len)

    # Print stats for whichever we have
    print("Text Length Statistics:")
    for base in ["title", "selftext", "fulltext"]:
        col = f"{base}_len"
        if col in posts_df.columns:
            print(f"{base.capitalize()} length - Mean: {posts_df[col].mean():.1f}, Median: {posts_df[col].median():.1f}")

    # Dashboard with bar charts
    length_cols = [c for c in ["title_len", "selftext_len", "fulltext_len"] if c in posts_df.columns]

    if length_cols:
        fig, axes = plt.subplots(1, len(length_cols), figsize=(6*len(length_cols), 5))
        if len(length_cols) == 1:
            axes = [axes]

        for i, col in enumerate(length_cols):
            axes[i].hist(posts_df[col], bins=50, edgecolor="black", alpha=0.7)
            axes[i].set_title(col.replace("_", " ").title())
            axes[i].set_xlabel("Character Count")
            axes[i].set_ylabel("Frequency")

        plt.tight_layout()
        plt.show()

    # Boxplot by subreddit
    if ("subreddit" in posts_df.columns) and ("fulltext_len" in posts_df.columns):
        plt.figure(figsize=(12, 6))

        sns.boxplot(data=posts_df, x="subreddit", y="fulltext_len")
        plt.title("Full Text Length by Subreddit")
        plt.xlabel("Subreddit")
        plt.ylabel("Character Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


def print_linkage_analysis(posts_df: pd.DataFrame, comments_df: pd.DataFrame) -> None:
    """Print linkage analysis between posts and comments.

    Args:
        posts_df: DataFrame containing posts
        comments_df: DataFrame containing comments with 'link_id' column
    """
    linked_posts = comments_df['link_id'].str.replace('t3_', '').nunique()
    print(f"\n{'='*50}")
    print(f"🔗 LINKAGE ANALYSIS")
    print(f"{'='*50}")
    print(f" Posts with comments: {linked_posts:,} / {len(posts_df):,} ({linked_posts/len(posts_df)*100:.1f}%)")
    print(f" Average comments per post: {len(comments_df) / len(posts_df):.1f}")
    print(f"{'='*50}")


def setup_pandas_display() -> None:
    """Set up pandas display options for better output formatting."""
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 120)
    pd.set_option('display.precision', 2)
    pd.set_option('display.float_format', '{:.2f}'.format)
