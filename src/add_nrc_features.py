#!/usr/bin/env python3
"""
Add NRC Emotion Lexicon anxiety scores to the dataset as metadata features.
This will create a new version of the dataset with NRC scores included.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Import from the same directory
from weak_label_nrc import label_text

def add_nrc_scores(input_file: str, output_file: str):
    """Add NRC anxiety scores to dataset."""

    print(f"Loading dataset from {input_file}")

    # Load the dataset
    if input_file.endswith('.parquet'):
        df = pd.read_parquet(input_file)
    else:
        df = pd.read_csv(input_file)

    print(f"Processing {len(df)} posts...")

    # Calculate NRC scores for each post
    nrc_scores = []
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"Processing post {idx+1}/{len(df)}")

        text = row.get('text_all', '') or row.get('text_main', '') or ''

        try:
            # Use absolute path from project root
            lexicon_path = Path(__file__).parent.parent / "data" / "raw" / "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
            result = label_text(str(text), lexicon_path=str(lexicon_path))
            nrc_scores.append(result['anxiety_score_norm'])  # Uses corrected sigmoid normalization
        except Exception as e:
            print(f"Error processing post {idx}: {e}")
            nrc_scores.append(0.5)  # Neutral score (0.5) for sigmoid normalization

    # Add NRC scores to dataframe
    df['anxiety_score'] = nrc_scores

    print(f"NRC scores calculated. Range: {min(nrc_scores):.3f} to {max(nrc_scores):.3f}")
    print(f"Mean NRC score: {np.mean(nrc_scores):.3f}")

    # Save updated dataset
    if output_file.endswith('.parquet'):
        df.to_parquet(output_file, index=False)
    else:
        df.to_csv(output_file, index=False)

    print(f"Updated dataset saved to {output_file}")
    return df

if __name__ == "__main__":
    # Add NRC scores to the main dataset
    # Use absolute paths from project root
    input_file = str(Path(__file__).parent.parent / "data" / "processed" / "reddit_anxiety_v1.parquet")
    output_file = str(Path(__file__).parent.parent / "data" / "processed" / "reddit_anxiety_v1_with_nrc.parquet")

    df = add_nrc_scores(input_file, output_file)

    print("\nSample of NRC scores:")
    print(df[['post_id', 'text_all', 'anxiety_score']].head())
