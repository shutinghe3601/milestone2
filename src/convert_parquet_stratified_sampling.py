"""Utilities for converting Parquet data and producing stratified samples.

Using example:
python src/convert_parquet_stratified_sampling.py --sample-b 600 --sample-f 50

Note: You need to update the sample-output path if sampling data for AI labeling.
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd


def convert_parquet_to_csv(parquet_path: Path, csv_path: Path) -> None:
    """Convert a Parquet file to CSV."""
    try:
        data = pd.read_parquet(parquet_path)
    except ImportError as exc:  # pandas raises this when the Parquet engine is missing
        raise ImportError(
            "pandas requires either pyarrow or fastparquet to read Parquet files. "
            "Install one of those packages and rerun the script."
        ) from exc

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(csv_path, index=False)


def stratified_sampling(b: int, f: int, csv_path: Path) -> pd.DataFrame:
    """Perform subreddit-stratified sampling based on total and floor quotas."""
    data = pd.read_csv(csv_path)

    B = b
    F = f
    unique_subreddits = data["subreddit"].unique()
    K = len(unique_subreddits)

    num_samples_dict = {}
    for subreddit in unique_subreddits:
        subreddit_rows = data[data["subreddit"] == subreddit]
        N_i = len(subreddit_rows)
        num_samples = F + round((B - F * K) * N_i / data.shape[0])
        num_samples_dict[subreddit] = num_samples

    sample_frames = []
    for subreddit, sample_size in num_samples_dict.items():
        available_rows = len(data[data["subreddit"] == subreddit])
        if available_rows == 0:
            continue
        adjusted_size = min(sample_size, available_rows)
        print(f"{subreddit}: {adjusted_size}")
        sample = data[data["subreddit"] == subreddit].sample(adjusted_size, random_state=42)
        sample_frames.append(sample)

    if not sample_frames:
        raise ValueError("No data available for stratified sampling.")

    sample_data = pd.concat(sample_frames).reset_index(drop=True)
    print(sample_data.head(10))
    print(sample_data.shape)
    return sample_data


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description="Convert a Parquet file to CSV and create a stratified sample."
    )
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--input",
        type=Path,
        default=default_root / "data/processed/reddit_anxiety_v1.parquet",
        help="Path to the input Parquet file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_root / "data/processed/reddit_anxiety_v1.csv",
        help="Path where the CSV file will be written",
    )
    parser.add_argument(
        "--sample-output",
        type=Path,
        default=default_root / "data/processed/sample_for_human_labeling.csv",
        help="Path where the stratified sample CSV will be written",
    )
    parser.add_argument(
        "--sample-b",
        dest="sample_b",
        type=int,
        default=50,
        help="Total number of rows to include in the stratified sample",
    )
    parser.add_argument(
        "--sample-f",
        dest="sample_f",
        type=int,
        default=5,
        help="Minimum number of rows per subreddit in the stratified sample",
    )
    return parser.parse_args()


def main(b: int, f: int, parquet_path: Path, csv_path: Path, sample_path: Path) -> None:
    """Convert Parquet to CSV and produce a stratified sample using the provided parameters."""
    convert_parquet_to_csv(parquet_path, csv_path)
    sample = stratified_sampling(b, f, csv_path)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(sample_path, index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args.sample_b, args.sample_f, args.input, args.output, args.sample_output)
