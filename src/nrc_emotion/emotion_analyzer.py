"""
Emotion analysis and scoring utilities.
"""

import math
from typing import Dict, List, Optional

from .config import (
    ANXIETY_THRESHOLDS,
    DEFAULT_STATISTICAL_PARAMS,
    DEFAULT_WEIGHTS,
    INTENSIFIER_WORDS,
    NEGATION_WORDS,
)


def apply_negation_intensifier(
    tokens: List[str],
    emotion_matches: Dict[int, List[str]],
    negation_window: int = 3,
    intensifier_weight: float = 1.5,
) -> Dict[str, float]:
    """
    Apply negation and intensifier rules to emotion matches.

    Args:
        tokens: List of processed tokens
        emotion_matches: Dictionary mapping token index -> list of matched emotions
        negation_window: Number of tokens affected by negation
        intensifier_weight: Multiplier for intensified emotions

    Returns:
        Dictionary mapping emotion -> weighted count
    """
    emotion_scores = {}

    # Track negation positions
    negation_positions = []
    for i, token in enumerate(tokens):
        if token in NEGATION_WORDS:
            negation_positions.append(i)

    # Track intensifier positions
    intensifier_positions = []
    for i, token in enumerate(tokens):
        if token in INTENSIFIER_WORDS:
            intensifier_positions.append(i)

    # Process emotion matches
    for token_idx, emotions in emotion_matches.items():
        for emotion in emotions:
            # Base weight
            weight = 1.0

            # Check for intensifiers (look backwards within window)
            for int_pos in intensifier_positions:
                if int_pos < token_idx and token_idx - int_pos <= negation_window:
                    weight *= intensifier_weight
                    break

            # Check for negation (look backwards within window)
            negated = False
            for neg_pos in negation_positions:
                if neg_pos < token_idx and token_idx - neg_pos <= negation_window:
                    negated = True
                    break

            # Apply negation
            if negated:
                weight *= -1

            # Add to emotion scores
            if emotion not in emotion_scores:
                emotion_scores[emotion] = 0.0
            emotion_scores[emotion] += weight

    return emotion_scores


def compute_emotion_counts_and_anxiety(
    emotion_scores: Dict[str, float],
    emotions: List[str],
    weights: Optional[Dict[str, float]] = None,
) -> tuple[Dict[str, int], float]:
    """
    Convert emotion scores to counts and compute raw anxiety score.

    Args:
        emotion_scores: Dictionary mapping emotion -> weighted score
        emotions: List of emotions to process
        weights: Emotion weights for anxiety scoring

    Returns:
        Tuple of (emotion_counts, anxiety_score_raw)
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    emo_counts = {}
    anxiety_score_raw = 0.0

    for emotion in emotions:
        count = max(
            0, round(emotion_scores.get(emotion, 0))
        )  # Ensure non-negative counts
        emo_counts[emotion] = count

        # Add to anxiety score (using the raw float score, not the rounded count)
        if emotion in weights:
            anxiety_score_raw += emotion_scores.get(emotion, 0) * weights[emotion]

    return emo_counts, anxiety_score_raw


def get_anxiety_label_threshold(anxiety_score_norm: float) -> int:
    """
    Convert normalized anxiety score to 1-5 label using fixed thresholds.

    Args:
        anxiety_score_norm: Normalized anxiety score

    Returns:
        Anxiety label (1-5)
    """
    for i in range(len(ANXIETY_THRESHOLDS) - 1):
        if ANXIETY_THRESHOLDS[i] <= anxiety_score_norm < ANXIETY_THRESHOLDS[i + 1]:
            return i + 1
    return 5  # fallback


def apply_threshold_scaling(anxiety_score_raw: float, n_tokens: int) -> float:
    """
    Apply threshold-based normalization to anxiety score with bounds.

    Args:
        anxiety_score_raw: Raw anxiety score
        n_tokens: Number of tokens in text

    Returns:
        Normalized anxiety score bounded to [-1.0, 1.0] range
    """
    normalized = anxiety_score_raw / (max(1, n_tokens) ** 0.7)
    # Apply bounds to prevent extreme values
    return max(-1.0, min(1.0, normalized))


def apply_statistical_scaling(
    anxiety_score_raw: float, statistical_params: Optional[Dict[str, float]] = None
) -> float:
    """
    Apply statistical scaling (MAD + Z-score + Sigmoid) like text_process.ipynb.

    Args:
        anxiety_score_raw: Raw anxiety score
        statistical_params: Dict with "median" and "mad" values
                          If None, uses default parameters

    Returns:
        Anxiety score in 0-1 range using statistical normalization
    """
    if statistical_params is None:
        statistical_params = DEFAULT_STATISTICAL_PARAMS

    median = statistical_params.get("median", 0.5)
    mad = statistical_params.get("mad", 0.3)

    # Prevent division by zero
    if mad <= 1e-6:
        mad = 1e-6

    # Apply MAD-based Z-score transformation
    z = (anxiety_score_raw - median) / (1.4826 * mad)

    # Apply sigmoid function to get 0-1 range
    statistical_score = 1.0 / (1.0 + math.exp(-z))

    return statistical_score


def compute_statistics(
    n_tokens: int,
    matched_words: set,
    emo_counts: Dict[str, int],
    emotions: List[str],
) -> Dict:
    """
    Compute emotion matching statistics.

    Args:
        n_tokens: Number of tokens
        matched_words: Set of words that matched emotions
        emo_counts: Dictionary of emotion counts
        emotions: List of emotions

    Returns:
        Dictionary with statistics
    """
    return {
        "total_words": n_tokens,
        "matched_words": len(matched_words),
        "match_rate": len(matched_words) / max(1, n_tokens),
        "emotions_found": [emo for emo in emotions if emo_counts.get(emo, 0) > 0],
        "top_emotions": sorted(
            [(emo, count) for emo, count in emo_counts.items() if count > 0],
            key=lambda x: x[1],
            reverse=True,
        )[:3],
    }
