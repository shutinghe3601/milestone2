"""
Simple NRC Emotion Lexicon Weak Labeling - Refactored Version
Implements emotion counting, anxiety scoring, and ordinal labeling (1-5 scale).
"""

from typing import Dict, List, Optional

from nrc_emotion.config import DEFAULT_EMOTIONS, DEFAULT_LEXICON_PATH, DEFAULT_WEIGHTS
from nrc_emotion.emotion_analyzer import (
    apply_negation_intensifier,
    apply_statistical_scaling,
    apply_threshold_scaling,
    compute_emotion_counts_and_anxiety,
    compute_statistics,
    get_anxiety_label_threshold,
)
from nrc_emotion.lexicon_loader import find_emotion_matches, load_lexicon
from nrc_emotion.text_preprocessor import preprocess_text


def label_text(
    text: str,
    emotions: Optional[List[str]] = None,
    lexicon_path: str = DEFAULT_LEXICON_PATH,
    weights: Optional[Dict[str, float]] = None,
    lowercase: bool = True,
    lemmatize: bool = True,
    negation_window: int = 3,
    intensifier_weight: float = 1.5,
    # Scaling method: "threshold", "statistical", or "both"
    scaling_method: str = "threshold",
    statistical_params: Optional[Dict[str, float]] = None,
    # Debug parameters
    verbose: bool = False,
    return_intermediate: bool = False,
    show_stats: bool = False,
    word_emotion_mapping: bool = False,
    matched_words_only: bool = False,
    expand_contractions: bool = False,
    remove_urls: bool = False,
    custom_emotions: Optional[List[str]] = None,
    custom_weights: Optional[Dict[str, float]] = None,
) -> Dict:
    """
    Label text with emotion counts and anxiety scores.

    Args:
        text: Input text to analyze
        emotions: List of emotions to analyze (defaults to DEFAULT_EMOTIONS)
        lexicon_path: Path to the NRC emotion lexicon file
        weights: Emotion weights for anxiety scoring (defaults to DEFAULT_WEIGHTS)
        lowercase: Whether to convert text to lowercase
        lemmatize: Whether to lemmatize words
        negation_window: Number of tokens affected by negation
        intensifier_weight: Multiplier for intensified emotions

        # Scaling method options:
        scaling_method: "threshold" (1-5 labels), "statistical" (0-1 scores), or "both"
        statistical_params: Dict with "median" and "mad" for statistical scaling

        # Debug parameters:
        verbose: Print detailed processing information
        return_intermediate: Return intermediate processing results
        show_stats: Show emotion matching statistics
        word_emotion_mapping: Return mapping of words to emotions
        matched_words_only: Only show words that matched emotions
        expand_contractions: Expand contractions (don't -> do not)
        remove_urls: Remove URL patterns from text
        custom_emotions: Override default emotions list
        custom_weights: Override default weights

    Returns:
        Dictionary with n_tokens, emo_counts, anxiety_score_raw, anxiety_score_norm
        Additional fields based on scaling_method:
        - "threshold": includes anxiety_label (1-5)
        - "statistical": includes anxiety_score_statistical (0-1)
        - "both": includes both anxiety_label and anxiety_score_statistical
    """
    # Handle custom overrides for debugging
    if custom_emotions is not None:
        emotions = custom_emotions.copy()
    elif emotions is None:
        emotions = DEFAULT_EMOTIONS.copy()

    if custom_weights is not None:
        weights = custom_weights.copy()
    elif weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    # Initialize debug info
    debug_info = {}

    if verbose:
        print(f"Processing text: '{text[:100]}{'...' if len(text) > 100 else ''}'")
        print(f"Using emotions: {emotions}")
        print(f"Using weights: {weights}")

    # Load lexicon
    lexicon = load_lexicon(lexicon_path, emotions)

    if verbose:
        for emotion in emotions:
            print(f"Loaded {len(lexicon.get(emotion, set()))} words for {emotion}")

    # Preprocess text
    processed_text, tokens = preprocess_text(
        text,
        lowercase=lowercase,
        lemmatize=lemmatize,
        expand_contractions_flag=expand_contractions,
        remove_urls_flag=remove_urls,
    )
    n_tokens = len(tokens)

    if verbose and processed_text != text:
        print(
            f"Text after preprocessing: '{processed_text[:100]}{'...' if len(processed_text) > 100 else ''}'"
        )

    if verbose:
        print(f"Tokens ({n_tokens}): {tokens}")

    debug_info["original_text"] = text
    debug_info["processed_text"] = processed_text
    debug_info["tokens"] = tokens

    if n_tokens == 0:
        return {
            "n_tokens": 0,
            "emo_counts": {emotion: 0 for emotion in emotions},
            "anxiety_score_raw": 0.0,
            "anxiety_score_norm": 0.0,
        }

    # Find emotion matches
    emotion_matches, word_emotions, matched_words = find_emotion_matches(
        tokens, lexicon, emotions
    )

    if verbose:
        print(f"Found {len(emotion_matches)} tokens with emotion matches")
        if emotion_matches:
            print(
                "Emotion matches:",
                {tokens[i]: emos for i, emos in emotion_matches.items()},
            )

    debug_info["emotion_matches"] = emotion_matches
    debug_info["word_emotions"] = word_emotions
    debug_info["matched_words"] = list(matched_words)

    # Apply negation and intensifier rules
    emotion_scores = apply_negation_intensifier(
        tokens, emotion_matches, negation_window, intensifier_weight
    )

    if verbose:
        print(f"Emotion scores after negation/intensifier: {emotion_scores}")

    debug_info["emotion_scores_raw"] = emotion_scores.copy()

    # Convert to integer counts and compute anxiety score
    emo_counts, anxiety_score_raw = compute_emotion_counts_and_anxiety(
        emotion_scores, emotions, weights
    )

    # Handle different scaling methods
    result = {
        "n_tokens": n_tokens,
        "emo_counts": emo_counts,
        "anxiety_score_raw": anxiety_score_raw,
    }

    if scaling_method in ["threshold", "both"]:
        # Current threshold-based normalization
        anxiety_score_norm = apply_threshold_scaling(anxiety_score_raw, n_tokens)
        result["anxiety_score_norm"] = anxiety_score_norm
        result["anxiety_label"] = get_anxiety_label_threshold(anxiety_score_norm)

        if verbose:
            print(f"Anxiety score (threshold norm): {anxiety_score_norm:.3f}")
            print(f"Anxiety label (1-5): {result['anxiety_label']}")

    if scaling_method in ["statistical", "both"]:
        # Statistical normalization (MAD + Z-score + Sigmoid)
        anxiety_score_statistical = apply_statistical_scaling(
            anxiety_score_raw, statistical_params
        )
        result["anxiety_score_statistical"] = anxiety_score_statistical

        if verbose:
            print(f"Anxiety score (statistical): {anxiety_score_statistical:.3f}")

    # For backward compatibility, always include anxiety_score_norm
    if "anxiety_score_norm" not in result:
        result["anxiety_score_norm"] = result["anxiety_score_statistical"]

    if verbose:
        print(f"Final emotion counts: {emo_counts}")
        print(f"Anxiety score (raw): {anxiety_score_raw:.3f}")

    # Add debug information if requested
    if return_intermediate:
        result["debug_info"] = debug_info

    if word_emotion_mapping and not return_intermediate:
        result["word_emotions"] = word_emotions

    if matched_words_only and not return_intermediate:
        result["matched_words"] = list(matched_words)

    if show_stats:
        stats = compute_statistics(n_tokens, matched_words, emo_counts, emotions)
        if return_intermediate:
            debug_info["stats"] = stats
        else:
            result["stats"] = stats

    return result


def run_demo(lexicon_path: str = DEFAULT_LEXICON_PATH):
    """
    Run demo with built-in sample texts.
    """
    demo_texts = [
        "I feel terrified and keep panicking about tomorrow's exam.",
        "I am not calm; I'm extremely anxious and scared.",
        "What a beautiful sunny day! I feel so happy and excited.",
        "I'm feeling okay, nothing special today.",
        "This is absolutely disgusting and makes me furious!",
    ]

    print("NRC Emotion Lexicon Demo (Refactored Version)")
    print("=" * 60)

    for i, text in enumerate(demo_texts, 1):
        print(f"\nExample {i}: {text}")
        result = label_text(text, lexicon_path=lexicon_path)
        anxiety_label = get_anxiety_label_threshold(result["anxiety_score_norm"])

        print(f"Tokens: {result['n_tokens']}")
        print(f"Emotion counts: {result['emo_counts']}")
        print(f"Anxiety score (raw): {result['anxiety_score_raw']:.3f}")
        print(f"Anxiety score (norm): {result['anxiety_score_norm']:.3f}")
        print(f"Anxiety label (1-5): {anxiety_label}")


# Example usage
if __name__ == "__main__":
    # Demo mode
    print("Running demo mode...")
    run_demo()

    print(f"\nFor comprehensive debug examples and parameter testing, run:")
    print(f"   python docs/test/week_label_examples.py")
