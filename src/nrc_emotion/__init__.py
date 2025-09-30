"""
NRC Emotion Lexicon Analysis Package

This package contains modules for analyzing text using the NRC Emotion Lexicon,
including emotion detection, anxiety scoring, and text preprocessing.
"""

from .config import (
    ANXIETY_THRESHOLDS,
    CONTRACTIONS,
    DEFAULT_EMOTIONS,
    DEFAULT_LEXICON_PATH,
    DEFAULT_WEIGHTS,
    INTENSIFIER_WORDS,
    NEGATION_WORDS,
)
from .emotion_analyzer import (
    apply_negation_intensifier,
    apply_statistical_scaling,
    apply_threshold_scaling,
    compute_emotion_counts_and_anxiety,
    compute_statistics,
    get_anxiety_label_threshold,
)
from .lexicon_loader import find_emotion_matches, get_lexicon_stats, load_lexicon
from .text_preprocessor import (
    expand_contractions,
    lemmatize_tokens,
    preprocess_text,
    remove_urls,
    tokenize_text,
)

__all__ = [
    # Config
    "DEFAULT_EMOTIONS",
    "DEFAULT_LEXICON_PATH",
    "DEFAULT_WEIGHTS",
    "ANXIETY_THRESHOLDS",
    "NEGATION_WORDS",
    "INTENSIFIER_WORDS",
    "CONTRACTIONS",
    # Emotion Analysis
    "apply_negation_intensifier",
    "apply_statistical_scaling",
    "apply_threshold_scaling",
    "compute_emotion_counts_and_anxiety",
    "compute_statistics",
    "get_anxiety_label_threshold",
    # Lexicon Loading
    "load_lexicon",
    "find_emotion_matches",
    "get_lexicon_stats",
    # Text Preprocessing
    "preprocess_text",
    "expand_contractions",
    "remove_urls",
    "tokenize_text",
    "lemmatize_tokens",
]
