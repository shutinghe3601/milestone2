"""
Configuration constants and default values for NRC Emotion Lexicon processing.
"""

from typing import Dict, List

# Default emotions from NRC Emotion Lexicon
DEFAULT_EMOTIONS: List[str] = [
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "trust",
    "negative",
    "positive",
]

# Default weights for anxiety scoring (adjusted to better handle positive emotions)
DEFAULT_WEIGHTS: Dict[str, float] = {
    "fear": 1.0,
    "sadness": 0.7,
    "anger": 0.6,
    "disgust": 0.6,
    "anticipation": 0.2,  # Reduced: anticipation can be positive
    "surprise": 0.1,  # Reduced: surprise can be positive
    "joy": -1.0,  # Increased negative weight
    "trust": -0.8,  # Increased negative weight
    "negative": 0.3,
    "positive": -0.5,  # Increased negative weight
}

# Negation and intensifier words
NEGATION_WORDS = {"no", "not", "never", "without", "hardly", "none", "neither", "nor"}

INTENSIFIER_WORDS = {
    "very",
    "extremely",
    "super",
    "so",
    "highly",
    "really",
    "quite",
    "totally",
}

# Common contractions for text preprocessing
CONTRACTIONS = {
    "don't": "do not",
    "can't": "cannot",
    "won't": "will not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "couldn't": "could not",
    "i'm": "i am",
    "you're": "you are",
    "it's": "it is",
    "that's": "that is",
    "we're": "we are",
    "they're": "they are",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'll": "i will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "we'll": "we will",
    "they'll": "they will",
}

# Default lexicon file path
DEFAULT_LEXICON_PATH = "./data/raw/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"

# Default statistical parameters for scaling
DEFAULT_STATISTICAL_PARAMS = {
    "median": 0.5,
    "mad": 0.3,
}

# Anxiety label thresholds (adjusted to reduce false positives)
ANXIETY_THRESHOLDS = [-float("inf"), -0.05, 0.05, 0.15, 0.30, float("inf")]
