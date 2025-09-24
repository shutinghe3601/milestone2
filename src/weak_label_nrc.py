"""
Simple NRC Emotion Lexicon Weak Labeling - String Input Only
Implements emotion counting, anxiety scoring, and ordinal labeling (1-5 scale).
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Set

# Try to import nltk for lemmatization (optional dependency)
try:
    import nltk
    from nltk.stem import WordNetLemmatizer

    NLTK_AVAILABLE = True

    # Automatically download required NLTK data if not available
    try:
        test_lemmatizer = WordNetLemmatizer()
        test_lemmatizer.lemmatize("testing")
    except:
        print("Downloading NLTK data...")
        try:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            print("NLTK data downloaded successfully")
        except:
            print("Failed to download NLTK data, using simple normalization")
            NLTK_AVAILABLE = False

except ImportError:
    NLTK_AVAILABLE = False
    WordNetLemmatizer = None

# Default emotions from NRC Emotion Lexicon
DEFAULT_EMOTIONS = [
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

# Default weights for anxiety scoring
DEFAULT_WEIGHTS = {
    "fear": 1.0,
    "sadness": 0.7,
    "anger": 0.6,
    "disgust": 0.6,
    "anticipation": 0.4,
    "surprise": 0.3,
    "joy": -0.5,
    "trust": -0.4,
    "negative": 0.3,
    "positive": -0.2,
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


def load_lexicon(lexicon_dir: str, emotions: List[str]) -> Dict[str, Set[str]]:
    """
    Load NRC emotion lexicon files from directory.

    Args:
        lexicon_dir: Path to directory containing emotion lexicon files
        emotions: List of emotions to load

    Returns:
        Dictionary mapping emotion -> set of words
    """
    lexicon = {}
    lexicon_path = Path(lexicon_dir)

    for emotion in emotions:
        file_path = lexicon_path / f"{emotion}-NRC-Emotion-Lexicon.txt"
        if file_path.exists():
            words = set()
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and "\t" in line:
                        parts = line.split("\t")
                        if len(parts) >= 2:
                            word = parts[0].strip()
                            score = parts[1].strip()
                            # Only include words with score 1 (associated with this emotion)
                            if word and score == "1":
                                words.add(word.lower())
            lexicon[emotion] = words
        else:
            print(
                f"Warning: Lexicon file not found for emotion '{emotion}': {file_path}"
            )
            lexicon[emotion] = set()

    return lexicon


def preprocess_text(
    text: str, lowercase: bool = True, lemmatize: bool = True
) -> List[str]:
    """
    Preprocess text and return list of tokens.

    Args:
        text: Input text
        lowercase: Whether to convert to lowercase
        lemmatize: Whether to lemmatize words

    Returns:
        List of processed tokens
    """
    if not text:
        return []

    # Convert to lowercase if requested
    if lowercase:
        text = text.lower()

    # Tokenize with regex [a-z]+
    tokens = re.findall(r"[a-z]+", text, re.IGNORECASE if not lowercase else 0)

    # Lemmatize if requested
    if lemmatize:
        if NLTK_AVAILABLE:
            try:
                lemmatizer = WordNetLemmatizer()
                # Try different POS tags for better lemmatization
                lemmatized_tokens = []
                for token in tokens:
                    # Try verb lemmatization first (most common for -ing endings)
                    lemma = lemmatizer.lemmatize(token, pos="v")
                    if lemma == token:
                        # If verb didn't work, try noun
                        lemma = lemmatizer.lemmatize(token, pos="n")
                    if lemma == token:
                        # If still no change, try adjective
                        lemma = lemmatizer.lemmatize(token, pos="a")
                    # Use the result regardless (keep original if no change)
                    lemmatized_tokens.append(lemma)
                tokens = lemmatized_tokens
            except Exception as e:
                # If NLTK fails, keep original tokens
                print(f"NLTK lemmatization failed: {e}, keeping original words")
                pass  # Keep original tokens
        # If NLTK not available, keep original tokens (no fallback normalization)

    return tokens


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


def label_text(
    text: str,
    emotions: Optional[List[str]] = None,
    lexicon_dir: str = "./data/emotion_lexicon",
    weights: Optional[Dict[str, float]] = None,
    lowercase: bool = True,
    lemmatize: bool = True,
    negation_window: int = 3,
    intensifier_weight: float = 1.5,
) -> Dict:
    """
    Label text with emotion counts and anxiety scores.

    Args:
        text: Input text to analyze
        emotions: List of emotions to analyze (defaults to DEFAULT_EMOTIONS)
        lexicon_dir: Directory containing NRC emotion lexicon files
        weights: Emotion weights for anxiety scoring (defaults to DEFAULT_WEIGHTS)
        lowercase: Whether to convert text to lowercase
        lemmatize: Whether to lemmatize words
        negation_window: Number of tokens affected by negation
        intensifier_weight: Multiplier for intensified emotions

    Returns:
        Dictionary with n_tokens, emo_counts, anxiety_score_raw, anxiety_score_norm
    """
    if emotions is None:
        emotions = DEFAULT_EMOTIONS.copy()
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    # Load lexicon
    lexicon = load_lexicon(lexicon_dir, emotions)

    # Preprocess text
    tokens = preprocess_text(text, lowercase, lemmatize)
    n_tokens = len(tokens)

    if n_tokens == 0:
        return {
            "n_tokens": 0,
            "emo_counts": {emotion: 0 for emotion in emotions},
            "anxiety_score_raw": 0.0,
            "anxiety_score_norm": 0.0,
        }

    # Find emotion matches
    emotion_matches = {}
    for i, token in enumerate(tokens):
        matched_emotions = []
        for emotion in emotions:
            if token in lexicon[emotion]:
                matched_emotions.append(emotion)
        if matched_emotions:
            emotion_matches[i] = matched_emotions

    # Apply negation and intensifier rules
    emotion_scores = apply_negation_intensifier(
        tokens, emotion_matches, negation_window, intensifier_weight
    )

    # Convert to integer counts and compute anxiety score
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

    # Normalize anxiety score
    anxiety_score_norm = anxiety_score_raw / (max(1, n_tokens) ** 0.7)

    return {
        "n_tokens": n_tokens,
        "emo_counts": emo_counts,
        "anxiety_score_raw": anxiety_score_raw,
        "anxiety_score_norm": anxiety_score_norm,
    }


def get_anxiety_label_threshold(anxiety_score_norm: float) -> int:
    """
    Convert normalized anxiety score to 1-5 label using fixed thresholds.
    """
    thresholds = [-float("inf"), -0.01, 0.01, 0.05, 0.10, float("inf")]
    for i in range(len(thresholds) - 1):
        if thresholds[i] <= anxiety_score_norm < thresholds[i + 1]:
            return i + 1
    return 5  # fallback


def run_demo(lexicon_dir: str = "./data/emotion_lexicon"):
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

    print("🔍 NRC Emotion Lexicon Demo")
    print("=" * 50)

    for i, text in enumerate(demo_texts, 1):
        print(f"\nExample {i}: {text}")
        result = label_text(text, lexicon_dir=lexicon_dir)
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

    print("\n" + "=" * 60)
    print("Usage Examples:")
    print("=" * 60)

    # Example 1: Basic usage
    text = "I am feeling very anxious about the upcoming exam."
    result = label_text(text)
    anxiety_label = get_anxiety_label_threshold(result["anxiety_score_norm"])

    print(f"\nExample 1: Basic usage")
    print(f"Text: {text}")
    print(f"Result: {result}")
    print(f"Anxiety label: {anxiety_label}")

    # Example 2: Custom emotions
    custom_emotions = ["fear", "sadness", "joy"]
    result2 = label_text(text, emotions=custom_emotions)

    print(f"\nExample 2: Custom emotions {custom_emotions}")
    print(f"Text: {text}")
    print(f"Emotion counts: {result2['emo_counts']}")
    print(f"Anxiety score: {result2['anxiety_score_norm']:.3f}")

    # Example 3: Batch processing
    texts = [
        "I love this sunny day!",
        "I'm terrified of spiders.",
        "This is just a normal day.",
    ]

    print(f"\nExample 3: Batch processing")
    for i, t in enumerate(texts):
        res = label_text(t)
        label = get_anxiety_label_threshold(res["anxiety_score_norm"])
        print(f"Text {i+1}: anxiety={res['anxiety_score_norm']:.3f}, label={label}")
