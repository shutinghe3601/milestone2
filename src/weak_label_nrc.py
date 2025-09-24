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


def load_lexicon(lexicon_path: str, emotions: List[str]) -> Dict[str, Set[str]]:
    """
    Load NRC emotion lexicon from a single file.

    Args:
        lexicon_path: Path to the NRC emotion lexicon file
        emotions: List of emotions to load

    Returns:
        Dictionary mapping emotion -> set of words
    """
    lexicon = {emotion: set() for emotion in emotions}

    file_path = Path(lexicon_path)
    if not file_path.exists():
        print(f"Warning: Lexicon file not found: {lexicon_path}")
        return lexicon

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and "\t" in line:
                    parts = line.split("\t")
                    if len(parts) >= 3:
                        word = parts[0].strip()
                        emotion = parts[1].strip()
                        score = parts[2].strip()

                        # Only include words with score 1 and if emotion is in our list
                        if word and emotion in emotions and score == "1":
                            lexicon[emotion].add(word.lower())
    except Exception as e:
        print(f"Error reading lexicon file {lexicon_path}: {e}")

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
    lexicon_path: str = "./data/raw/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
    weights: Optional[Dict[str, float]] = None,
    lowercase: bool = True,
    lemmatize: bool = True,
    negation_window: int = 3,
    intensifier_weight: float = 1.5,
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
        Additional debug info if debug parameters are enabled
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

    # Enhanced text preprocessing
    processed_text = text

    # Apply additional preprocessing if requested
    if expand_contractions:
        contractions = {
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
        for contraction, expansion in contractions.items():
            processed_text = processed_text.replace(contraction, expansion)

    if remove_urls:
        import re

        # Remove common URL patterns
        url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        processed_text = re.sub(url_pattern, "", processed_text)

    if verbose and processed_text != text:
        print(
            f"Text after preprocessing: '{processed_text[:100]}{'...' if len(processed_text) > 100 else ''}'"
        )

    # Preprocess text
    tokens = preprocess_text(processed_text, lowercase, lemmatize)
    n_tokens = len(tokens)

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
    emotion_matches = {}
    word_emotions = {}  # For debugging: word -> emotions
    matched_words = set()  # For debugging: words that matched

    for i, token in enumerate(tokens):
        matched_emotions = []
        for emotion in emotions:
            if token in lexicon[emotion]:
                matched_emotions.append(emotion)
                matched_words.add(token)
        if matched_emotions:
            emotion_matches[i] = matched_emotions
            word_emotions[token] = matched_emotions

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

    if verbose:
        print(f"Final emotion counts: {emo_counts}")
        print(f"Anxiety score (raw): {anxiety_score_raw:.3f}")
        print(f"Anxiety score (normalized): {anxiety_score_norm:.3f}")

    # Build result dictionary
    result = {
        "n_tokens": n_tokens,
        "emo_counts": emo_counts,
        "anxiety_score_raw": anxiety_score_raw,
        "anxiety_score_norm": anxiety_score_norm,
    }

    # Add debug information if requested
    if return_intermediate:
        result["debug_info"] = debug_info

    if word_emotion_mapping and not return_intermediate:
        result["word_emotions"] = word_emotions

    if matched_words_only and not return_intermediate:
        result["matched_words"] = list(matched_words)

    if show_stats:
        stats = {
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
        if return_intermediate:
            debug_info["stats"] = stats
        else:
            result["stats"] = stats

    return result


def get_anxiety_label_threshold(anxiety_score_norm: float) -> int:
    """
    Convert normalized anxiety score to 1-5 label using fixed thresholds.
    """
    thresholds = [-float("inf"), -0.01, 0.01, 0.05, 0.10, float("inf")]
    for i in range(len(thresholds) - 1):
        if thresholds[i] <= anxiety_score_norm < thresholds[i + 1]:
            return i + 1
    return 5  # fallback


def run_demo(lexicon_path: str = "./data/raw/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"):
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

    # Example 2: Debug mode with verbose output
    print(f"\nExample 2: Debug mode")
    text2 = "I'm extremely scared and can't stop worrying."
    result2 = label_text(text2, verbose=True, expand_contractions=True, show_stats=True)
    print(
        f"Anxiety label: {get_anxiety_label_threshold(result2['anxiety_score_norm'])}"
    )

    # Example 3: Custom emotions and weights
    custom_emotions = ["fear", "sadness", "joy"]
    custom_weights = {"fear": 2.0, "sadness": 1.5, "joy": -1.0}

    print(f"\nExample 3: Custom emotions {custom_emotions}")
    text3 = "I feel scared and sad but also a bit happy."
    result3 = label_text(
        text3,
        custom_emotions=custom_emotions,
        custom_weights=custom_weights,
        show_stats=True,
    )
    print(f"Text: {text3}")
    print(f"Emotion counts: {result3['emo_counts']}")
    print(f"Anxiety score: {result3['anxiety_score_norm']:.3f}")
    print(f"Statistics: {result3.get('stats', {})}")

    # Example 4: Word-level analysis
    print(f"\nExample 4: Word-level analysis")
    text4 = "Happy thoughts mixed with anxious feelings."
    result4 = label_text(text4, word_emotion_mapping=True, matched_words_only=True)
    print(f"Text: {text4}")
    print(f"Word emotions: {result4.get('word_emotions', {})}")
    print(f"Matched words: {result4.get('matched_words', [])}")

    print(f"\n💡 For more comprehensive debug examples, run:")
    print(f"   python examples/debug_examples.py")
