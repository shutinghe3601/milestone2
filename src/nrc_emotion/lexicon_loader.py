"""
NRC Emotion Lexicon loading utilities.
"""

from pathlib import Path
from typing import Dict, List, Set

from .config import DEFAULT_LEXICON_PATH


def load_lexicon(
    lexicon_path: str = DEFAULT_LEXICON_PATH, emotions: List[str] = None
) -> Dict[str, Set[str]]:
    """
    Load NRC emotion lexicon from a single file.

    Args:
        lexicon_path: Path to the NRC emotion lexicon file
        emotions: List of emotions to load

    Returns:
        Dictionary mapping emotion -> set of words
    """
    if emotions is None:
        from .config import DEFAULT_EMOTIONS

        emotions = DEFAULT_EMOTIONS

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


def get_lexicon_stats(lexicon: Dict[str, Set[str]]) -> Dict[str, int]:
    """
    Get statistics about the loaded lexicon.

    Args:
        lexicon: Dictionary mapping emotion -> set of words

    Returns:
        Dictionary mapping emotion -> word count
    """
    return {emotion: len(words) for emotion, words in lexicon.items()}


def find_emotion_matches(
    tokens: List[str], lexicon: Dict[str, Set[str]], emotions: List[str]
) -> tuple[Dict[int, List[str]], Dict[str, List[str]], Set[str]]:
    """
    Find emotion matches for tokens in the lexicon.

    Args:
        tokens: List of processed tokens
        lexicon: Dictionary mapping emotion -> set of words
        emotions: List of emotions to check

    Returns:
        Tuple of:
        - emotion_matches: Dict mapping token index -> list of matched emotions
        - word_emotions: Dict mapping word -> list of emotions (for debugging)
        - matched_words: Set of words that matched any emotion
    """
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

    return emotion_matches, word_emotions, matched_words
