"""
Text preprocessing utilities for NRC emotion analysis.
"""

import re
from typing import List

from .config import CONTRACTIONS

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


def expand_contractions(text: str) -> str:
    """
    Expand contractions in text (e.g., don't -> do not).

    Args:
        text: Input text

    Returns:
        Text with contractions expanded
    """
    processed_text = text
    for contraction, expansion in CONTRACTIONS.items():
        processed_text = processed_text.replace(contraction, expansion)
    return processed_text


def remove_urls(text: str) -> str:
    """
    Remove URL patterns from text.

    Args:
        text: Input text

    Returns:
        Text with URLs removed
    """
    url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    return re.sub(url_pattern, "", text)


def tokenize_text(text: str, lowercase: bool = True) -> List[str]:
    """
    Tokenize text using regex pattern [a-z]+.

    Args:
        text: Input text
        lowercase: Whether to convert to lowercase

    Returns:
        List of tokens
    """
    if not text:
        return []

    # Convert to lowercase if requested
    if lowercase:
        text = text.lower()

    # Tokenize with regex [a-z]+
    tokens = re.findall(r"[a-z]+", text, re.IGNORECASE if not lowercase else 0)
    return tokens


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """
    Lemmatize tokens using NLTK WordNetLemmatizer.

    Args:
        tokens: List of tokens to lemmatize

    Returns:
        List of lemmatized tokens
    """
    if not NLTK_AVAILABLE:
        return tokens

    try:
        lemmatizer = WordNetLemmatizer()
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

        return lemmatized_tokens
    except Exception as e:
        # If NLTK fails, keep original tokens
        print(f"NLTK lemmatization failed: {e}, keeping original words")
        return tokens


def preprocess_text(
    text: str,
    lowercase: bool = True,
    lemmatize: bool = True,
    expand_contractions_flag: bool = False,
    remove_urls_flag: bool = False,
) -> tuple[str, List[str]]:
    """
    Comprehensive text preprocessing pipeline.

    Args:
        text: Input text
        lowercase: Whether to convert to lowercase
        lemmatize: Whether to lemmatize words
        expand_contractions_flag: Whether to expand contractions
        remove_urls_flag: Whether to remove URLs

    Returns:
        Tuple of (processed_text, tokens)
    """
    if not text:
        return "", []

    processed_text = text

    # Apply additional preprocessing if requested
    if expand_contractions_flag:
        processed_text = expand_contractions(processed_text)

    if remove_urls_flag:
        processed_text = remove_urls(processed_text)

    # Tokenize
    tokens = tokenize_text(processed_text, lowercase)

    # Lemmatize if requested
    if lemmatize:
        tokens = lemmatize_tokens(tokens)

    return processed_text, tokens
