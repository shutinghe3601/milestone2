# Text preprocessing utilities extracted from the NMF exploration notebook.

from __future__ import annotations

import re
import string
from typing import Dict, Iterable, List, Tuple, Union

try:
    import emoji  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    emoji = None

import numpy as np
from nltk.stem import SnowballStemmer
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer


c_dict = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "I would",
    "i'd've": "I would have",
    "i'll": "I will",
    "i'll've": "I will have",
    "i'm": "I am",
    "i've": "I have",
    "isn't": "is not",
    "it'd": "it had",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there had",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'alls": "you alls",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have",
}

c_re = re.compile("(%s)" % "|".join(c_dict.keys()))

DOMAIN_STOP = {
    # Platform terms
    "gpt", "chatgpt", "codex", "claude", "openai", "anthropic", "llm", "ai",
    # Generic discourse
    "work", "time", "user", "people", "person", "post", "new", "need", "way", 
    "model", "code", "someth", "peopl", "whi", "becaus", "tri", "actual",
    "know", "think", "want", "make", "say", "said", "tell", "come", "look",
    "use", "thing", "year", "day", "month", "week", "get", "got", "go", "went",
    "see", "seen", "feel", "felt", "help", "try", "start", "end", "give", "take",
    "find", "found", "seem", "believe", "understand", "mean", "happen", "turn",
    "keep", "put", "call", "ask", "let", "leave", "move", "show", "bring",
    # Reddit-specific
    "like", "just", "realli", "didn", "ll", "weren", "ve", "don", "im",
    "anyon", "doe", "similar", "situat", "kind", "reddit", "thread", "comment", 
    "subreddit", "upvote"
}

custom_stops = {'<cmt>', '[deleted]', 'amp', 'http', 'https', 'edit',
                'deleted', 'thing', 'stuff', 'really', 'just', 'like',
                'thanks', 'lol', 'im', 'dont', 'ive', 'youre', 'thats'}

punc = list(set(string.punctuation))

# Domain-specific cleaning constants
PUNCT_TRIM = ".,!?:;**()[]{}\"'\"\"'–-—/\\"
REPLACE_MAP = str.maketrans(
    {
        "\u2019": "'",  # curly apostrophe (')
        "\u201c": '"',  # left double quote (")
        "\u201d": '"',  # right double quote (")
        "\u2013": "-",  # en dash (–)
        "\u2014": "-",  # em dash (—)
    }
)
DOMAIN_TRASH = {
    "[text]",
    "[image]",
    "[removed]",
    "[deleted]",
    "redirected",
    "fool49",
    "faq",
    "summary__",
    "__extended",
    "mega",
    "topics",
}
SHORT_KEEP = {"ecg", "sad", "ptsd", "mom", "dad", "anx"}

NUMBER_PATTERN = re.compile(r"[0-9]+")
BASE_STOPWORDS = set(ENGLISH_STOP_WORDS) | custom_stops | DOMAIN_STOP




def _to_documents(text: str) -> List[str]:
    """Split a long text into pseudo-documents for TF-IDF fitting."""
    if not text:
        return []
    # Use paragraph-like breaks when available; fall back to full text.
    docs = [segment.strip() for segment in re.split(r"\n{2,}", text) if segment.strip()]
    if not docs:
        docs = [text]
    return docs


def auto_stop_from_tfidf(
    text: str,
    *,
    min_df: int = 1,
    idf_quantile: float = 0.1,
    ngram_range: Tuple[int, int] = (1, 1),
) -> List[str]:
    """
    Derive frequently occurring terms by fitting TF-IDF on the provided text.

    The text is segmented into paragraph-like chunks to create a small corpus.
    Terms whose inverse document frequency (IDF) falls below the specified
    quantile are returned as candidate stopwords.
    """
    docs = _to_documents(text)
    if not docs:
        return []

    # Ensure ``min_df`` is valid for the number of documents we have.
    min_df = min(min_df, max(1, len(docs)))

    vec = TfidfVectorizer(
        stop_words=None,
        lowercase=True,
        min_df=min_df,
        ngram_range=ngram_range,
    )
    X = vec.fit_transform(docs)
    if X.shape[1] == 0:
        return []

    idf = vec.idf_
    vocab = np.array(vec.get_feature_names_out())

    threshold = np.quantile(idf, idf_quantile)
    return sorted(vocab[idf <= threshold])


def expandContractions(text: str, c_re: re.Pattern[str] = c_re) -> str:
    """Expand contractions using the prepared lookup regex."""

    def replace(match: re.Match[str]) -> str:
        return c_dict[match.group(0)]

    return c_re.sub(replace, text)


def emojis_to_text(text: str) -> str:
    """Replace emojis with descriptive text labels for cleaner tokenization."""
    if emoji is None:
        return text
    try:
        return emoji.demojize(text, language="en", delimiters=(" ", " "))
    except Exception:
        return text


def casual_tokenizer(text: str) -> List[str]:
    """Tokenize text while preserving contractions and punctuation."""
    tokenizer = TweetTokenizer()
    return tokenizer.tokenize(text)


def ensure_tokens(x) -> List[str]:
    """Coerce input into a list of string tokens."""
    if isinstance(x, list):
        return [str(t) for t in x]
    if isinstance(x, str):
        return x.lower().split()
    return []


def normalize_tokens_min(text) -> List[str]:
    """Minimal tokenization with domain-specific cleaning and unit joining."""
    toks = str(text).split()
    out: List[str] = []
    prev = ""

    for t in toks:
        t = t.translate(REPLACE_MAP).strip().strip(PUNCT_TRIM)
        if not t:
            prev = ""
            continue
        low = t.lower()

        if low in c_dict:
            out.extend(c_dict[low].split())
            prev = ""
            continue

        if low in DOMAIN_TRASH or any(f in low for f in ("http", "reddit.com", "message/compose")):
            prev = ""
            continue
        if low.startswith("<") or low.endswith(">") or low in {"cmt", "discussion", "text"}:
            prev = ""
            continue

        if prev.isdigit() and low in {"mg", "mcg", "bpm", "kg", "lbs", "%"}:
            out[-1] = prev + low
            prev = out[-1]
            continue

        if len(low) < 3 and low not in SHORT_KEEP:
            prev = ""
            continue

        out.append(low)
        prev = low

    return out


def process_text(
    text: str,
    stemmer: SnowballStemmer | None = None,
    *,
    extra_stopwords: Iterable[str] | None = None,
    return_stopwords: bool = False
    ):
    """
    Full preprocessing pipeline: tokenize, normalize, expand contractions, stem, and filter stopwords.
    Optionally returns automatically discovered stopwords and the combined stopword list.
    """
    stop_words = set(BASE_STOPWORDS)

    if stemmer is None:
        stemmer = SnowballStemmer("english")

    text = emojis_to_text(text)
    tokens = casual_tokenizer(text)

    tokens = [token.lower() for token in tokens]
    tokens = [NUMBER_PATTERN.sub("", token) for token in tokens]

    tokens = [expandContractions(token, c_re=c_re) for token in tokens]
    tokens = [stemmer.stem(token) for token in tokens]

    tokens = [
        w
        for w in tokens
        if w not in punc
        and w not in stop_words
        and len(w) > 1
        and " " not in w
    ]

    if return_stopwords:
        return tokens, sorted(stop_words)
    return tokens

def process_text_tfidf(
    text: str,
    stemmer: SnowballStemmer | None = None,
    *,
    extra_stopwords: Iterable[str] | None = None,
    return_stopwords: bool = False,
    auto_stop_kwargs: Dict[str, Union[int, float, Tuple[int, int]]] | None = None,
) -> Union[List[str], Tuple[List[str], List[str], List[str]]]:
    """
    Full preprocessing pipeline with TF-IDF-based stopword augmentation: tokenize, normalize, expand contractions, stem, and filter stopwords.
    Optionally returns automatically discovered stopwords and the combined stopword list.
    """
    auto_stop_kwargs = auto_stop_kwargs or {}
    auto_stop = auto_stop_from_tfidf(text, **auto_stop_kwargs)

    stop_words = set(BASE_STOPWORDS)
    stop_words.update(auto_stop)
    if extra_stopwords is not None:
        stop_words.update(str(w).lower() for w in extra_stopwords)

    if stemmer is None:
        stemmer = SnowballStemmer("english")

    text = emojis_to_text(text)
    tokens = casual_tokenizer(text)

    tokens = [token.lower() for token in tokens]
    tokens = [NUMBER_PATTERN.sub("", token) for token in tokens]

    tokens = [expandContractions(token, c_re=c_re) for token in tokens]
    tokens = [stemmer.stem(token) for token in tokens]

    tokens = [
        w
        for w in tokens
        if w not in punc
        and w not in stop_words
        and len(w) > 1
        and " " not in w
    ]

    if return_stopwords:
        return tokens, auto_stop, sorted(stop_words)
    return tokens


def process_text_fordistilBERT(
    text: str,
    tfidf: bool = False,
    stemmer: SnowballStemmer | None = None,
    *,
    extra_stopwords: Iterable[str] | None = None,
    return_stopwords: bool = False,
    auto_stop_kwargs: Dict[str, Union[int, float, Tuple[int, int]]] | None = None,
) -> Union[List[str], Tuple[List[str], List[str], List[str]]]:
    """
    Full preprocessing pipeline with TF-IDF-based stopword augmentation: tokenize, normalize, expand contractions, stem, and filter stopwords.
    Optionally returns automatically discovered stopwords and the combined stopword list.
    """


    stop_words = set(BASE_STOPWORDS)

    if tfidf:
        auto_stop_kwargs = auto_stop_kwargs or {}
        auto_stop = auto_stop_from_tfidf(text, **auto_stop_kwargs)
        stop_words.update(auto_stop)
    if extra_stopwords is not None:
        stop_words.update(str(w).lower() for w in extra_stopwords)

    if stemmer is None:
        stemmer = SnowballStemmer("english")

    text = emojis_to_text(text)
    tokens = casual_tokenizer(text)

    tokens = [token.lower() for token in tokens]
    tokens = [NUMBER_PATTERN.sub("", token) for token in tokens]

    tokens = [expandContractions(token, c_re=c_re) for token in tokens]
    tokens = [stemmer.stem(token) for token in tokens]

    tokens = [
        w
        for w in tokens
        if w not in punc
        and w not in stop_words
        and len(w) > 1
        and " " not in w
    ]

    if return_stopwords:
        if tfidf:
            return tokens, auto_stop, sorted(stop_words)
        else:
            return tokens, [], sorted(stop_words)
    return tokens