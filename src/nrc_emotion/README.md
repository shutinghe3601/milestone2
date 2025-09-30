# NRC Emotion Analysis Package

This package contains specialized modules for analyzing text using the NRC Emotion Lexicon. It provides emotion detection, anxiety scoring, and comprehensive text preprocessing capabilities.

## 📁 Package Structure

```
src/nrc_emotion/
├── __init__.py           # Package initialization and exports
├── config.py             # Configuration constants and defaults
├── lexicon_loader.py     # NRC lexicon loading utilities
├── text_preprocessor.py  # Text preprocessing pipeline
├── emotion_analyzer.py   # Emotion analysis and scoring
└── README.md            # This file
```

## 🔧 Modules Overview

### `config.py`
- **Purpose**: Central configuration for all NRC emotion analysis
- **Contents**: 
  - Default emotions list
  - Anxiety scoring weights (optimized for positive/negative balance)
  - Anxiety label thresholds (adjusted to reduce false positives)
  - Negation and intensifier words
  - Common contractions mapping

### `lexicon_loader.py`
- **Purpose**: Load and manage NRC Emotion Lexicon data
- **Key Functions**:
  - `load_lexicon()`: Load emotion words from lexicon file
  - `find_emotion_matches()`: Match tokens to emotions
  - `get_lexicon_stats()`: Get lexicon statistics

### `text_preprocessor.py`
- **Purpose**: Comprehensive text preprocessing pipeline
- **Key Functions**:
  - `preprocess_text()`: Main preprocessing pipeline
  - `expand_contractions()`: Expand contractions (don't → do not)
  - `remove_urls()`: Remove URL patterns
  - `tokenize_text()`: Tokenize text using regex
  - `lemmatize_tokens()`: Lemmatize using NLTK (optional)

### `emotion_analyzer.py`
- **Purpose**: Core emotion analysis and anxiety scoring
- **Key Functions**:
  - `apply_negation_intensifier()`: Apply negation/intensifier rules
  - `compute_emotion_counts_and_anxiety()`: Calculate emotion counts and anxiety scores
  - `apply_threshold_scaling()`: Apply threshold-based normalization
  - `apply_statistical_scaling()`: Apply statistical normalization (MAD + Z-score + Sigmoid)
  - `get_anxiety_label_threshold()`: Convert scores to 1-5 labels
  - `compute_statistics()`: Generate analysis statistics

## 🚀 Usage

### Basic Import
```python
from nrc_emotion import (
    DEFAULT_EMOTIONS, 
    DEFAULT_WEIGHTS,
    load_lexicon,
    preprocess_text,
    apply_negation_intensifier,
    get_anxiety_label_threshold
)
```

### Using the Main Interface
```python
from weak_label_nrc import label_text

# Basic usage
result = label_text("I feel anxious about tomorrow.")
print(f"Anxiety label: {result['anxiety_label']}")

# Advanced usage with custom parameters
result = label_text(
    "I'm extremely worried!",
    scaling_method="both",
    expand_contractions=True,
    verbose=True
)
```

## 🎯 Key Improvements

### 1. **Modular Design**
- Each module has a single responsibility
- Easy to test and maintain individual components
- Clear separation of concerns

### 2. **Optimized Anxiety Scoring**
- **Fixed positive text misclassification**: Positive texts now correctly get low anxiety labels
- **Balanced emotion weights**: Increased negative weights for positive emotions
- **Adjusted thresholds**: Reduced false positives with higher thresholds

### 3. **Comprehensive Configuration**
- All settings centralized in `config.py`
- Easy to modify weights, thresholds, and word lists
- Well-documented parameter meanings

### 4. **Enhanced Text Processing**
- Robust preprocessing pipeline
- Optional NLTK integration with fallback
- Support for contractions, URLs, and various text formats

## 📊 Anxiety Scoring Improvements

| Text Type | Before | After | Status |
|-----------|--------|-------|--------|
| Positive texts | Label 5 ❌ | Label 1 ✅ | Fixed |
| Neutral texts | Label 1 ✅ | Label 1 ✅ | Maintained |
| High anxiety | Label 5 ✅ | Label 5 ✅ | Maintained |

### New Weight Configuration
- `joy`: -0.5 → **-1.0** (stronger positive influence)
- `positive`: -0.2 → **-0.5** (stronger positive influence)  
- `trust`: -0.4 → **-0.8** (stronger positive influence)
- `anticipation`: 0.4 → **0.2** (reduced, can be positive)
- `surprise`: 0.3 → **0.1** (reduced, can be positive)

### New Threshold Configuration
- **Label 1**: -∞ to -0.05 (very low anxiety)
- **Label 2**: -0.05 to 0.05 (low anxiety)
- **Label 3**: 0.05 to 0.15 (moderate anxiety)
- **Label 4**: 0.15 to 0.30 (high anxiety)
- **Label 5**: 0.30 to ∞ (very high anxiety)

## 🔄 Migration from Old Structure

The package maintains full backward compatibility. Existing code using `weak_label_nrc.py` will continue to work without changes.

**Old import paths** (still work):
```python
from weak_label_nrc import label_text, get_anxiety_label_threshold
```

**New modular imports** (for advanced usage):
```python
from nrc_emotion.config import DEFAULT_EMOTIONS, DEFAULT_WEIGHTS
from nrc_emotion.emotion_analyzer import apply_negation_intensifier
from nrc_emotion.text_preprocessor import preprocess_text
```
