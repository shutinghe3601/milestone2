# SIADS 696: Milestone II Team Project

### TriggerLens: Predicting Anxiety Triggers from Reddit Data


## Project Overview

Mental health discussions on social media can be freeing for many, but they also risk amplifying anxiety-provoking content. With [19.1% of U.S. adults experiencing an anxiety disorder annually and 31.1% facing one in their lifetime](https://www.nimh.nih.gov/health/statistics/any-anxiety-disorder), understanding how online content triggers anxiety is crucial.

TriggerLens combines unsupervised and supervised approaches to build an analytical lens that both identifies discussion themes and predicts 'trigger scores' for Reddit posts, aiming to develop a predictive framework for assessing anxiety trigger potential.

## Goal

To develop a predictive framework that can assess the anxiety trigger potential of Reddit posts, allowing for future research into content moderation strategies which may reduce harmful exposure while preserving valuable mental health discourse.

## Team Members

- **Maria McKay (MM)**
- **Shen Shu (SS)**
- **Shuting He (SH)**

## Project Structure

```
milestone2/
├── configs/                    # Configuration files
│   └── pull_config.yml        # Reddit API and data collection settings
├── src/                       # Source code
│   ├── pull_reddit.py         # Reddit data collection
│   ├── build_datacard.py      # Data card generation
│   ├── weak_label_nrc.py      # NRC emotion lexicon weak labeling
│   └── utils/                 # Utility functions
│       ├── cleaning.py        # Text cleaning utilities
│       ├── language_detection.py # Language detection
│       └── logging_config.py  # Logging configuration
├── notebooks/                 # Jupyter notebooks
│   ├── 01_quick_qc.ipynb     # Quick quality check
│   └── 02_eda_weaklabels.ipynb # Exploratory data analysis
├── data/                      # Data storage
│   ├── raw/                   # Raw Reddit data and NRC lexicon
│   ├── interim/               # Processed data
│   ├── emotion_lexicon/       # Legacy NRC emotion files (deprecated)
│   └── external/              # External datasets
├── reports/                   # Analysis reports and visualizations
└── docs/                      # Documentation
```

## Methodology

### Part A: Unsupervised Learning
- **Dataset**: Reddit posts from mental health-related subreddits
- **Approach**:
  - Non-negative Matrix Factorization (NMF) with TF-IDF features
  - BERTopic with sentence embeddings for semantic topic modeling
- **Goal**: Identify interpretable discussion themes and rank topics by anxiety intensity

### Part B: Supervised Learning
- **Dataset**: Reddit posts with NRC Emotion Lexicon weak labels + GoEmotions dataset
- **Models**: Linear Regression, Random Forest, Ordinal Logistic Regression, XGBoost
- **Features**: Semantic embeddings derived from topic representations
- **Goal**: Predict anxiety trigger scores for topics not well covered by NRC lexicon

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/shutinghe3601/milestone2.git
   cd milestone2
   ```

2. **Create and activate virtual environment:**
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp secret.env.example secret.env
   # Edit secret.env with your Reddit API credentials
   ```

## Usage

### Data Collection
```bash
python src/pull_reddit.py
```

### Weak Labeling with NRC Emotion Lexicon
```bash
# Run demo with default NRC lexicon file
python src/weak_label_nrc.py

# Use custom text labeling in Python
from src.weak_label_nrc import label_text
result = label_text("I feel anxious about tomorrow's exam.")
print(result)
```

### Data Card Generation
```bash
python src/build_datacard.py
```

### Analysis
Open and run the Jupyter notebooks in the `notebooks/` directory for exploratory data analysis.

## NRC Emotion Lexicon Weak Labeling

The `weak_label_nrc.py` module provides automated emotion detection and anxiety scoring using the NRC Emotion Lexicon.

### Features
- **10 Emotion Categories**: anger, anticipation, disgust, fear, joy, sadness, surprise, trust, negative, positive
- **Anxiety Scoring**: Weighted combination of emotions with normalization
- **Advanced Processing**: 
  - NLTK lemmatization (optional)
  - Negation detection ("not happy" → reduced positive score)
  - Intensifier detection ("very scared" → amplified fear score)
- **Ordinal Labels**: 1-5 scale anxiety classification

## External Datasets

- **[NRC Emotion Lexicon v0.92](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm)**: Word-level emotion associations for 14,155+ words across 10 categories (8 emotions + positive/negative sentiment)
  - **File**: `data/raw/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt`
  - **Format**: Tab-separated values (word, emotion, score)
  - **Usage**: Automated weak labeling for anxiety scoring
- **[GoEmotions](https://research.google/blog/goemotions-a-dataset-for-fine-grained-emotion-classification/)**: 58,000 labeled Reddit comments with fine-grained emotions
- **[Reddit API](https://www.reddit.com/dev/api/oauth/)**: Mental health-related subreddits data collection

## Evaluation

- **Topic Quality**: Coherence scores comparing BERTopic vs NMF
- **Prediction Performance**: MAE, RMSE, Pearson/Spearman correlations
- **Visualizations**: Topic-anxiety heatmaps, subreddit distributions, word clouds

## License

This project is for educational purposes as part of SIADS 696 coursework.
