# SIADS 696: Milestone II Team Project

## TriggerLens: Predicting Anxiety Triggers from Reddit Data

> **Work in Progress** — This project is under active development. Documentation and implementations are subject to change.

---

## Overview

Mental health discussions on social media can be freeing for many, but they also risk amplifying anxiety-provoking content. With [19.1% of U.S. adults experiencing an anxiety disorder annually](https://www.nimh.nih.gov/health/statistics/any-anxiety-disorder), understanding how online content triggers anxiety is crucial.

**TriggerLens** combines unsupervised and supervised machine learning approaches to:
1. Identify discussion themes in mental health communities
2. Predict anxiety trigger potential of Reddit posts

### Goal

Develop a predictive framework for assessing anxiety trigger potential in social media content, enabling future research into content moderation strategies.

### Team

- **Maria McKay**
- **Shen Shu**
- **Shuting He**

## Project Structure

> Note: Final project structure is to be determined and subject to reorganization.

---

## Methodology

### Unsupervised Learning (Topic Modeling)
- **Data**: Reddit posts from mental health-related communities
- **Methods**:
  - Non-negative Matrix Factorization (NMF)
  - BERTopic
- **Goal**: Discover discussion themes and topic distributions

### Supervised Learning (Classification)
- **Data**: Labeled posts (hand-annotated + AI labels generated using NRC Emotion Lexicon)
- **Methods**:
  - Random Forest
  - DistilBERT fine-tuning
- **Goal**: Predict anxiety trigger potential

**Status**: Model training complete. Results pending final analysis.

---

## Key Results

> **Coming Soon**: Detailed results and findings will be documented upon project completion.

### Preliminary Findings

- **Topic Modeling**: 15 interpretable themes identified across 6,283 Reddit posts
  - NPMI coherence: 0.725
  - Topic purity: 71.5%

- **Classification**: Best model achieved AUC = 0.849
  - Combined labeling approach (hand + AI) outperformed individual methods
  - 9,717 features (TF-IDF + topics + metadata)

---

## Setup

1. **Clone repository**:
   ```bash
   git clone https://github.com/shutinghe3601/milestone2.git
   cd milestone2
   ```

2. **Create virtual environment**:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment setup** (optional, for data collection):
   ```bash
   cp secret.env.example secret.env
   # Add Reddit API credentials to secret.env
   ```

---

## Usage

### Run Notebooks

Analysis notebooks are in the `notebooks/` directory:

| Notebook | Description |
|----------|-------------|
| `01_quick_qc.ipynb` | Data quality checks |
| `02_clean_merge.ipynb` | Data preprocessing |
| `03_topic_modeling_byNMF.ipynb` | NMF topic modeling |
| `04_topic_modeling_byBERTopic.ipynb` | BERTopic modeling |
| `05_dataset_comparison_analysis.ipynb` | Label comparison |
| `06_text_classification_byDistilBERT.ipynb` | DistilBERT training |
| `07_text_classification_random_forest.ipynb` | Random Forest training |

### Data Collection (Optional)

```bash
python src/pull_reddit.py  # Requires Reddit API credentials
```

### Weak Labeling

```bash
python src/weak_label_nrc.py  # NRC emotion-based labeling (used for AI label generation)
```

---

## Data Sources

| Source | Description | Usage |
|--------|-------------|-------|
| **Reddit API** | Mental health subreddit posts | Primary dataset |
| **[NRC Emotion Lexicon](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm)** | Word-emotion associations (14K+ words) | AI label generation |
| **[GoEmotions](https://research.google/blog/goemotions-a-dataset-for-fine-grained-emotion-classification/)** | 58K labeled Reddit comments | Reference dataset |

---

## Project Status

- [x] Data collection
- [x] Data preprocessing
- [x] Topic modeling (NMF & BERTopic)
- [x] Model training (Random Forest & DistilBERT)
- [ ] Final evaluation and comparison
- [ ] Documentation and report writing
- [ ] Presentation preparation

---

## License

Educational project for SIADS 696 coursework • University of Michigan
