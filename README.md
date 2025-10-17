# SIADS 696: Milestone II Team Project

A machine learning framework that identifies and predicts anxiety-triggering content in Reddit posts using unsupervised topic modeling and supervised classification.

## Overview

This project addresses the challenge of identifying anxiety-provoking content in online mental health discussions. With 19.1% of U.S. adults experiencing anxiety disorders annually, understanding how social media content affects mental health is crucial for developing better content moderation strategies.

## Methodology

**Data**: 6,283 Reddit posts from mental health communities (r/anxiety, r/HealthAnxiety, r/mentalhealth, etc.)

**Approach**:
1. **Unsupervised Learning**: Non-negative Matrix Factorization (NMF) and BERTopic to discover 15 interpretable discussion themes
2. **Supervised Learning**: Train Random Forest, Logistic Regression, and DistilBERT models on hybrid-labeled data
3. **Feature Engineering**: Combine TF-IDF (9,699 features), topic distributions (15 features), and metadata including NRC emotion-based anxiety scores (3 features)

**Labeling Strategy**: Hybrid approach combining 599 expert human annotations with 407 AI-generated labels using GPT-3.5-turbo API, both on 0-5 anxiety severity scale.

## Results

| Model | AUC | Key Strength |
|-------|-----|--------------|
| DistilBERT | 0.927 | Best overall performance |
| Logistic Regression | 0.901 | High recall (94.3%) |
| Random Forest | 0.847 | Interpretable features |

**Topic Modeling**: Achieved 0.725 NPMI coherence with 15 distinct themes spanning clinical symptoms, interpersonal relationships, and technical discussions.

## Quick Start

1. **Setup**:
   ```bash
   git clone <repository-url>
   cd milestone2
   pip install -r requirements.txt
   ```

2. **Generate NRC features** (if needed):
   ```bash
   python add_nrc_features.py
   ```

3. **Run complete analysis pipeline**:

   **Data Processing & Quality Control:**
   - `notebooks/01_quick_qc.ipynb` - Data quality checks and exploratory analysis
   - `notebooks/02_clean_merge.ipynb` - Data preprocessing and merging

   **Unsupervised Learning (Topic Modeling):**
   - `notebooks/03_topic_modeling_byNMF.ipynb` - NMF topic discovery and analysis
   - `notebooks/04_topic_modeling_byBERTopic.ipynb` - BERTopic modeling comparison

   **Supervised Learning (Classification):**
   - `notebooks/05_dataset_comparison_analysis.ipynb` - Human vs AI label comparison
   - `notebooks/06_text_classification_byDistilBERT.ipynb` - DistilBERT fine-tuning
   - `notebooks/07_text_classification_random_forest.ipynb` - Random Forest with failure analysis
   - `notebooks/10_text_classification_logreg_final.ipynb` - Logistic Regression final model

   **Main Analysis**: Start with `07_text_classification_random_forest.ipynb` for comprehensive results

## Project Structure

```
milestone2/
├── notebooks/           # Complete analysis pipeline (01-10)
├── src/                # Source code and utilities
├── data/               # Raw and processed datasets
│   ├── raw/           # Original Reddit data and lexicons
│   ├── processed/     # Clean datasets and labels
│   └── interim/       # Intermediate processing files
├── artifacts/          # Trained models and results
├── FINAL_REPORT.md     # Complete project documentation
└── requirements.txt    # Python dependencies
```

## Key Files

- `FINAL_REPORT.md` - Complete project documentation and results
- `add_nrc_features.py` - Generate NRC emotion-based features
- `src/weak_label_nrc.py` - NRC Emotion Lexicon processing
- `src/simple_ai_labeling.py` - GPT-3.5-turbo AI labeling
- `data/processed/reddit_anxiety_v1_with_nrc.parquet` - Main dataset with NRC scores
- `artifacts/` - Pre-trained models (TF-IDF vectorizer, NMF model, classification models)

## Data Sources

- **Reddit API**: Mental health community posts
- **NRC Emotion Lexicon**: Word-emotion associations for anxiety scoring
- **Human Annotations**: Expert ratings on 0-5 anxiety scale
- **GPT-3.5-turbo**: AI-generated weak labels for data augmentation

## Citation

Educational project for SIADS 696 • University of Michigan School of Information

---

*This research contributes to understanding how machine learning can help identify potentially harmful content in mental health discussions while preserving valuable peer support.*
