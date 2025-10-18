# SIADS 696: Milestone II Team Project

A machine learning framework that identifies and predicts anxiety-triggering content in Reddit posts using unsupervised topic modeling and supervised classification.

## Overview

With 19.1% of U.S. adults experiencing anxiety disorders annually (National Institute of Mental Health, n.d.), understanding how social media content affects mental health is crucial. TriggerLens combines unsupervised and supervised learning to identify discussion themes and predict anxiety trigger potential in Reddit posts.

## Methodology

**Data**: 6,283 Reddit posts from 8 communities (r/anxiety, r/HealthAnxiety, r/mentalhealth, etc.) collected via [Reddit API](https://www.reddit.com/dev/api/).

**Approach**:
1. **Unsupervised Learning**: NMF and BERTopic topic modeling to discover interpretable discussion themes
2. **Supervised Learning**: Random Forest, Logistic Regression, and DistilBERT models trained on hybrid-labeled data
3. **Feature Engineering**: TF-IDF (9,699 features), topic distributions (15 features), and metadata including NRC emotion scores (3 features)

**Labeling Strategy**: Hybrid approach combining 599  human annotations with 407 AI-generated labels using GPT-3.5-turbo API (OpenAI, 2023), both on 0-5 anxiety severity scale. NRC anxiety scores derived from [NRC Emotion Lexicon](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm) (Mohammad & Turney, 2013).

## Results

| Model | AUC | AP |
|-------|-----|----|
| DistilBERT | 0.927 | 0.932 |
| Logistic Regression | 0.894 | 0.571 |
| Random Forest | 0.847 | 0.564 |

**Topic Modeling**: NMF achieved 0.725 NPMI coherence with 15 distinct themes spanning clinical symptoms, interpersonal relationships, and technical discussions. BERTopic achieved comparable coherence (0.730) with better topic diversity (0.97) but fewer topics (7). DistilBERT was also used for supervised learning with transformer-based embeddings.

## Quick Start

1. **Setup**:
   ```bash
   git clone https://github.com/shutinghe3601/milestone2.git
   cd milestone2
   pip install -r requirements.txt
   ```

2. **Run Analysis**:
   ```bash
   jupyter notebook notebooks/
   ```

3. **Start with Data Overview**:
   - Begin with `01_quick_qc.ipynb` for data exploration
   - Follow the notebook sequence for complete analysis

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_quick_qc.ipynb` | Data quality checks |
| `02_clean_merge.ipynb` | Data preprocessing |
| `03_topic_modeling_byNMF.ipynb` | NMF topic modeling |
| `04_topic_modeling_byBERTopic.ipynb` | BERTopic modeling |
| `05_dataset_comparison_analysis.ipynb` | Label comparison |
| `06_text_classification_byDistilBERT.ipynb` | DistilBERT training |
| `07_text_classification_random_forest.ipynb` | Random Forest training |
| `08_text_classification_logistic_regression_classifier.ipynb` | Logistic Regression training |


## References

- Mohammad, S. M., & Turney, P. D. (2013). Crowdsourcing a word-emotion association lexicon. *Computational Intelligence*, 29(3), 436–465.
- National Institute of Mental Health. (n.d.). Any anxiety disorder. U.S. Department of Health and Human Services. https://www.nimh.nih.gov/health/statistics/any-anxiety-disorder
- OpenAI. (2023). GPT-3.5-turbo [Large language model]. https://platform.openai.com/docs/models/gpt-3-5-turbo
- Reddit API. (n.d.). Reddit API Documentation. https://www.reddit.com/dev/api/

## License

Educational project for SIADS 696 coursework • University of Michigan
