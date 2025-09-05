# External Data

This directory contains external datasets and lexicons used in the project.

## NRC Emotion Lexicon
- **File**: `NRC-Emotion-Lexicon-Wordlevel-v0.92.txt`
- **Source**: [NRC Emotion Lexicon](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm)
- **Description**: Word-level emotion associations for 10,170 words across 8 emotions (anger, fear, anticipation, trust, surprise, sadness, joy, disgust) and 2 sentiments (negative, positive)

## Usage
The NRC lexicon will be used by `src/weak_label_nrc.py` to apply emotion scores to Reddit documents.

## Download Instructions
1. Visit the NRC Emotion Lexicon website
2. Download the word-level lexicon
3. Place the file in this directory
4. Update the lexicon path in your configuration
