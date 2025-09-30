#!/usr/bin/env python3
"""
Debug Examples for Enhanced NRC Emotion Lexicon Weak Labeling

This file demonstrates the new debugging parameters added to the label_text function.
Run this file to see various debugging features in action.
"""

import os
import sys

# Add the src directory to Python path
src_path = os.path.join(os.path.dirname(__file__), "..", "..", "src")
sys.path.append(src_path)

from weak_label_nrc import get_anxiety_label_threshold, label_text


def basic_example():
    """Basic usage without debug parameters"""
    print("=" * 60)
    print("1. BASIC USAGE")
    print("=" * 60)

    text = "I feel terrified and keep panicking about tomorrow's exam."
    result = label_text(text)
    anxiety_label = get_anxiety_label_threshold(result["anxiety_score_norm"])

    print(f"Text: {text}")
    print(f"Tokens: {result['n_tokens']}")
    print(f"Emotion counts: {result['emo_counts']}")
    print(f"Anxiety score: {result['anxiety_score_norm']:.3f}")
    print(f"Anxiety label: {anxiety_label}")
    print()


def scaling_modes_example():
    """Demonstrate the three scaling modes: threshold, statistical, both"""
    print("=" * 60)
    print("2. SCALING MODES COMPARISON")
    print("=" * 60)

    text = "I'm extremely anxious and terrified about tomorrow's exam."
    print(f"Text: {text}")
    print()

    # Mode 1: Threshold (default, 1-5 labels)
    print("THRESHOLD MODE (1-5 labels):")
    result1 = label_text(text, scaling_method="threshold")
    print(f"   Raw score: {result1['anxiety_score_raw']:.3f}")
    print(f"   Normalized: {result1['anxiety_score_norm']:.3f}")
    print(f"   Label: {result1['anxiety_label']}")
    print()

    # Mode 2: Statistical (0-1 scores like text_process.ipynb)
    print("STATISTICAL MODE (0-1 scores):")
    result2 = label_text(text, scaling_method="statistical")
    print(f"   Raw score: {result2['anxiety_score_raw']:.3f}")
    print(f"   Statistical: {result2['anxiety_score_statistical']:.3f}")
    print()

    # Mode 3: Both modes
    print("BOTH MODES:")
    result3 = label_text(text, scaling_method="both")
    print(f"   Raw score: {result3['anxiety_score_raw']:.3f}")
    print(f"   Threshold label (1-5): {result3['anxiety_label']}")
    print(f"   Statistical score (0-1): {result3['anxiety_score_statistical']:.3f}")
    print()

    # Mode 4: Custom statistical parameters
    print("CUSTOM STATISTICAL PARAMETERS:")
    custom_params = {"median": 1.0, "mad": 0.5}
    result4 = label_text(
        text, scaling_method="statistical", statistical_params=custom_params
    )
    print(f"   Custom params: {custom_params}")
    print(f"   Statistical score: {result4['anxiety_score_statistical']:.3f}")
    print()


def verbose_example():
    """Demonstrate verbose output"""
    print("=" * 60)
    print("3. VERBOSE MODE DEBUGGING")
    print("=" * 60)

    text = "I'm extremely anxious and scared about this situation."
    result = label_text(text, verbose=True, expand_contractions=True)
    print()


def comprehensive_debug():
    """Show all debug information"""
    print("=" * 60)
    print("4. COMPREHENSIVE DEBUG INFORMATION")
    print("=" * 60)

    text = "I don't feel happy, but I'm not completely sad either."
    result = label_text(
        text,
        return_intermediate=True,
        expand_contractions=True,
        remove_urls=False,
        show_stats=True,
        verbose=False,  # Set to False to avoid duplicate output
    )

    print(f"Text: {text}")
    print(f"Result keys: {list(result.keys())}")
    print()

    # Show debug information
    debug_info = result.get("debug_info", {})
    print("Debug Information:")
    print(f"  Original text: {debug_info.get('original_text', 'N/A')}")
    print(f"  Processed text: {debug_info.get('processed_text', 'N/A')}")
    print(f"  Tokens: {debug_info.get('tokens', [])}")
    print(f"  Word emotions: {debug_info.get('word_emotions', {})}")
    print(f"  Matched words: {debug_info.get('matched_words', [])}")

    # Show statistics
    stats = result.get("stats", debug_info.get("stats", {}))
    if stats:
        print("\nStatistics:")
        print(f"  Total words: {stats.get('total_words', 0)}")
        print(f"  Matched words: {stats.get('matched_words', 0)}")
        print(f"  Match rate: {stats.get('match_rate', 0):.2%}")
        print(f"  Emotions found: {stats.get('emotions_found', [])}")
        print(f"  Top emotions: {stats.get('top_emotions', [])}")
    print()


def custom_emotions_example():
    """Test custom emotions and weights"""
    print("=" * 60)
    print("5. CUSTOM EMOTIONS AND WEIGHTS")
    print("=" * 60)

    text = "I feel very scared and sad about losing my job."

    # Test with custom subset of emotions
    custom_emotions = ["fear", "sadness", "joy", "anger"]
    custom_weights = {
        "fear": 2.0,  # Double fear weight
        "sadness": 1.5,  # Increase sadness weight
        "joy": -1.0,  # Negative joy weight
        "anger": 1.2,  # Slightly increase anger
    }

    print("Standard configuration:")
    result1 = label_text(text, show_stats=True)
    print(f"  Anxiety score: {result1['anxiety_score_norm']:.3f}")
    print(f"  Top emotions: {result1['stats']['top_emotions']}")

    print("\nCustom configuration:")
    result2 = label_text(
        text,
        custom_emotions=custom_emotions,
        custom_weights=custom_weights,
        show_stats=True,
        verbose=True,
    )
    print(f"  Anxiety score: {result2['anxiety_score_norm']:.3f}")
    print(f"  Top emotions: {result2['stats']['top_emotions']}")
    print()


def text_preprocessing_example():
    """Test various text preprocessing options"""
    print("=" * 60)
    print("6. TEXT PREPROCESSING OPTIONS")
    print("=" * 60)

    text = (
        "I can't visit https://example.com because I'm terrified! Don't worry though."
    )

    print("Original text:", text)
    print()

    # Without preprocessing
    print("A. Without preprocessing:")
    result1 = label_text(text, matched_words_only=True, verbose=True)
    print(f"   Matched words: {result1.get('matched_words', [])}")
    print()

    # With contraction expansion
    print("B. With contraction expansion:")
    result2 = label_text(
        text, expand_contractions=True, matched_words_only=True, verbose=True
    )
    print(f"   Matched words: {result2.get('matched_words', [])}")
    print()

    # With URL removal and contraction expansion
    print("C. With both URL removal and contraction expansion:")
    result3 = label_text(
        text,
        expand_contractions=True,
        remove_urls=True,
        matched_words_only=True,
        verbose=True,
    )
    print(f"   Matched words: {result3.get('matched_words', [])}")
    print()


def word_emotion_mapping_example():
    """Show word-to-emotion mapping"""
    print("=" * 60)
    print("7. WORD-TO-EMOTION MAPPING")
    print("=" * 60)

    text = "I feel angry, sad, happy, and surprised at the same time."
    result = label_text(text, word_emotion_mapping=True, show_stats=True)

    print(f"Text: {text}")
    print(f"Word-emotion mapping: {result.get('word_emotions', {})}")
    print(f"Emotion counts: {result['emo_counts']}")
    print(f"Statistics: {result.get('stats', {})}")
    print()


def comparison_example():
    """Compare different parameter combinations"""
    print("=" * 60)
    print("8. PARAMETER COMPARISON")
    print("=" * 60)

    text = "I'm extremely worried and can't stop feeling anxious."

    configs = [
        {"name": "Default", "params": {}},
        {"name": "With contractions", "params": {"expand_contractions": True}},
        {
            "name": "High fear weight",
            "params": {
                "custom_weights": {
                    "fear": 2.0,
                    "sadness": 0.8,
                    "anger": 0.8,
                    "disgust": 0.8,
                    "anticipation": 0.5,
                    "surprise": 0.4,
                    "joy": -0.6,
                    "trust": -0.5,
                    "negative": 0.4,
                    "positive": -0.3,
                }
            },
        },
        {
            "name": "Fear focus only",
            "params": {
                "custom_emotions": ["fear", "anxiety", "negative"],
                "custom_weights": {"fear": 1.5, "negative": 1.0},
            },
        },
    ]

    print(f"Text: {text}")
    print()

    for config in configs:
        print(f"{config['name']}:")
        result = label_text(text, **config["params"])
        anxiety_label = get_anxiety_label_threshold(result["anxiety_score_norm"])
        print(f"  Anxiety score: {result['anxiety_score_norm']:.3f}")
        print(f"  Anxiety label: {anxiety_label}")
        print(
            f"  Top emotions: {[(emo, count) for emo, count in result['emo_counts'].items() if count > 0][:3]}"
        )
        print()


def performance_analysis_example():
    """Analyze performance with statistics"""
    print("=" * 60)
    print("9. PERFORMANCE ANALYSIS")
    print("=" * 60)

    texts = [
        "I love this sunny day!",
        "I'm terrified of spiders and feel disgusted.",
        "This is just a normal day with mixed feelings.",
        "I'm extremely anxious about everything in my life right now.",
    ]

    print("Analyzing multiple texts with statistics:")
    print()

    for i, text in enumerate(texts, 1):
        result = label_text(text, show_stats=True)
        stats = result["stats"]

        print(f"Text {i}: {text}")
        print(f"  Match rate: {stats['match_rate']:.1%}")
        print(f"  Emotions found: {stats['emotions_found']}")
        print(f"  Anxiety score: {result['anxiety_score_norm']:.3f}")
        print(
            f"  Anxiety label: {get_anxiety_label_threshold(result['anxiety_score_norm'])}"
        )
        print()


def main():
    """Run all examples"""
    print("Enhanced NRC Emotion Lexicon - Debug Parameter Examples")
    print("=" * 80)
    print()

    try:
        basic_example()
        scaling_modes_example()  # New: Three scaling modes comparison
        verbose_example()
        comprehensive_debug()
        custom_emotions_example()
        text_preprocessing_example()
        word_emotion_mapping_example()
        comparison_example()
        performance_analysis_example()

        print("All examples completed successfully!")
        print()
        print("Tips for debugging:")
        print("- Use verbose=True to see step-by-step processing")
        print("- Use return_intermediate=True to get all internal data")
        print("- Use show_stats=True to analyze emotion matching performance")
        print(
            "- Use scaling_method='threshold'/'statistical'/'both' for different outputs"
        )
        print("- Use custom_emotions/custom_weights to test different configurations")
        print("- Use expand_contractions=True for better text normalization")
        print()
        print("Scaling Methods:")
        print("- 'threshold': 1-5 ordinal labels (good for categorical analysis)")
        print("- 'statistical': 0-1 continuous scores (good for ML modeling)")
        print("- 'both': get both outputs for comparison")

    except Exception as e:
        print(f"Error during examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
