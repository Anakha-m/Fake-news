"""
Evaluation Report Generator for Academic Presentation.
Loads saved model metrics, confusion matrix, and classification report,
and displays formatted tables and insights for viva/presentation.
"""

import os
import json
import joblib

def display_evaluation_report():
    metrics_path = "ml_model/saved_models/model_metrics.json"
    if not os.path.exists(metrics_path):
        print(f"Metrics file not found at {metrics_path}. Please train models first.")
        return

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("\n" + "="*80)
    print("      FAKE NEWS DETECTION SYSTEM (SDG PROJECT) - EVALUATION REPORT")
    print("="*80)
    
    summary = data.get("dataset_summary", {})
    print(f"Dataset Size:          {summary.get('total_records', 'N/A')} records")
    print(f"Real News Samples:     {summary.get('real_count', 'N/A')}")
    print(f"Fake News Samples:     {summary.get('fake_count', 'N/A')}")
    print(f"Training Split:        {data.get('train_samples', 'N/A')} samples (80%)")
    print(f"Unseen Test Split:     {data.get('test_samples', 'N/A')} samples (20%)")
    print(f"Feature Dimensions:    {data.get('feature_count', 'N/A')} TF-IDF n-grams (1-2)")
    print(f"Selected Best Model:   {data.get('best_model_name', 'N/A')}")
    print("="*80)

    print("\n" + "-"*80)
    print(f"{'Model Name':<26} | {'Accuracy':<10} | {'Precision (Fake)':<16} | {'Recall (Fake)':<14} | {'F1 (Fake)':<10}")
    print("-"*80)

    for m in data.get("models_compared", []):
        print(f"{m['model_name']:<26} | {m['accuracy']*100:<9.2f}% | {m['precision_fake']*100:<15.2f}% | {m['recall_fake']*100:<13.2f}% | {m['f1_fake']*100:<9.2f}%")
    print("-"*80)

    best_name = data.get("best_model_name")
    best_m = next((m for m in data.get("models_compared", []) if m["model_name"] == best_name), None)

    if best_m:
        cm = best_m.get("confusion_matrix", {})
        print("\nCONFUSION MATRIX (Unseen Test Set):")
        print(f"  True Negatives (Actual REAL -> Pred REAL): {cm.get('true_negatives_real', 0)}")
        print(f"  False Positives (Actual REAL -> Pred FAKE): {cm.get('false_positives_fake_flagged_as_real', 0)}")
        print(f"  False Negatives (Actual FAKE -> Pred REAL): {cm.get('false_negatives_real_flagged_as_fake', 0)}")
        print(f"  True Positives  (Actual FAKE -> Pred FAKE): {cm.get('true_positives_fake', 0)}")

        print("\nDETAILED CLASSIFICATION REPORT:")
        print(best_m.get("classification_report_str", "N/A"))

    print("="*80 + "\n")

if __name__ == "__main__":
    display_evaluation_report()
