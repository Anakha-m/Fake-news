"""
Model Evaluation and Metric Calculation Module for Fake News Detection.
Computes Accuracy, Precision, Recall, F1-Score, Confusion Matrices, and Classification Reports.
"""

import json
from typing import Dict, Any, List
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_model_performance(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Evaluates binary classification performance with special emphasis on the FAKE class (label=1).
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec_fake = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec_fake = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1_fake = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    
    prec_macro = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average='macro', zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    report_dict = classification_report(
        y_true, 
        y_pred, 
        target_names=["REAL (0)", "FAKE (1)"], 
        output_dict=True,
        zero_division=0
    )
    report_str = classification_report(
        y_true, 
        y_pred, 
        target_names=["REAL (0)", "FAKE (1)"],
        zero_division=0
    )

    metrics = {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "precision_fake": round(prec_fake, 4),
        "recall_fake": round(rec_fake, 4),
        "f1_fake": round(f1_fake, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "confusion_matrix": {
            "true_negatives_real": int(tn),
            "false_positives_fake_flagged_as_real": int(fp),
            "false_negatives_real_flagged_as_fake": int(fn),
            "true_positives_fake": int(tp)
        },
        "classification_report": report_dict,
        "classification_report_str": report_str
    }

    return metrics


def print_comparison_table(metrics_list: List[Dict[str, Any]]) -> None:
    """Prints a clean ASCII comparison table of all evaluated models."""
    print("\n" + "="*80)
    print(f"{'Model Name':<26} | {'Accuracy':<10} | {'Precision (Fake)':<16} | {'Recall (Fake)':<14} | {'F1 (Fake)':<10}")
    print("="*80)
    for m in metrics_list:
        print(f"{m['model_name']:<26} | {m['accuracy']*100:<9.2f}% | {m['precision_fake']*100:<15.2f}% | {m['recall_fake']*100:<13.2f}% | {m['f1_fake']*100:<9.2f}%")
    print("="*80 + "\n")
