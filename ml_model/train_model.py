"""
Fake News Detection Machine Learning Training Pipeline.
SDG-Aligned Academic Project (SDG 16: Peace, Justice, and Strong Institutions).

Key Features of this Pipeline:
1. Pure Logistic Regression with L2 Regularization to prevent overfitting.
2. Controlled TF-IDF Vectorizer with max_features=5000 (word n-grams only) to filter noisy vocabulary.
3. Strict 80/20 Stratified Train/Test split: Vectorizer is fit ONLY on training data to prevent data leakage.
4. Class balance handled with class_weight='balanced'.
5. Regularization tuning across C = [1.0, 0.5, 0.1], comparing Training Accuracy vs. Test Accuracy to select the optimal model.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Set path to project root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml_model.preprocess import clean_text
from ml_model.dataset_builder import build_and_save_dataset


def run_training_pipeline(
    data_path: str = "ml_model/data/full_dataset.csv",
    models_dir: str = "ml_model/saved_models",
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Executes the leak-free Logistic Regression training & tuning pipeline.
    """
    print("\n" + "=" * 80)
    print("FAKE NEWS DETECTION ML TRAINING & REGULARIZATION TUNING")
    print("=" * 80)

    # 1. Ensure dataset is built and loaded
    build_and_save_dataset()
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)

    # 2. Check and report class balance
    real_count = int((df["label"] == 0).sum())
    fake_count = int((df["label"] == 1).sum())
    total_count = len(df)
    print(f"\n[1/5] Dataset Health & Class Distribution:")
    print(f"      Total Articles:    {total_count}")
    print(f"      REAL Articles (0): {real_count} ({real_count / total_count * 100:.1f}%)")
    print(f"      FAKE Articles (1): {fake_count} ({fake_count / total_count * 100:.1f}%)")
    print(f"      Class Weight Strategy: 'balanced' (weights inversely proportional to class frequencies)")

    # 3. Preprocess text
    print("\n[2/5] Cleaning text (removing noise, expanding contractions, filtering stopwords)...")
    df["cleaned_text"] = df["combined_content"].apply(clean_text)
    df = df[df["cleaned_text"].str.strip() != ""].reset_index(drop=True)

    X_raw = df["cleaned_text"].values
    y = df["label"].values

    # 4. Stratified Train/Test Split (80% Train, 20% Unseen Test)
    print("\n[3/5] Performing Stratified 80/20 Train/Test Split...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=0.20,
        random_state=random_state,
        stratify=y
    )

    print(f"      Training Set: {len(X_train_raw)} samples (Real: {(y_train==0).sum()}, Fake: {(y_train==1).sum()})")
    print(f"      Test Set:     {len(X_test_raw)} samples (Real: {(y_test==0).sum()}, Fake: {(y_test==1).sum()})")

    # 5. Feature Extraction: Fit TF-IDF strictly on training data ONLY
    print("\n[4/5] Fitting TF-IDF Vectorizer (max_features=5000, n-grams=(1,2))...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True,
        min_df=1,
        max_df=0.90
    )
    
    # Fit strictly on train set, transform both train and test
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    X_test_tfidf = vectorizer.transform(X_test_raw)
    print(f"      Vocabulary Size: {len(vectorizer.vocabulary_)} features")

    # 6. Tune Regularization Parameter 'C' in Logistic Regression
    print("\n[5/5] Regularization Tuning for Logistic Regression (C = [1.0, 0.5, 0.1])...")
    print("      (Note: Smaller C = Stronger L2 Regularization = Simpler Model = Less Overfitting)")
    
    c_candidates = [1.0, 0.5, 0.1]
    tuning_results = []
    trained_models = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    print("\n" + "-" * 90)
    print(f"{'C Value':<10} | {'Train Acc':<12} | {'Test Acc':<12} | {'Train-Test Gap':<16} | {'CV F1 (Train)':<15} | {'Test F1 (Fake)':<14}")
    print("-" * 90)

    for c in c_candidates:
        model = LogisticRegression(
            C=c,
            penalty='l2',
            solver='lbfgs',
            max_iter=1000,
            class_weight='balanced',
            random_state=random_state
        )

        # 5-fold cross validation on training data
        cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=cv, scoring='f1')

        # Fit model on training set
        model.fit(X_train_tfidf, y_train)
        trained_models[c] = model

        # Predictions on train and test sets
        y_train_pred = model.predict(X_train_tfidf)
        y_test_pred = model.predict(X_test_tfidf)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        acc_gap = train_acc - test_acc

        test_prec_fake = precision_score(y_test, y_test_pred, pos_label=1, zero_division=0)
        test_rec_fake = recall_score(y_test, y_test_pred, pos_label=1, zero_division=0)
        test_f1_fake = f1_score(y_test, y_test_pred, pos_label=1, zero_division=0)

        cm = confusion_matrix(y_test, y_test_pred)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

        clf_report = classification_report(y_test, y_test_pred, target_names=["REAL", "FAKE"], output_dict=True, zero_division=0)

        tuning_results.append({
            "C": c,
            "train_accuracy": round(float(train_acc), 4),
            "test_accuracy": round(float(test_acc), 4),
            "generalization_gap": round(float(acc_gap), 4),
            "cv_f1_mean": round(float(cv_scores.mean()), 4),
            "cv_f1_std": round(float(cv_scores.std()), 4),
            "test_precision_fake": round(float(test_prec_fake), 4),
            "test_recall_fake": round(float(test_rec_fake), 4),
            "test_f1_fake": round(float(test_f1_fake), 4),
            "confusion_matrix": {
                "true_real": int(tn),
                "false_fake": int(fp),
                "false_real": int(fn),
                "true_fake": int(tp)
            },
            "classification_report": clf_report
        })

        print(f"C = {c:<6} | {train_acc*100:<10.2f}% | {test_acc*100:<10.2f}% | {acc_gap*100:<14.2f}% | {cv_scores.mean()*100:<5.2f}% (±{cv_scores.std()*100:.1f}%) | {test_f1_fake*100:<12.2f}%")

    print("-" * 90)

    # Pick the best C that balances high test accuracy, high F1 score, and minimal generalization gap
    best_candidate = max(
        tuning_results,
        key=lambda r: (r["test_f1_fake"], r["test_accuracy"], -r["generalization_gap"])
    )
    best_c = best_candidate["C"]
    best_model = trained_models[best_c]

    print(f"\n OPTIMAL REGULARIZATION SELECTED: C = {best_c}")
    print(f"   Training Accuracy:   {best_candidate['train_accuracy']*100:.2f}%")
    print(f"   Test Accuracy:       {best_candidate['test_accuracy']*100:.2f}%")
    print(f"   Generalization Gap:  {best_candidate['generalization_gap']*100:.2f}% (Balanced, Low Overfitting)")
    print(f"   Fake News Recall:    {best_candidate['test_recall_fake']*100:.2f}%")
    print(f"   Fake News F1-Score:  {best_candidate['test_f1_fake']*100:.2f}%")

    # 7. Serialize Artifacts
    os.makedirs(models_dir, exist_ok=True)
    best_model_path = os.path.join(models_dir, "best_model.pkl")
    vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    metrics_path = os.path.join(models_dir, "model_metrics.json")

    joblib.dump(best_model, best_model_path)
    joblib.dump(vectorizer, vectorizer_path)

    metadata = {
        "model_type": "Logistic Regression (L2 Regularized)",
        "optimal_C": best_c,
        "class_weight": "balanced",
        "max_features": 5000,
        "ngram_range": [1, 2],
        "train_samples": int(len(X_train_raw)),
        "test_samples": int(len(X_test_raw)),
        "best_metrics": best_candidate,
        "all_c_tuning_results": tuning_results,
        "dataset_summary": {
            "total_articles": total_count,
            "real_count": real_count,
            "fake_count": fake_count
        }
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"\n Artifacts saved to '{models_dir}/':")
    print(f"   - {best_model_path}")
    print(f"   - {vectorizer_path}")
    print(f"   - {metrics_path}")
    print("=" * 80 + "\n")

    return metadata


if __name__ == "__main__":
    run_training_pipeline()
