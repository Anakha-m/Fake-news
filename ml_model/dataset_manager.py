"""
Dataset Manager & Loader for Fake News Detection ML Pipeline.
Handles dataset ingestion, validation, deduplication, and sanity checking.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any


def load_and_validate_dataset(csv_path: str = "ml_model/data/full_dataset.csv") -> pd.DataFrame:
    """
    Loads dataset, enforces schema, checks missing values, removes duplicates,
    verifies label mappings (0=REAL, 1=FAKE), and returns a clean DataFrame.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at: {csv_path}. Please run dataset builder first.")

    df = pd.read_csv(csv_path)

    # 1. Check required columns
    required_cols = {"title", "text", "label"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"Dataset must contain columns: {required_cols}. Found: {list(df.columns)}")

    # 2. Check and handle missing values
    initial_rows = len(df)
    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    
    # Filter out empty records
    df = df[(df["title"].str.strip() != "") | (df["text"].str.strip() != "")].copy()
    
    # Create combined_content if not present
    if "combined_content" not in df.columns:
        df["combined_content"] = df["title"] + " " + df["text"]

    # 3. Deduplication
    df = df.drop_duplicates(subset=["combined_content"]).reset_index(drop=True)
    dedup_rows = len(df)

    # 4. Verify labels
    df["label"] = df["label"].astype(int)
    unique_labels = set(df["label"].unique())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"Labels must strictly be binary 0 (REAL) and 1 (FAKE). Found: {unique_labels}")

    real_count = (df["label"] == 0).sum()
    fake_count = (df["label"] == 1).sum()

    print("\n" + "="*50)
    print("DATASET VALIDATION & HEALTH CHECK")
    print("="*50)
    print(f"Total Rows Loaded:      {initial_rows}")
    print(f"Valid Rows After Clean: {dedup_rows}")
    print(f"REAL News Samples (0):  {real_count} ({real_count/dedup_rows*100:.1f}%)")
    print(f"FAKE News Samples (1):  {fake_count} ({fake_count/dedup_rows*100:.1f}%)")
    print("Label Mapping Verified: 0 -> REAL, 1 -> FAKE")
    print("="*50 + "\n")

    return df


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Returns metadata dictionary summarizing dataset composition."""
    return {
        "total_records": int(len(df)),
        "real_count": int((df["label"] == 0).sum()),
        "fake_count": int((df["label"] == 1).sum()),
        "categories": list(df["category"].unique()) if "category" in df.columns else [],
        "sources": list(df["source"].unique()) if "source" in df.columns else []
    }
