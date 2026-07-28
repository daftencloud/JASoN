"""
evaluate_mixed_fused.py
----------------------------------------------------
Same evaluation report as evaluate_clean_fused_augmented.py (accuracy,
per-class precision/recall, confusion matrix) but for the ORIGINAL
mixed fused dataset (features_fused.csv, unfiltered -- includes
zero-padded rows where not all 3 sensors recorded), matching what
fused_random_forest.pkl was actually trained on. Useful for directly
comparing against the clean-filtered version's numbers.

Usage:
    python evaluate_mixed_fused.py
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                              classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split

DATA_PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


def main():
    df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "features_fused.csv"))
    print(f"Mixed fused dataset (matches fused_random_forest.pkl): "
          f"{len(df)} trials, {df['gesture'].nunique()} gestures")
    print(df["gesture"].value_counts())

    feature_cols = [c for c in df.columns if c not in ("gesture", "person", "trial")]
    X, y = df[feature_cols], df["gesture"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"\nHeld-out accuracy: {acc:.3f}\n")
    print(classification_report(y_test, preds, zero_division=0))

    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, preds, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title("Mixed fused model (matches fused_random_forest.pkl)")
    fig.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, "confusion_mixed_fused.png")
    fig.savefig(out_path, dpi=150)
    print(f"Confusion matrix saved -> {out_path}")


if __name__ == "__main__":
    main()
