"""
evaluate_scoped.py
----------------------------------------------------
Evaluates each SCOPED specialist (trained only on its own owned
gestures, via train_scoped_specialists.py) with real accuracy numbers
AND confusion matrices -- so we can see exactly which gestures get
mixed up with which, instead of guessing. Directly follows the TA's
"report more than final accuracy... include a confusion matrix" advice.

Usage:
    python evaluate_scoped.py
"""

import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from train_scoped_specialists import SENSOR_OWNED_GESTURES

DATA_PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    for sensor, owned_gestures in SENSOR_OWNED_GESTURES.items():
        features_path = os.path.join(DATA_PROCESSED_DIR, f"features_{sensor}.csv")
        if not os.path.exists(features_path):
            print(f"Skipping {sensor}: no features file")
            continue

        df = pd.read_csv(features_path)
        df = df[df["gesture"].isin(owned_gestures)]

        if df.empty or df["gesture"].nunique() < 2:
            print(f"Skipping {sensor}: not enough data/classes")
            continue

        feature_cols = [c for c in df.columns if c not in ("gesture", "person", "trial")]
        X = df[feature_cols]
        y = df["gesture"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )

        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        print(f"\n=== {sensor} (scoped to {sorted(owned_gestures)}) ===")
        print(f"Overall accuracy: {acc:.3f}")
        print(f"\nPer-class report:")
        print(classification_report(y_test, preds, zero_division=0))

        labels = sorted(y.unique())
        cm = confusion_matrix(y_test, preds, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(7, 7))
        disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
        ax.set_title(f"{sensor} specialist (scoped)")
        fig.tight_layout()
        out_path = os.path.join(FIGURES_DIR, f"confusion_{sensor}_scoped.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Confusion matrix saved -> {out_path}")


if __name__ == "__main__":
    main()
