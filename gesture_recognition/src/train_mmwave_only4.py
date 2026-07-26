"""
train_mmwave_only4.py
----------------------------------------------------
One-off script: trains a Random Forest on ONLY the 4 mmWave-designated
gestures (raise_arms, soli, open_close_fist, palm_up_down), excluding
push/pull even though they have mmWave data too (since that data came
from the full 3-sensor collection, not a mmWave-only session).

Usage:
    python train_mmwave_only4.py
"""

import pickle
import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

DATA_PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

TARGET_GESTURES = {"raise_arms", "soli", "open_close_fist", "palm_up_down"}


def main():
    df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "features_mmwave.csv"))
    df = df[df["gesture"].isin(TARGET_GESTURES)]

    print(f"Training on {len(df)} trials across gestures: {sorted(df['gesture'].unique())}")

    X = df.drop(columns=["gesture", "person", "trial"])
    y = df["gesture"]

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)

    out_path = os.path.join(MODELS_DIR, "mmwave_4gesture_random_forest.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({
            "model": model,
            "feature_columns": list(X.columns),
            "feature_set": "mmwave",
        }, f)

    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
