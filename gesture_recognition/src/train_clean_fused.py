"""
train_clean_fused.py
----------------------------------------------------
Trains a fused model using ONLY trials where all 3 sensors genuinely
recorded real data (no zero-padded rows from single-sensor
collections). This removes the "inconsistent training examples" issue
where the same gesture label sometimes has real UWB/mmWave signal and
sometimes doesn't (e.g. push: Jon's trials have real UWB+mmWave, Nav's
don't) -- which could be diluting whatever real cross-sensor signal
exists.

Usage:
    python train_clean_fused.py
"""

import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

DATA_PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def main():
    df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "features_fused.csv"))

    # A "clean" row is one where imu, uwb, and mmwave all have real
    # (non-zero) data -- check one representative column per sensor.
    imu_col = next((c for c in df.columns if c.startswith("imu_")), None)
    uwb_col = next((c for c in df.columns if c.startswith("uwb_")), None)
    mmwave_col = next((c for c in df.columns if c.startswith("mmwave_")), None)

    before_count = len(df)
    clean_df = df[
        (df[imu_col] != 0) & (df[uwb_col] != 0) & (df[mmwave_col] != 0)
    ]
    after_count = len(clean_df)

    print(f"Before: {before_count} total trials")
    print(f"After filtering to genuine 3-sensor rows: {after_count} trials")
    print(f"\nGestures remaining and their counts:")
    print(clean_df["gesture"].value_counts())

    feature_cols = [c for c in clean_df.columns if c not in ("gesture", "person", "trial")]
    X, y = clean_df[feature_cols], clean_df["gesture"]

    if y.nunique() < 2:
        print("Not enough distinct gestures remain after filtering -- aborting")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"\nHeld-out accuracy: {acc:.3f}\n")
    print(classification_report(y_test, preds, zero_division=0))

    final_model = RandomForestClassifier(n_estimators=200, random_state=42)
    final_model.fit(X, y)

    os.makedirs(MODELS_DIR, exist_ok=True)
    out_path = os.path.join(MODELS_DIR, "fused_clean_random_forest.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({
            "model": final_model,
            "feature_columns": feature_cols,
            "feature_set": "fused",
        }, f)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
