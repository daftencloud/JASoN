"""
evaluate_augmented.py
----------------------------------------------------
Proper augmented-training evaluation: augmentation happens AFTER the
train/test split, and ONLY on the training fold. The test fold always
stays 100% real data -- otherwise accuracy numbers would be
meaningless (testing on near-duplicates of training data makes any
model look artificially perfect).

Compares, for each feature set (imu, uwb, mmwave, fused):
  - KNN on real data only (baseline, matches evaluate.py)
  - Random Forest on real data only (baseline, matches evaluate.py)
  - Random Forest on augmented training data (does augmentation help?)
  - MLP on real data only (the "additional classifier" your lab requires)
  - MLP on augmented training data (does augmentation help the model
    most likely to need more data?)

Usage:
    python evaluate_augmented.py
"""

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from augment_features import augment

DATA_PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

FEATURE_FILES = {
    "imu": "features_imu.csv",
    "uwb": "features_uwb.csv",
    "mmwave": "features_mmwave.csv",
    "fused": "features_fused.csv",
}


def load(path):
    df = pd.read_csv(path)
    return df


def to_xy(df):
    feature_cols = [c for c in df.columns if c not in ("gesture", "person", "trial")]
    return df[feature_cols], df["gesture"]


def safe_knn_k(y, default=5):
    min_class_count = y.value_counts().min()
    return max(1, min(default, min_class_count - 1))


def evaluate_random_split(train_df, test_df, model_builder):
    X_train, y_train = to_xy(train_df)
    X_test, y_test = to_xy(test_df)
    model = model_builder(y_train)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = []

    for feature_name, filename in FEATURE_FILES.items():
        path = os.path.join(DATA_PROCESSED_DIR, filename)
        if not os.path.exists(path):
            continue

        df = load(path)
        if df.shape[0] < 10 or df["gesture"].nunique() < 2:
            print(f"Skipping {feature_name}: not enough data/classes")
            continue

        # Split REAL data into train/test first -- augmentation happens
        # only after this split, and only on the train side.
        train_df, test_df = train_test_split(
            df, test_size=0.25, stratify=df["gesture"], random_state=42
        )

        augmented_train_df = augment(train_df, multiplier=4, noise_std=0.05)

        models = {
            "knn (real only)": lambda y: make_pipeline(
                StandardScaler(), KNeighborsClassifier(n_neighbors=safe_knn_k(y))
            ),
            "random_forest (real only)": lambda y: RandomForestClassifier(
                n_estimators=200, random_state=42
            ),
            "random_forest (augmented train)": lambda y: RandomForestClassifier(
                n_estimators=200, random_state=42
            ),
            "mlp (real only)": lambda y: make_pipeline(
                StandardScaler(),
                MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42),
            ),
            "mlp (augmented train)": lambda y: make_pipeline(
                StandardScaler(),
                MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42),
            ),
        }

        for model_label, builder in models.items():
            use_augmented = "augmented" in model_label
            train_source = augmented_train_df if use_augmented else train_df

            try:
                acc = evaluate_random_split(train_source, test_df, builder)
            except Exception as e:
                print(f"  {feature_name} / {model_label}: FAILED ({e})")
                continue

            results.append({
                "feature_set": feature_name,
                "model": model_label,
                "test_accuracy": round(acc, 3),
                "train_trials": len(train_source),
                "test_trials": len(test_df),
            })
            print(f"  {feature_name} / {model_label}: {acc:.3f} "
                  f"(trained on {len(train_source)}, tested on {len(test_df)} real trials)")

    results_df = pd.DataFrame(results)
    out_path = os.path.join(RESULTS_DIR, "augmented_comparison.csv")
    results_df.to_csv(out_path, index=False)

    print(f"\n=== Full comparison (sorted by feature_set, then accuracy) ===")
    print(results_df.sort_values(["feature_set", "test_accuracy"], ascending=[True, False])
          .to_string(index=False))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
