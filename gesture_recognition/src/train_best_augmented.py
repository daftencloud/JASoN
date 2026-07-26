"""
train_best_augmented.py
----------------------------------------------------
Trains and SAVES the best-performing setup from evaluate_augmented.py's
comparison (MLP on augmented training data) as a real, loadable model
-- unlike evaluate_augmented.py, which only compares accuracy numbers
and never saves a usable .pkl.

This trains on ALL real data (not just a train split) plus augmented
synthetic variants, since for the final saved model you want it to
learn from everything available, not hold back a test slice (that
held-back testing already happened in evaluate_augmented.py).

Usage:
    python train_best_augmented.py --feature-set fused
    python train_best_augmented.py --feature-set imu
"""

import argparse
import os
import pickle

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from augment_features import augment

DATA_PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-set", required=True,
                         choices=["imu", "uwb", "mmwave", "fused"])
    parser.add_argument("--multiplier", type=int, default=4)
    parser.add_argument("--noise-std", type=float, default=0.05)
    args = parser.parse_args()

    features_path = os.path.join(DATA_PROCESSED_DIR, f"features_{args.feature_set}.csv")
    df = pd.read_csv(features_path)

    augmented_df = augment(df, multiplier=args.multiplier, noise_std=args.noise_std)
    print(f"Training on {len(df)} real + {len(augmented_df) - len(df)} "
          f"synthetic trials ({len(augmented_df)} total)")

    feature_cols = [c for c in augmented_df.columns if c not in ("gesture", "person", "trial")]
    X = augmented_df[feature_cols]
    y = augmented_df["gesture"]

    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42),
    )
    model.fit(X, y)

    os.makedirs(MODELS_DIR, exist_ok=True)
    out_path = os.path.join(MODELS_DIR, f"{args.feature_set}_mlp_augmented.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({
            "model": model,
            "feature_columns": feature_cols,
            "feature_set": args.feature_set,
        }, f)

    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
