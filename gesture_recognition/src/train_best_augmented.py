"""
train_best_augmented.py
----------------------------------------------------
Trains and SAVES the best-performing augmented setup as a real,
loadable model -- unlike evaluate_augmented.py, which only compares
accuracy numbers and never saves a usable .pkl.

Supports both Random Forest and MLP (use whichever evaluate_augmented.py
showed as the real winner for a given feature set -- don't assume MLP
is always best, check the actual numbers first).

This trains on ALL real data plus augmented synthetic variants, since
for the final saved model you want it to learn from everything
available -- the held-back test slice already happened in
evaluate_augmented.py, this script is for producing the actual demo-
ready model.

Usage:
    python train_best_augmented.py --feature-set mmwave --model-type random_forest
    python train_best_augmented.py --feature-set uwb --model-type random_forest
    python train_best_augmented.py --feature-set fused --model-type mlp
"""

import argparse
import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from augment_features import augment

DATA_PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def build_model(model_type):
    if model_type == "random_forest":
        return RandomForestClassifier(n_estimators=200, random_state=42)
    elif model_type == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-set", required=True,
                         choices=["imu", "uwb", "mmwave", "fused"])
    parser.add_argument("--model-type", required=True,
                         choices=["random_forest", "mlp"])
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

    model = build_model(args.model_type)
    model.fit(X, y)

    os.makedirs(MODELS_DIR, exist_ok=True)
    out_path = os.path.join(MODELS_DIR, f"{args.feature_set}_{args.model_type}_augmented.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({
            "model": model,
            "feature_columns": feature_cols,
            "feature_set": args.feature_set,
        }, f)

    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
