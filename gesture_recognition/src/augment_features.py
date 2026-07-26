"""
augment_features.py
----------------------------------------------------
Data augmentation for small datasets: generates synthetic additional
training samples by adding small Gaussian noise to real feature
vectors, per gesture/person. This is a standard, low-risk technique
for boosting small datasets (recommended in the UWB lab's "Optional
Extensions" section) -- it does NOT invent new information, it just
makes the model less sensitive to exact feature values by showing it
slightly-perturbed versions of real trials.

IMPORTANT: augmentation happens AFTER the real train/test or
held-out-person split logic decides which trials are "real" data --
this script only augments a given features CSV; use it on a copy
reserved for training, never let synthetic samples leak into a test
set (that would make accuracy look artificially good).

Usage:
    python augment_features.py --input ../data/processed/features_mmwave.csv \
        --output ../data/processed/features_mmwave_augmented.csv \
        --multiplier 4 --noise-std 0.05
"""

import argparse
import os

import numpy as np
import pandas as pd


def augment(df: pd.DataFrame, multiplier: int, noise_std: float) -> pd.DataFrame:
    """
    For each real trial, generates `multiplier` synthetic variants by
    adding Gaussian noise scaled to each feature column's own standard
    deviation (so noise is proportional to each feature's natural
    scale, not a fixed absolute amount).
    """
    feature_cols = [c for c in df.columns if c not in ("gesture", "person", "trial")]
    col_stds = df[feature_cols].std().replace(0, 1e-6)  # avoid zero-noise columns

    augmented_rows = [df.copy()]  # keep all real data too

    for i in range(multiplier):
        synthetic = df.copy()
        noise = np.random.normal(
            loc=0.0, scale=noise_std, size=synthetic[feature_cols].shape
        ) * col_stds.values
        synthetic[feature_cols] = synthetic[feature_cols].values + noise
        synthetic["trial"] = synthetic["trial"].astype(str) + f"_aug{i+1}"
        augmented_rows.append(synthetic)

    return pd.concat(augmented_rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--multiplier", type=int, default=4,
                         help="How many synthetic copies to generate per real trial")
    parser.add_argument("--noise-std", type=float, default=0.05,
                         help="Noise standard deviation as a fraction of each "
                              "feature's own std (0.05 = 5%% of natural variation)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} real trials from {args.input}")

    augmented_df = augment(df, args.multiplier, args.noise_std)
    print(f"Generated {len(augmented_df) - len(df)} synthetic trials "
          f"(total: {len(augmented_df)})")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    augmented_df.to_csv(args.output, index=False)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
