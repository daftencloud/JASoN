"""
train.py
----------------------------------------------------
Trains KNN and Random Forest models on every features_*.csv found in
data/processed/ (produced by extract_features.py) -- one model per
(feature-set, model-type) combination -- and saves each trained model
to models/.

Usage:
    python src/train.py
"""

import glob
import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_features(path):
    df = pd.read_csv(path)
    X = df.drop(columns=["gesture", "person", "trial"])
    y = df["gesture"]
    groups = df["person"]
    return X, y, groups


def make_models():
    return {
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    feature_files = sorted(glob.glob(os.path.join(DATA_PROCESSED_DIR, "features_*.csv")))
    if not feature_files:
        print("No features_*.csv files found -- run extract_features.py first.")
        return

    for feature_path in feature_files:
        feature_set_name = os.path.basename(feature_path).replace("features_", "").replace(".csv", "")
        X, y, groups = load_features(feature_path)

        if X.shape[1] == 0 or len(y) == 0:
            print(f"Skipping {feature_set_name}: no usable rows/columns")
            continue

        for model_name, model in make_models().items():
            model.fit(X, y)  # train on ALL data here -- evaluate.py handles
                               # the actual train/test and held-out-person
                               # splits used for reporting accuracy
            out_path = os.path.join(MODELS_DIR, f"{feature_set_name}_{model_name}.pkl")
            with open(out_path, "wb") as f:
                pickle.dump({
                    "model": model,
                    "feature_columns": list(X.columns),
                    "feature_set": feature_set_name,
                }, f)
            print(f"Trained {feature_set_name} / {model_name} on {len(y)} trials "
                  f"-> {out_path}")


if __name__ == "__main__":
    main()
