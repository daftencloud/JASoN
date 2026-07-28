"""
evaluate_clean_fused_augmented.py
----------------------------------------------------
Leak-free augmentation comparison (same methodology as
evaluate_augmented.py -- augment only the training fold, after
splitting) specifically for the CLEAN fused dataset (only trials
where all 3 sensors genuinely recorded real data, no zero-padding).

Compares KNN, RF, RF+augmented, MLP, MLP+augmented, and saves whichever
wins as the real deployable model.

Usage:
    python evaluate_clean_fused_augmented.py
"""

import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from augment_features import augment

DATA_PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def safe_knn_k(y, default=5):
    min_class_count = y.value_counts().min()
    return max(1, min(default, min_class_count - 1))


def main():
    df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "features_fused.csv"))

    imu_col = next((c for c in df.columns if c.startswith("imu_")), None)
    uwb_col = next((c for c in df.columns if c.startswith("uwb_")), None)
    mmwave_col = next((c for c in df.columns if c.startswith("mmwave_")), None)
    clean_df = df[(df[imu_col] != 0) & (df[uwb_col] != 0) & (df[mmwave_col] != 0)]

    print(f"Clean fused dataset: {len(clean_df)} trials, "
          f"{clean_df['gesture'].nunique()} gestures")

    feature_cols = [c for c in clean_df.columns if c not in ("gesture", "person", "trial")]

    train_df, test_df = train_test_split(
        clean_df, test_size=0.25, stratify=clean_df["gesture"], random_state=42
    )
    augmented_train_df = augment(train_df, multiplier=4, noise_std=0.05)

    X_test, y_test = test_df[feature_cols], test_df["gesture"]

    setups = {
        "knn (real only)": (train_df, make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=safe_knn_k(train_df["gesture"])))),
        "random_forest (real only)": (train_df, RandomForestClassifier(n_estimators=200, random_state=42)),
        "random_forest (augmented)": (augmented_train_df, RandomForestClassifier(n_estimators=200, random_state=42)),
        "mlp (real only)": (train_df, make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42))),
        "mlp (augmented)": (augmented_train_df, make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42))),
    }

    best_acc, best_label, best_model = -1, None, None
    for label, (train_source, model) in setups.items():
        X_train, y_train = train_source[feature_cols], train_source["gesture"]
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"  {label}: {acc:.3f} (trained on {len(train_source)}, tested on {len(test_df)} real trials)")
        if acc > best_acc:
            best_acc, best_label, best_model = acc, label, model

    print(f"\nBEST: {best_label} ({best_acc:.3f})")
    print(classification_report(y_test, best_model.predict(X_test), zero_division=0))

    # Retrain winner on ALL clean data for the final deployable model.
    final_train_source = augment(clean_df, multiplier=4, noise_std=0.05) if "augmented" in best_label else clean_df
    if "mlp" in best_label:
        final_model = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42))
    else:
        final_model = RandomForestClassifier(n_estimators=200, random_state=42)
    final_model.fit(final_train_source[feature_cols], final_train_source["gesture"])

    os.makedirs(MODELS_DIR, exist_ok=True)
    out_path = os.path.join(MODELS_DIR, "fused_clean_best.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({
            "model": final_model,
            "feature_columns": feature_cols,
            "feature_set": "fused",
            "winning_setup": best_label,
        }, f)
    print(f"\nSaved best model -> {out_path}")


if __name__ == "__main__":
    main()
