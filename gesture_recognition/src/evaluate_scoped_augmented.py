"""
evaluate_scoped_augmented.py
----------------------------------------------------
Same leak-free methodology as evaluate_augmented.py (augment only the
training fold, after splitting), but scoped to each sensor's OWNED
gestures only (matching train_scoped_specialists.py) -- a fair,
correct comparison for the specialists actually used in the live demo.

Usage:
    python evaluate_scoped_augmented.py
"""

import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from augment_features import augment
from train_scoped_specialists import SENSOR_OWNED_GESTURES

DATA_PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    for sensor, owned_gestures in SENSOR_OWNED_GESTURES.items():
        features_path = os.path.join(DATA_PROCESSED_DIR, f"features_{sensor}.csv")
        if not os.path.exists(features_path):
            continue

        df = pd.read_csv(features_path)
        df = df[df["gesture"].isin(owned_gestures)]
        if df.empty or df["gesture"].nunique() < 2:
            continue

        train_df, test_df = train_test_split(
            df, test_size=0.3, stratify=df["gesture"], random_state=42
        )
        augmented_train_df = augment(train_df, multiplier=4, noise_std=0.05)

        feature_cols = [c for c in df.columns if c not in ("gesture", "person", "trial")]
        X_test, y_test = test_df[feature_cols], test_df["gesture"]

        setups = {
            "random_forest (real only)": (train_df, RandomForestClassifier(n_estimators=200, random_state=42)),
            "random_forest (augmented)": (augmented_train_df, RandomForestClassifier(n_estimators=200, random_state=42)),
            "mlp (real only)": (train_df, make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42))),
            "mlp (augmented)": (augmented_train_df, make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42))),
        }

        print(f"\n=== {sensor} (scoped to {sorted(owned_gestures)}) ===")
        best_acc, best_label, best_model = -1, None, None
        for label, (train_source, model) in setups.items():
            X_train, y_train = train_source[feature_cols], train_source["gesture"]
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            print(f"  {label}: {acc:.3f}")
            if acc > best_acc:
                best_acc, best_label, best_model = acc, label, model

        print(f"  BEST: {best_label} ({best_acc:.3f})")
        print(classification_report(y_test, best_model.predict(X_test), zero_division=0))

        # Retrain the winner on ALL real+augmented data (not just the
        # train split) and save it as the real deployable model.
        train_source = augment(df, multiplier=4, noise_std=0.05) if "augmented" in best_label else df
        final_model_type = "mlp" if "mlp" in best_label else "random_forest"
        final_model = (
            make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42))
            if final_model_type == "mlp"
            else RandomForestClassifier(n_estimators=200, random_state=42)
        )
        final_model.fit(train_source[feature_cols], train_source["gesture"])

        import pickle
        out_path = os.path.join(MODELS_DIR, f"{sensor}_specialist_scoped_best.pkl")
        with open(out_path, "wb") as f:
            pickle.dump({
                "model": final_model,
                "feature_columns": feature_cols,
                "feature_set": sensor,
                "owned_gestures": sorted(owned_gestures),
                "winning_setup": best_label,
            }, f)
        print(f"  Saved best model -> {out_path}")


if __name__ == "__main__":
    main()
