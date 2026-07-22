"""
train_and_evaluate.py
----------------------------------------------------
Trains and compares models on the feature sets produced by
extract_features.py:

  1. IMU-only baseline  vs  fused (IMU+UWB) model      -- sensor fusion comparison
  2. KNN  vs  Random Forest                             -- model comparison
  3. Random train/test split  vs  held-out-person split -- generalization check

Prints accuracy for every combination and saves confusion matrix plots
for the two most important models (IMU-only baseline, fused model) to
outputs/.

Usage:
    python train_and_evaluate.py
"""

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

FEATURES_DIR = "data/features"
OUT_DIR = "outputs"


def load(name):
    df = pd.read_csv(os.path.join(FEATURES_DIR, name))
    X = df.drop(columns=["person", "gesture"])
    y = df["gesture"]
    groups = df["person"]
    return X, y, groups


def evaluate_random_split(X, y, model, label):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"  [{label}] random split accuracy: {acc:.3f}")
    return acc, y_test, preds


def evaluate_held_out_person(X, y, groups, model, label):
    """More realistic: train on everyone except one person, test on them."""
    people = groups.unique()
    accs = []
    for held_out in people:
        train_mask = groups != held_out
        test_mask = groups == held_out
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        model.fit(X[train_mask], y[train_mask])
        preds = model.predict(X[test_mask])
        acc = accuracy_score(y[test_mask], preds)
        accs.append(acc)
        print(f"    held out '{held_out}': accuracy = {acc:.3f}")
    mean_acc = np.mean(accs) if accs else float("nan")
    print(f"  [{label}] mean held-out-person accuracy: {mean_acc:.3f}")
    return mean_acc


def make_models():
    return {
        "KNN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }


def plot_confusion(y_test, preds, labels, title, out_path):
    cm = confusion_matrix(y_test, preds, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved confusion matrix -> {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=== IMU-only baseline ===")
    X_imu, y_imu, groups_imu = load("imu_only_features.csv")

    print("=== UWB-only ===")
    X_uwb, y_uwb, groups_uwb = load("uwb_only_features.csv")

    print("=== Fused (IMU + UWB) ===")
    X_fused, y_fused, groups_fused = load("fused_features.csv")

    models = make_models()

    results = []
    last_test_preds = {}

    for feature_name, (X, y, groups) in {
        "IMU-only": (X_imu, y_imu, groups_imu),
        "UWB-only": (X_uwb, y_uwb, groups_uwb),
        "Fused": (X_fused, y_fused, groups_fused),
    }.items():
        for model_name, model in models.items():
            label = f"{feature_name} / {model_name}"
            print(f"\n{label}")
            acc_random, y_test, preds = evaluate_random_split(X, y, model, label)
            acc_person = evaluate_held_out_person(X, y, groups, model, label)
            results.append({
                "features": feature_name,
                "model": model_name,
                "random_split_acc": acc_random,
                "held_out_person_acc": acc_person,
            })
            last_test_preds[label] = (y_test, preds, sorted(y.unique()))

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUT_DIR, "model_comparison.csv"), index=False)
    print("\n=== Summary ===")
    print(results_df.to_string(index=False))

    # Save confusion matrices for the two headline comparisons:
    # IMU-only baseline vs Fused, both with Random Forest.
    for label in ["IMU-only / Random Forest", "Fused / Random Forest"]:
        if label in last_test_preds:
            y_test, preds, labels = last_test_preds[label]
            safe_name = label.replace(" ", "_").replace("/", "_")
            plot_confusion(y_test, preds, labels, label,
                            os.path.join(OUT_DIR, f"confusion_{safe_name}.png"))


if __name__ == "__main__":
    main()
