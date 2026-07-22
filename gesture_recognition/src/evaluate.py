"""
evaluate.py
----------------------------------------------------
Evaluates every feature-set / model combination two ways:
  1. Random train/test split
  2. Held-out-person split (train on everyone except one person, test
     on them) -- the more realistic generalization check your lab
     requires

Saves a summary table to results/model_comparison.csv and confusion
matrix plots for the fused model and the best single-sensor model to
results/figures/.

Usage:
    python src/evaluate.py
"""

import glob
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

DATA_PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


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


def evaluate_random_split(X, y, model):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds), y_test, preds


def evaluate_held_out_person(X, y, groups, model):
    accs = []
    for person in groups.unique():
        train_mask = groups != person
        test_mask = groups == person
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        model.fit(X[train_mask], y[train_mask])
        preds = model.predict(X[test_mask])
        accs.append(accuracy_score(y[test_mask], preds))
    return float(np.mean(accs)) if accs else float("nan")


def plot_confusion(y_test, preds, labels, title, out_path):
    cm = confusion_matrix(y_test, preds, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(9, 9))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    feature_files = sorted(glob.glob(os.path.join(DATA_PROCESSED_DIR, "features_*.csv")))
    if not feature_files:
        print("No features_*.csv files found -- run extract_features.py first.")
        return

    results = []
    best_preds = {}

    for feature_path in feature_files:
        feature_set_name = os.path.basename(feature_path).replace("features_", "").replace(".csv", "")
        X, y, groups = load_features(feature_path)
        if X.shape[1] == 0 or len(y) == 0:
            continue

        for model_name, model in make_models().items():
            label = f"{feature_set_name}/{model_name}"
            acc_random, y_test, preds = evaluate_random_split(X, y, model)
            acc_person = evaluate_held_out_person(X, y, groups, model)
            results.append({
                "feature_set": feature_set_name,
                "model": model_name,
                "random_split_accuracy": round(acc_random, 3),
                "held_out_person_accuracy": round(acc_person, 3),
            })
            best_preds[label] = (y_test, preds, sorted(y.unique()))

    results_df = pd.DataFrame(results).sort_values(
        "held_out_person_accuracy", ascending=False
    )
    results_df.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)

    print("=== Model comparison (sorted by held-out-person accuracy) ===")
    print(results_df.to_string(index=False))

    fused_label = "fused/random_forest"
    if fused_label in best_preds:
        y_test, preds, labels = best_preds[fused_label]
        plot_confusion(y_test, preds, labels, "Fused (all sensors) - Random Forest",
                        os.path.join(FIGURES_DIR, "confusion_fused.png"))

    single_sensor_results = results_df[
        (results_df["feature_set"] != "fused") & (results_df["model"] == "random_forest")
    ]
    if not single_sensor_results.empty:
        best_single = single_sensor_results.iloc[0]["feature_set"]
        label = f"{best_single}/random_forest"
        if label in best_preds:
            y_test, preds, labels = best_preds[label]
            plot_confusion(y_test, preds, labels, f"Best single sensor: {best_single}",
                            os.path.join(FIGURES_DIR, "confusion_best_single.png"))
        print(f"\nBest single-sensor feature set: {best_single}")

    fused_acc = results_df[
        (results_df["feature_set"] == "fused") & (results_df["model"] == "random_forest")
    ]["held_out_person_accuracy"]
    if not fused_acc.empty and not single_sensor_results.empty:
        fused_val = fused_acc.iloc[0]
        best_single_val = single_sensor_results.iloc[0]["held_out_person_accuracy"]
        did_fusion_help = fused_val > best_single_val
        print(f"\nDid fusion help? Fused={fused_val:.3f} vs best single "
              f"({single_sensor_results.iloc[0]['feature_set']})={best_single_val:.3f} "
              f"-> {'YES' if did_fusion_help else 'NO'}")

    print(f"\nFull results saved to {os.path.join(RESULTS_DIR, 'model_comparison.csv')}")
    print(f"Confusion matrices saved to {FIGURES_DIR}/")
    print("\nCompare per-gesture confusion matrix rows against "
          "gestures.py's EXPECTED_STRONGEST_SENSOR table for your writeup's "
          "'which gestures are easy/hard, and did our sensor predictions hold up' section.")


if __name__ == "__main__":
    main()
