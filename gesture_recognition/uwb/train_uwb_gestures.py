from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_GESTURES = [
    "clockwise",
    "anticlockwise",
    "clapping",
    "two_arm_boxing",
    "t_arms",
]


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a UWB inter-wrist range gesture classifier."
    )
    parser.add_argument(
        "datasets",
        nargs="+",
        help="Dataset folders containing trials.csv and trial session folders.",
    )
    parser.add_argument(
        "--gestures",
        nargs="+",
        default=DEFAULT_GESTURES,
        help="Gesture labels to retain from trials.csv.",
    )
    parser.add_argument(
        "--side",
        choices=["controller", "controlee"],
        default="controller",
        help="Which board's saved range CSV to use.",
    )
    parser.add_argument(
        "--feature-set",
        choices=["baseline", "proposal"],
        default="proposal",
        help="Proposal adds movement, valley, periodicity, and close-distance features.",
    )
    parser.add_argument("--resample-points", type=int, default=60)
    parser.add_argument("--feature-fs", type=float, default=50.0)
    parser.add_argument("--close-threshold-cm", type=float, default=25.0)
    parser.add_argument("--max-distance-cm", type=float, default=1000.0)
    parser.add_argument("--mad-z", type=float, default=5.0)
    parser.add_argument(
        "--classifier",
        choices=[
            "random_forest",
            "svm_rbf",
            "svm_linear",
            "svm_poly",
            "knn",
            "decision_tree",
        ],
        default="random_forest",
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--svm-c", type=float, default=3.0)
    parser.add_argument("--knn-neighbors", type=int, default=5)
    parser.add_argument(
        "--test-collector",
        action="append",
        help="Hold out this collector for testing. Repeat or use comma-separated names.",
    )
    parser.add_argument(
        "--model-out",
        help="Output .joblib path. Default: models/uwb_gesture_model_<timestamp>.joblib",
    )
    parser.add_argument(
        "--confusion-out",
        help="Output confusion matrix PNG path. Default: beside the model.",
    )
    return parser.parse_args()


def normalize_label(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def normalize_filter(values: Sequence[str] | None) -> set[str] | None:
    if not values:
        return None
    result: set[str] = set()
    for value in values:
        for part in value.split(","):
            part = normalize_label(part)
            if part:
                result.add(part)
    return result or None


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_manifest(dataset_dir: Path) -> list[dict[str, str]]:
    manifest = dataset_dir / "trials.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing trials.csv: {manifest}")
    with manifest.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["_dataset_dir"] = str(dataset_dir)
    return rows


def resolve_session_dir(row: dict[str, str]) -> Path:
    session = Path(row.get("session_dir", ""))
    if session.is_absolute():
        return session
    return (Path(row["_dataset_dir"]) / session).resolve()


def find_ranging_csv(session_dir: Path, side: str) -> Path | None:
    candidates = [
        session_dir / side / "ranging_samples.csv",
        session_dir / f"{side}_ranging_samples.csv",
        session_dir / "ranging_samples.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    recursive = sorted(session_dir.rglob("ranging_samples.csv"))
    for candidate in recursive:
        if side in candidate.parts:
            return candidate
    return recursive[0] if recursive else None


def read_ranging_samples(path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8-sig") as file:
        for index, row in enumerate(csv.DictReader(file)):
            distance = safe_float(row.get("distance_cm"))
            if distance is None:
                continue
            status = row.get("status", "Ok") or "Ok"
            sequence = safe_float(row.get("sequence"), index)
            interval = safe_float(row.get("interval_ms"))
            time_s = safe_float(row.get("time_s"))
            samples.append(
                {
                    "distance_cm": distance,
                    "status": status,
                    "sequence": int(sequence) if sequence is not None else index,
                    "interval_ms": interval,
                    "time_s": time_s,
                }
            )
    return samples


def valid_distances(
    samples: Iterable[dict[str, Any]], max_distance_cm: float, mad_z: float
) -> list[float]:
    values = [
        float(sample["distance_cm"])
        for sample in samples
        if str(sample.get("status", "Ok")).lower() == "ok"
        and 0.0 < float(sample["distance_cm"]) <= max_distance_cm
    ]
    if len(values) < 5:
        return values

    ordered = sorted(values)
    median = ordered[len(ordered) // 2]
    deviations = sorted(abs(value - median) for value in values)
    mad = deviations[len(deviations) // 2]
    if mad <= 1e-9:
        return values

    scale = 1.4826 * mad
    return [value for value in values if abs(value - median) / scale <= mad_z]


def resample(values: Sequence[float], points: int) -> list[float]:
    import numpy as np

    if points < 2:
        raise ValueError("resample-points must be at least 2")
    if not values:
        return [0.0] * points
    if len(values) == 1:
        return [float(values[0])] * points

    source_x = np.linspace(0.0, 1.0, len(values))
    target_x = np.linspace(0.0, 1.0, points)
    return np.interp(target_x, source_x, values).astype(float).tolist()


def basic_stats(arr: "Any") -> list[float]:
    import numpy as np

    return [
        float(np.mean(arr)),
        float(np.std(arr)),
        float(np.min(arr)),
        float(np.max(arr)),
        float(np.median(arr)),
        float(np.percentile(arr, 25)),
        float(np.percentile(arr, 75)),
        float(np.ptp(arr)),
    ]


def count_local_minima(arr: "Any", min_prominence: float = 3.0) -> tuple[int, list[int], list[float]]:
    import numpy as np

    indices: list[int] = []
    prominences: list[float] = []
    for index in range(1, len(arr) - 1):
        if arr[index] <= arr[index - 1] and arr[index] < arr[index + 1]:
            local_high = min(float(arr[index - 1]), float(arr[index + 1]))
            prominence = local_high - float(arr[index])
            if prominence >= min_prominence:
                indices.append(index)
                prominences.append(prominence)
    return len(indices), indices, prominences


def baseline_feature_names(points: int) -> list[str]:
    names = [
        "mean_cm", "std_cm", "min_cm", "max_cm", "median_cm",
        "q25_cm", "q75_cm", "range_cm",
        "start_cm", "end_cm", "end_minus_start_cm",
        "mean_abs_velocity", "max_abs_velocity", "velocity_std",
        "total_movement_cm", "direction_changes",
    ]
    names.extend(f"shape_{index:03d}" for index in range(points))
    return names


def proposal_feature_names(points: int) -> list[str]:
    names = baseline_feature_names(points)
    names.extend(
        [
            "close_fraction",
            "linear_slope",
            "trend_r2",
            "first_last_third_delta",
            "increasing_fraction",
            "decreasing_fraction",
            "valley_count",
            "mean_valley_prominence",
            "max_valley_prominence",
            "mean_valley_interval_s",
            "valley_interval_cv",
            "dominant_frequency_hz",
            "spectral_concentration",
            "autocorrelation_peak",
            "autocorrelation_peak_lag_s",
        ]
    )
    return names


def extract_features(
    values: Sequence[float],
    points: int,
    feature_set: str,
    fs: float,
    close_threshold_cm: float,
) -> list[float]:
    import numpy as np

    shape = np.asarray(resample(values, points), dtype=float)
    velocity = np.diff(shape)
    acceleration = np.diff(velocity)

    direction_changes = 0
    if len(velocity) >= 2:
        signs = np.sign(velocity)
        signs = signs[signs != 0]
        if len(signs) >= 2:
            direction_changes = int(np.sum(signs[1:] != signs[:-1]))

    features = basic_stats(shape)
    features.extend(
        [
            float(shape[0]),
            float(shape[-1]),
            float(shape[-1] - shape[0]),
            float(np.mean(np.abs(velocity))) if len(velocity) else 0.0,
            float(np.max(np.abs(velocity))) if len(velocity) else 0.0,
            float(np.std(velocity)) if len(velocity) else 0.0,
            float(np.sum(np.abs(velocity))) if len(velocity) else 0.0,
            float(direction_changes),
        ]
    )
    features.extend(shape.astype(float).tolist())

    if feature_set == "baseline":
        return features

    x = np.linspace(0.0, 1.0, len(shape))
    slope, intercept = np.polyfit(x, shape, 1)
    fitted = slope * x + intercept
    residual_ss = float(np.sum((shape - fitted) ** 2))
    total_ss = float(np.sum((shape - np.mean(shape)) ** 2))
    trend_r2 = 1.0 - residual_ss / total_ss if total_ss > 1e-12 else 0.0

    third = max(1, len(shape) // 3)
    first_last_delta = float(np.mean(shape[-third:]) - np.mean(shape[:third]))
    increasing_fraction = float(np.mean(velocity > 0)) if len(velocity) else 0.0
    decreasing_fraction = float(np.mean(velocity < 0)) if len(velocity) else 0.0

    prominence_threshold = max(2.0, 0.08 * float(np.ptp(shape)))
    valley_count, valley_indices, prominences = count_local_minima(
        shape, min_prominence=prominence_threshold
    )
    valley_intervals = np.diff(valley_indices) / fs if len(valley_indices) >= 2 else np.asarray([])
    mean_interval = float(np.mean(valley_intervals)) if len(valley_intervals) else 0.0
    interval_cv = (
        float(np.std(valley_intervals) / np.mean(valley_intervals))
        if len(valley_intervals) and np.mean(valley_intervals) > 1e-12
        else 0.0
    )

    centered = shape - np.mean(shape)
    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    frequencies = np.fft.rfftfreq(len(centered), d=1.0 / fs)
    if len(spectrum) > 1:
        non_dc = spectrum.copy()
        non_dc[0] = 0.0
        dominant_index = int(np.argmax(non_dc))
        dominant_frequency = float(frequencies[dominant_index])
        spectral_concentration = float(non_dc[dominant_index] / max(np.sum(non_dc), 1e-12))
    else:
        dominant_frequency = 0.0
        spectral_concentration = 0.0

    if np.std(centered) > 1e-9:
        autocorr = np.correlate(centered, centered, mode="full")[len(centered) - 1 :]
        autocorr = autocorr / max(float(autocorr[0]), 1e-12)
        min_lag = max(1, int(round(0.15 * fs)))
        if min_lag < len(autocorr):
            search = autocorr[min_lag:]
            peak_offset = int(np.argmax(search))
            autocorrelation_peak = float(search[peak_offset])
            autocorrelation_peak_lag = float((min_lag + peak_offset) / fs)
        else:
            autocorrelation_peak = 0.0
            autocorrelation_peak_lag = 0.0
    else:
        autocorrelation_peak = 0.0
        autocorrelation_peak_lag = 0.0

    features.extend(
        [
            float(np.mean(shape <= close_threshold_cm)),
            float(slope),
            float(trend_r2),
            first_last_delta,
            increasing_fraction,
            decreasing_fraction,
            float(valley_count),
            float(np.mean(prominences)) if prominences else 0.0,
            float(np.max(prominences)) if prominences else 0.0,
            mean_interval,
            interval_cv,
            dominant_frequency,
            spectral_concentration,
            autocorrelation_peak,
            autocorrelation_peak_lag,
        ]
    )
    return features


def build_classifier(args: argparse.Namespace, train_count: int):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    if args.classifier.startswith("svm_"):
        kernel = args.classifier.removeprefix("svm_")
        params = {
            "kernel": kernel,
            "C": args.svm_c,
            "class_weight": "balanced",
            "probability": True,
            "random_state": args.random_state,
        }
        return make_pipeline(StandardScaler(), SVC(**params)), params

    if args.classifier == "knn":
        neighbors = min(max(1, args.knn_neighbors), train_count)
        params = {"n_neighbors": neighbors, "weights": "distance"}
        return make_pipeline(StandardScaler(), KNeighborsClassifier(**params)), params

    if args.classifier == "random_forest":
        params = {
            "n_estimators": 500,
            "class_weight": "balanced_subsample",
            "random_state": args.random_state,
            "n_jobs": -1,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
        }
        return RandomForestClassifier(**params), params

    params = {
        "class_weight": "balanced",
        "random_state": args.random_state,
        "min_samples_leaf": 2,
    }
    return DecisionTreeClassifier(**params), params


def main() -> int:
    args = parse_args()

    try:
        import joblib
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.metrics import (
            ConfusionMatrixDisplay,
            accuracy_score,
            balanced_accuracy_score,
            classification_report,
            confusion_matrix,
        )
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        print(
            "Missing dependency. Install with:\n"
            "  python -m pip install numpy scikit-learn matplotlib joblib\n"
            f"Original error: {exc}",
            file=sys.stderr,
        )
        return 2

    allowed_gestures = {normalize_label(label) for label in args.gestures}
    held_out_collectors = normalize_filter(args.test_collector)
    dataset_dirs = [Path(path).expanduser().resolve() for path in args.datasets]

    rows: list[dict[str, str]] = []
    for dataset_dir in dataset_dirs:
        try:
            rows.extend(read_manifest(dataset_dir))
        except FileNotFoundError as exc:
            print(f"Warning: {exc}", file=sys.stderr)

    X: list[list[float]] = []
    y: list[str] = []
    collectors: list[str] = []
    used_rows: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []

    for row in rows:
        gesture = normalize_label(row.get("gesture", ""))
        if gesture not in allowed_gestures:
            continue

        session_dir = resolve_session_dir(row)
        sample_path = find_ranging_csv(session_dir, args.side)
        if sample_path is None:
            skipped.append({"gesture": gesture, "session": str(session_dir), "reason": "missing range CSV"})
            continue

        samples = read_ranging_samples(sample_path)
        distances = valid_distances(samples, args.max_distance_cm, args.mad_z)
        if len(distances) < 10:
            skipped.append({"gesture": gesture, "session": str(session_dir), "reason": f"only {len(distances)} valid samples"})
            continue

        feature_vector = extract_features(
            distances,
            points=args.resample_points,
            feature_set=args.feature_set,
            fs=args.feature_fs,
            close_threshold_cm=args.close_threshold_cm,
        )
        if not all(math.isfinite(value) for value in feature_vector):
            skipped.append({"gesture": gesture, "session": str(session_dir), "reason": "non-finite feature"})
            continue

        X.append(feature_vector)
        y.append(gesture)
        collectors.append(normalize_label(row.get("collector", "unknown")))
        used_rows.append(row)

    if not X:
        print("No usable trials were found.", file=sys.stderr)
        return 1

    counts = Counter(y)
    missing = sorted(allowed_gestures - set(counts))
    print("Usable trials by gesture:")
    for gesture in sorted(counts):
        print(f"  {gesture}: {counts[gesture]}")
    if missing:
        print(f"Missing gestures: {', '.join(missing)}", file=sys.stderr)
        return 1
    if min(counts.values()) < 2:
        print("Each gesture needs at least two usable trials.", file=sys.stderr)
        return 1

    X_array = np.asarray(X, dtype=float)
    y_array = np.asarray(y)
    collector_array = np.asarray(collectors)

    if held_out_collectors:
        test_mask = np.asarray([collector in held_out_collectors for collector in collector_array])
        train_mask = ~test_mask
        if not np.any(test_mask) or not np.any(train_mask):
            print("Collector holdout produced an empty training or test set.", file=sys.stderr)
            return 1
        X_train, X_test = X_array[train_mask], X_array[test_mask]
        y_train, y_test = y_array[train_mask], y_array[test_mask]
        split_method = {"type": "collector_holdout", "collectors": sorted(held_out_collectors)}
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_array,
            y_array,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y_array,
        )
        split_method = {"type": "stratified_random", "test_size": args.test_size}

    model, classifier_params = build_classifier(args, len(X_train))
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = float(accuracy_score(y_test, predictions))
    balanced_accuracy = float(balanced_accuracy_score(y_test, predictions))
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    labels = sorted(allowed_gestures)
    matrix = confusion_matrix(y_test, predictions, labels=labels)

    model_path = (
        Path(args.model_out).expanduser().resolve()
        if args.model_out
        else (Path(__file__).resolve().parent / "models" / f"uwb_gesture_model_{timestamp()}.joblib")
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    confusion_path = (
        Path(args.confusion_out).expanduser().resolve()
        if args.confusion_out
        else model_path.with_name(model_path.stem + "_confusion.png")
    )

    feature_names = (
        proposal_feature_names(args.resample_points)
        if args.feature_set == "proposal"
        else baseline_feature_names(args.resample_points)
    )

    payload = {
        "model": model,
        "classes": labels,
        "feature_set": args.feature_set,
        "feature_names": feature_names,
        "feature_params": {
            "resample_points": args.resample_points,
            "feature_fs": args.feature_fs,
            "close_threshold_cm": args.close_threshold_cm,
            "max_distance_cm": args.max_distance_cm,
            "mad_z": args.mad_z,
        },
        "classifier": args.classifier,
        "classifier_params": classifier_params,
        "side": args.side,
        "training_datasets": [str(path) for path in dataset_dirs],
        "metrics": {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "classification_report": report,
            "confusion_matrix": matrix.tolist(),
            "labels": labels,
        },
        "split": split_method,
        "training_trial_count": int(len(X_train)),
        "test_trial_count": int(len(X_test)),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    joblib.dump(payload, model_path)

    metadata_path = model_path.with_suffix(".json")
    json_payload = {key: value for key, value in payload.items() if key != "model"}
    json_payload["usable_trials_by_gesture"] = dict(sorted(counts.items()))
    json_payload["skipped_trials"] = skipped
    metadata_path.write_text(json.dumps(json_payload, indent=2, sort_keys=True) + "\n")

    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    display.plot(xticks_rotation=30, values_format="d")
    plt.title(f"UWB Gesture Classification — {args.classifier}")
    plt.tight_layout()
    plt.savefig(confusion_path, dpi=180)
    plt.close()

    print(f"\nAccuracy:          {accuracy:.3f}")
    print(f"Balanced accuracy: {balanced_accuracy:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print(f"Model:             {model_path}")
    print(f"Metadata:          {metadata_path}")
    print(f"Confusion matrix:  {confusion_path}")
    if skipped:
        print(f"Skipped trials:    {len(skipped)} (details in metadata JSON)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())