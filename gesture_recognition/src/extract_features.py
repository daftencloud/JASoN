"""
extract_features.py
----------------------------------------------------
Reads data/processed/trial_manifest.csv (from combine_datasets.py) and
computes one feature row per (person, gesture, trial), for each sensor
individually AND a fused row combining all sensors present in that
trial. Writes:

    data/processed/features_imu.csv
    data/processed/features_uwb.csv
    data/processed/features_mmwave.csv
    data/processed/features_wifi.csv
    data/processed/features_rfid.csv
    data/processed/features_fused.csv

train.py and evaluate.py read these directly.

Usage:
    python src/extract_features.py
"""

import ast
import json
import os

import numpy as np
import pandas as pd

DATA_PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MANIFEST_PATH = os.path.join(DATA_PROCESSED_DIR, "trial_manifest.csv")


def _safe_json_load(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return None


def _basic_stats(values, prefix):
    """mean/std/min/max/energy for a 1D list of numbers, NaN-safe."""
    arr = np.array([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    if arr.size == 0:
        return {f"{prefix}_mean": 0, f"{prefix}_std": 0,
                f"{prefix}_min": 0, f"{prefix}_max": 0, f"{prefix}_energy": 0}
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_energy": float(np.sum(arr ** 2)),
    }


def extract_imu_features(df: pd.DataFrame) -> dict:
    feats = {}
    for col in ["ax", "ay", "az", "gx", "gy", "gz"]:
        if col in df.columns:
            feats.update(_basic_stats(df[col].tolist(), f"imu_{col}"))
    return feats


def extract_uwb_features(df: pd.DataFrame) -> dict:
    feats = {}
    for col in ["range_anchor1", "range_anchor2"]:
        if col not in df.columns:
            continue
        vals = df[col].dropna().tolist()
        stats = _basic_stats(vals, f"uwb_{col}")
        feats.update(stats)
        # crude slope: last - first over the trial
        numeric_vals = [v for v in vals if v is not None]
        if len(numeric_vals) > 1:
            feats[f"uwb_{col}_slope"] = numeric_vals[-1] - numeric_vals[0]
        else:
            feats[f"uwb_{col}_slope"] = 0
    return feats


def extract_mmwave_features(df: pd.DataFrame) -> dict:
    all_ranges, all_dopplers, all_angles, all_elevations = [], [], [], []
    point_counts = []

    if "points" in df.columns:
        for raw in df["points"]:
            points = _safe_json_load(raw)
            if not points:
                point_counts.append(0)
                continue
            point_counts.append(len(points))
            for p in points:
                all_ranges.append(p.get("range"))
                all_dopplers.append(p.get("doppler"))
                all_angles.append(p.get("angle"))
                all_elevations.append(p.get("elevation"))

    feats = {}
    feats.update(_basic_stats(all_ranges, "mmwave_range"))
    feats.update(_basic_stats(all_dopplers, "mmwave_doppler"))
    feats.update(_basic_stats(all_angles, "mmwave_angle"))
    feats.update(_basic_stats(all_elevations, "mmwave_elevation"))
    feats.update(_basic_stats(point_counts, "mmwave_point_count"))
    return feats


def extract_wifi_features(df: pd.DataFrame) -> dict:
    feats = {}
    if "csi_amplitudes" not in df.columns:
        return feats

    all_vectors = []
    for raw in df["csi_amplitudes"]:
        vec = _safe_json_load(raw)
        if vec:
            all_vectors.append(vec)

    if not all_vectors:
        return {"wifi_variance_mean": 0, "wifi_variance_max": 0}

    # Pad/truncate to the most common length so we can stack into a matrix.
    lengths = [len(v) for v in all_vectors]
    common_len = max(set(lengths), key=lengths.count)
    matrix = np.array([v for v in all_vectors if len(v) == common_len])

    # Per-subcarrier variance over time -- same "motion score" idea as
    # your earlier CSI presence-detection lab.
    subcarrier_variances = np.var(matrix, axis=0) if matrix.shape[0] > 1 else np.zeros(common_len)
    feats["wifi_variance_mean"] = float(np.mean(subcarrier_variances))
    feats["wifi_variance_max"] = float(np.max(subcarrier_variances))
    feats["wifi_variance_std"] = float(np.std(subcarrier_variances))
    return feats


def extract_rfid_features(df: pd.DataFrame) -> dict:
    if "tags" not in df.columns:
        return {}

    per_tag_rssi = {}   # tag_id -> list of rssi
    per_tag_visible_count = {}
    total_rows = 0

    for raw in df["tags"]:
        tags = _safe_json_load(raw)
        if not tags:
            continue
        total_rows += 1
        for tag_id, info in tags.items():
            per_tag_rssi.setdefault(tag_id, [])
            per_tag_visible_count.setdefault(tag_id, 0)
            if info.get("visible"):
                per_tag_visible_count[tag_id] += 1
                if info.get("rssi") is not None:
                    per_tag_rssi[tag_id].append(info["rssi"])

    feats = {}
    tag_ids_sorted = sorted(per_tag_rssi.keys())
    occluded_count = 0
    for i, tag_id in enumerate(tag_ids_sorted):
        visible_fraction = (per_tag_visible_count[tag_id] / total_rows) if total_rows else 0
        feats.update(_basic_stats(per_tag_rssi[tag_id], f"rfid_tag{i}_rssi"))
        feats[f"rfid_tag{i}_visible_fraction"] = visible_fraction
        if visible_fraction < 0.5:
            occluded_count += 1

    feats["rfid_occluded_fraction"] = (
        occluded_count / len(tag_ids_sorted) if tag_ids_sorted else 0
    )
    return feats


EXTRACTORS = {
    "imu": extract_imu_features,
    "uwb": extract_uwb_features,
    "mmwave": extract_mmwave_features,
    "wifi": extract_wifi_features,
    "rfid": extract_rfid_features,
}


def main():
    if not os.path.exists(MANIFEST_PATH):
        print(f"No manifest found at {MANIFEST_PATH} -- run combine_datasets.py first.")
        return

    manifest = pd.read_csv(MANIFEST_PATH)
    manifest = manifest[~manifest["empty"]]  # skip sensor-trial files with 0 samples

    per_sensor_rows = {sensor: [] for sensor in EXTRACTORS}
    fused_rows = []

    grouped = manifest.groupby(["gesture", "person", "trial"])
    for (gesture, person, trial), group in grouped:
        base = {"gesture": gesture, "person": person, "trial": trial}
        fused_row = dict(base)

        for _, row in group.iterrows():
            sensor = row["sensor"]
            raw_path = os.path.join(DATA_PROCESSED_DIR, row["raw_path"])
            if not os.path.exists(raw_path) or sensor not in EXTRACTORS:
                continue

            try:
                df = pd.read_csv(raw_path)
            except pd.errors.EmptyDataError:
                continue

            feats = EXTRACTORS[sensor](df)
            per_sensor_rows[sensor].append({**base, **feats})
            fused_row.update(feats)

        fused_rows.append(fused_row)

    fused_df = pd.DataFrame(fused_rows).fillna(0)
    fused_df.to_csv(os.path.join(DATA_PROCESSED_DIR, "features_fused.csv"), index=False)
    print(f"Wrote fused features: {fused_df.shape[0]} trials, {fused_df.shape[1]} columns")

    for sensor, rows in per_sensor_rows.items():
        if not rows:
            print(f"  (no {sensor} data found -- skipping features_{sensor}.csv)")
            continue
        df = pd.DataFrame(rows).fillna(0)
        out_path = os.path.join(DATA_PROCESSED_DIR, f"features_{sensor}.csv")
        df.to_csv(out_path, index=False)
        print(f"  {sensor}: {df.shape[0]} trials, {df.shape[1]} columns -> {out_path}")


if __name__ == "__main__":
    main()
