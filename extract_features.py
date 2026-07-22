"""
extract_features.py
----------------------------------------------------
Reads every raw trial CSV under data/raw/<gesture>/<person>_<n>.csv
(as written by collect_data.py) and builds one feature row per trial,
combining IMU-only features, UWB-only features, and a fused (early
fusion) feature set. Saves three CSVs to data/features/:

    imu_only_features.csv
    uwb_only_features.csv
    fused_features.csv

Each row = one trial, with columns: person, gesture, <feature columns>

Run this after you've collected trials for all gestures/people.
"""

import glob
import os

import numpy as np
import pandas as pd

RAW_DIR = "data/raw"
OUT_DIR = "data/features"


def parse_trial(path):
    """Split one trial's raw CSV back into separate IMU and UWB dataframes."""
    df = pd.read_csv(path)

    imu_rows = df[df["source"] == "imu"]["raw_line"]
    uwb_rows = df[df["source"] == "uwb"]["raw_line"]

    imu_records = []
    for line in imu_rows:
        parts = line.split(",")
        if len(parts) == 7:
            imu_records.append([float(x) for x in parts])
    imu_df = pd.DataFrame(
        imu_records, columns=["t", "ax", "ay", "az", "gx", "gy", "gz"]
    )

    # UWB lines are timestamp_ms, anchor_address, range_m -- split by anchor
    uwb_records = []
    for line in uwb_rows:
        parts = line.split(",")
        if len(parts) == 3:
            t, anchor, rng = parts
            uwb_records.append([float(t), anchor, float(rng)])
    uwb_df = pd.DataFrame(uwb_records, columns=["t", "anchor", "range_m"])

    return imu_df, uwb_df


def imu_features(imu_df):
    """Simple statistical features per axis over the whole trial window."""
    feats = {}
    for col in ["ax", "ay", "az", "gx", "gy", "gz"]:
        if len(imu_df) == 0:
            feats.update({f"imu_{col}_mean": 0, f"imu_{col}_std": 0,
                          f"imu_{col}_min": 0, f"imu_{col}_max": 0,
                          f"imu_{col}_energy": 0})
            continue
        vals = imu_df[col].values
        feats[f"imu_{col}_mean"] = np.mean(vals)
        feats[f"imu_{col}_std"] = np.std(vals)
        feats[f"imu_{col}_min"] = np.min(vals)
        feats[f"imu_{col}_max"] = np.max(vals)
        feats[f"imu_{col}_energy"] = np.sum(vals ** 2)
    return feats


def uwb_features(uwb_df):
    """Statistical features per anchor: mean/std/range and overall slope."""
    feats = {}
    anchors = sorted(uwb_df["anchor"].unique()) if len(uwb_df) else []

    # Always create slots for up to 2 anchors so every row has the same
    # columns even if one anchor briefly dropped out during a trial.
    for i in range(2):
        prefix = f"uwb_anchor{i+1}"
        if i < len(anchors):
            sub = uwb_df[uwb_df["anchor"] == anchors[i]].sort_values("t")
            r = sub["range_m"].values
            t = sub["t"].values
            feats[f"{prefix}_mean"] = np.mean(r)
            feats[f"{prefix}_std"] = np.std(r)
            feats[f"{prefix}_min"] = np.min(r)
            feats[f"{prefix}_max"] = np.max(r)
            # crude slope: total range change over the trial duration
            feats[f"{prefix}_slope"] = (
                (r[-1] - r[0]) / max(t[-1] - t[0], 1) if len(r) > 1 else 0
            )
        else:
            feats[f"{prefix}_mean"] = 0
            feats[f"{prefix}_std"] = 0
            feats[f"{prefix}_min"] = 0
            feats[f"{prefix}_max"] = 0
            feats[f"{prefix}_slope"] = 0
    return feats


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    imu_rows, uwb_rows, fused_rows = [], [], []

    for gesture_dir in sorted(glob.glob(os.path.join(RAW_DIR, "*"))):
        gesture = os.path.basename(gesture_dir)
        for trial_path in sorted(glob.glob(os.path.join(gesture_dir, "*.csv"))):
            fname = os.path.basename(trial_path)
            person = fname.split("_")[0]

            imu_df, uwb_df = parse_trial(trial_path)
            i_feats = imu_features(imu_df)
            u_feats = uwb_features(uwb_df)

            base = {"person": person, "gesture": gesture}
            imu_rows.append({**base, **i_feats})
            uwb_rows.append({**base, **u_feats})
            fused_rows.append({**base, **i_feats, **u_feats})

    pd.DataFrame(imu_rows).to_csv(os.path.join(OUT_DIR, "imu_only_features.csv"), index=False)
    pd.DataFrame(uwb_rows).to_csv(os.path.join(OUT_DIR, "uwb_only_features.csv"), index=False)
    pd.DataFrame(fused_rows).to_csv(os.path.join(OUT_DIR, "fused_features.csv"), index=False)

    print(f"Wrote {len(fused_rows)} trials to {OUT_DIR}/")


if __name__ == "__main__":
    main()
