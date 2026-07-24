"""
convert_uwb_sessions.py
----------------------------------------------------
One-time converter: pulls real, already-collected UWB trial data from
a teammate's UWB_lab-style dataset (sessions/range_<collector>_<gesture>_trial_<N>/
folders, each with controller/ranging_samples.csv and trial_metadata.json)
and converts it into this project's standard raw-data format:

    data/raw/<gesture>/<collector>_<trial>_uwb.csv

with columns matching what extract_features.py's extract_uwb_features()
expects: timestamp_ms, distance_cm.

Usage:
    python convert_uwb_sessions.py \
        --sessions-dir ~/COSMOS-Cluster10/final_project/gesture_recognition/uwb/datasets/uwb_five_gestures/sessions \
        --output-dir ~/COSMOS-Cluster10/final_project/gesture_recognition/data/raw
"""

import argparse
import csv
import json
import os
import re

# NOTE: collector is restricted to letters only (no underscores/digits)
# so it can't accidentally "eat" part of a multi-word gesture name like
# "t_arms" or "two_arm_boxing". The gesture group is greedy but backtracks
# correctly to the one unique "_trial_" marker in each folder name.
TRIAL_FOLDER_PATTERN = re.compile(r"range_(?P<collector>[a-zA-Z]+)_(?P<gesture>.+)_trial_(?P<trial>\d+)")


def convert_trial(trial_dir, output_dir):
    folder_name = os.path.basename(trial_dir.rstrip("/"))
    match = TRIAL_FOLDER_PATTERN.match(folder_name)
    if not match:
        print(f"  Skipping (unrecognized folder name): {folder_name}")
        return False

    collector = match.group("collector")
    gesture = match.group("gesture")
    trial_num = int(match.group("trial"))

    csv_path = os.path.join(trial_dir, "controller", "ranging_samples.csv")
    if not os.path.exists(csv_path):
        print(f"  Skipping (no ranging_samples.csv): {folder_name}")
        return False

    samples = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["status"] != "Ok":
                continue  # skip RangingRxTimeout etc, same as live reader does
            samples.append({
                "timestamp_ms": float(row["time_s"]) * 1000,
                "distance_cm": float(row["distance_cm"]),
            })

    if not samples:
        print(f"  Skipping (0 valid Ok samples): {folder_name}")
        return False

    gesture_dir = os.path.join(output_dir, gesture)
    os.makedirs(gesture_dir, exist_ok=True)
    out_path = os.path.join(gesture_dir, f"{collector}_{trial_num:03d}_uwb.csv")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp_ms", "distance_cm"])
        writer.writeheader()
        writer.writerows(samples)

    print(f"  {folder_name} -> {out_path} ({len(samples)} samples)")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    sessions_dir = os.path.expanduser(args.sessions_dir)
    output_dir = os.path.expanduser(args.output_dir)

    trial_dirs = sorted([
        os.path.join(sessions_dir, d) for d in os.listdir(sessions_dir)
        if os.path.isdir(os.path.join(sessions_dir, d)) and d.startswith("range_")
    ])

    print(f"Found {len(trial_dirs)} trial folders in {sessions_dir}\n")

    converted = 0
    for trial_dir in trial_dirs:
        if convert_trial(trial_dir, output_dir):
            converted += 1

    print(f"\nConverted {converted}/{len(trial_dirs)} trials successfully.")


if __name__ == "__main__":
    main()
