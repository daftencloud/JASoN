"""
combine_datasets.py
----------------------------------------------------
Merges the per-sensor-per-trial raw CSVs from data/raw/ (written by
collect.py) into one combined manifest CSV that extract_features.py
reads from: data/processed/trial_manifest.csv

Also useful for merging collection sessions from different group
members done at different times/laptops -- just make sure everyone's
data/raw/ folders get merged into one before running this (copy each
person's data/raw/<gesture>/ contents together, this script doesn't
need to change either way since it just scans whatever's present).

Usage:
    python src/combine_datasets.py
"""

import glob
import os
import re

import pandas as pd

DATA_RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
DATA_PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

FILENAME_PATTERN = re.compile(r"^(?P<person>[^_]+)_(?P<trial>\d+)_(?P<sensor>.+)\.csv$")


def main():
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

    rows = []
    for gesture_dir in sorted(glob.glob(os.path.join(DATA_RAW_DIR, "*"))):
        if not os.path.isdir(gesture_dir):
            continue
        gesture = os.path.basename(gesture_dir)

        for csv_path in sorted(glob.glob(os.path.join(gesture_dir, "*.csv"))):
            fname = os.path.basename(csv_path)
            match = FILENAME_PATTERN.match(fname)
            if not match:
                print(f"Skipping unrecognized filename: {fname}")
                continue

            person = match.group("person")
            trial = match.group("trial")
            sensor = match.group("sensor")

            # Confirm the file actually has data (collect.py writes an
            # empty file if a sensor produced zero samples in a trial).
            is_empty = os.path.getsize(csv_path) == 0

            rows.append({
                "gesture": gesture,
                "person": person,
                "trial": trial,
                "sensor": sensor,
                "raw_path": os.path.relpath(csv_path, DATA_PROCESSED_DIR),
                "empty": is_empty,
            })

    manifest_df = pd.DataFrame(rows)
    out_path = os.path.join(DATA_PROCESSED_DIR, "trial_manifest.csv")
    manifest_df.to_csv(out_path, index=False)

    print(f"Wrote manifest with {len(manifest_df)} sensor-trial entries -> {out_path}")

    if len(manifest_df) > 0:
        print("\nTrials per gesture:")
        print(manifest_df.groupby("gesture")["trial"].nunique().to_string())
        print("\nSensors seen:", sorted(manifest_df["sensor"].unique()))
        empty_count = manifest_df["empty"].sum()
        if empty_count > 0:
            print(f"\nWARNING: {empty_count} sensor-trial files are empty "
                  f"(sensor produced 0 samples) -- check those readers/wiring.")


if __name__ == "__main__":
    main()
