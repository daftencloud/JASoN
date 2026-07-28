"""
train_scoped_specialists.py
----------------------------------------------------
Trains each sensor's specialist model using ONLY the gestures that
sensor is actually assigned to own (per the team's real hardware
spreadsheet), instead of every gesture that happens to have non-zero
data for that sensor.

WHY THIS MATTERS: when a gesture was collected with all 3 sensors
running simultaneously (e.g. Jon's clockwise/clapping/left/right/
bye_bye data), each sensor picks up SOME real signal even for
gestures it wasn't designed to detect (your arm does move a little
during a UWB "clockwise" gesture, so IMU sees something real, not
noise). If you train the IMU specialist on every gesture with any IMU
data, it genuinely learns to recognize gestures it should never be
asked about -- that's not a routing bug, it's real learned behavior
from cross-contaminated training data. This script fixes that at the
source: IMU's specialist literally can't predict "clockwise" anymore,
because "clockwise" was never one of its training classes.

Usage:
    python train_scoped_specialists.py
"""

import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

DATA_PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# The real, single-sensor-per-gesture design from the team spreadsheet.
SENSOR_OWNED_GESTURES = {
    "imu": {"pull", "push", "left", "right", "bye_bye", "one_arm_boxing", "rest"},
    "uwb": {"clockwise", "counterclockwise", "clapping", "two_arm_boxing", "t_arm"},
    "mmwave": {"raise_arms", "soli", "open_close_fist", "palm_up_down"},
}


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    for sensor, owned_gestures in SENSOR_OWNED_GESTURES.items():
        features_path = os.path.join(DATA_PROCESSED_DIR, f"features_{sensor}.csv")
        if not os.path.exists(features_path):
            print(f"Skipping {sensor}: {features_path} not found")
            continue

        df = pd.read_csv(features_path)
        before_count = len(df)
        df = df[df["gesture"].isin(owned_gestures)]
        after_count = len(df)

        print(f"\n{sensor}: {before_count} total trials -> {after_count} trials "
              f"after restricting to its {len(owned_gestures)} owned gestures")
        print(f"  Owned gestures: {sorted(owned_gestures)}")
        print(f"  Gestures actually present: {sorted(df['gesture'].unique())}")

        if df.empty or df["gesture"].nunique() < 2:
            print(f"  Not enough data/classes to train -- skipping")
            continue

        feature_cols = [c for c in df.columns if c not in ("gesture", "person", "trial")]
        X = df[feature_cols]
        y = df["gesture"]

        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X, y)

        out_path = os.path.join(MODELS_DIR, f"{sensor}_specialist_scoped.pkl")
        with open(out_path, "wb") as f:
            pickle.dump({
                "model": model,
                "feature_columns": feature_cols,
                "feature_set": sensor,
                "owned_gestures": sorted(owned_gestures),
            }, f)
        print(f"  Saved -> {out_path}")


if __name__ == "__main__":
    main()
