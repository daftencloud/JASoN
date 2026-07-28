"""
plot_imu_signals.py
----------------------------------------------------
Directly follows the TA's advice: "for gestures such as push and
pull, check whether at least one axis shows the expected acceleration
pattern." Plots raw ax/ay/az signals over time for several real
trials of each gesture, so we can SEE whether push and pull are
actually separable in the data we already collected -- no new
training or data collection needed.

Usage:
    python plot_imu_signals.py --gestures push,pull --num-trials 5
    python plot_imu_signals.py --gestures clockwise,counterclockwise --num-trials 5 --sensor uwb
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

DATA_RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "figures")


def plot_imu(gestures, num_trials):
    fig, axes = plt.subplots(len(gestures), 1, figsize=(10, 4 * len(gestures)), sharex=False)
    if len(gestures) == 1:
        axes = [axes]

    for ax_plot, gesture in zip(axes, gestures):
        gesture_dir = os.path.join(DATA_RAW_DIR, gesture)
        trial_files = sorted(glob.glob(os.path.join(gesture_dir, "*_imu.csv")))[:num_trials]

        if not trial_files:
            print(f"No IMU trials found for '{gesture}'")
            continue

        for trial_path in trial_files:
            df = pd.read_csv(trial_path)
            if "ax" not in df.columns or df.empty:
                continue
            label = os.path.basename(trial_path).replace("_imu.csv", "")
            t = range(len(df))
            ax_plot.plot(t, df["ax"], alpha=0.7, label=label)

        ax_plot.set_title(f"{gesture} -- ax (accel x-axis) over time, "
                           f"{len(trial_files)} real trials")
        ax_plot.set_xlabel("Sample index")
        ax_plot.set_ylabel("ax (g)")
        ax_plot.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax_plot.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"imu_signal_check_{'_'.join(gestures)}.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved -> {out_path}")
    print("\nWhat to look for: do the two gestures show CONSISTENTLY "
          "different sign/shape on this axis across trials? If push and "
          "pull trials both look similar (same sign, same shape), that "
          "means the TRAINING DATA itself doesn't clearly separate them "
          "on this axis -- a real data quality issue, not a live-only "
          "problem. If push/pull ARE clearly different here, the issue "
          "is specific to live capture (e.g. board orientation).")


def plot_uwb(gestures, num_trials):
    fig, axes = plt.subplots(len(gestures), 1, figsize=(10, 4 * len(gestures)), sharex=False)
    if len(gestures) == 1:
        axes = [axes]

    for ax_plot, gesture in zip(axes, gestures):
        gesture_dir = os.path.join(DATA_RAW_DIR, gesture)
        trial_files = sorted(glob.glob(os.path.join(gesture_dir, "*_uwb.csv")))[:num_trials]

        if not trial_files:
            print(f"No UWB trials found for '{gesture}'")
            continue

        for trial_path in trial_files:
            df = pd.read_csv(trial_path)
            if "distance_cm" not in df.columns or df.empty:
                continue
            label = os.path.basename(trial_path).replace("_uwb.csv", "")
            t = range(len(df))
            ax_plot.plot(t, df["distance_cm"], alpha=0.7, label=label)

        ax_plot.set_title(f"{gesture} -- distance_cm over time, "
                           f"{len(trial_files)} real trials")
        ax_plot.set_xlabel("Sample index")
        ax_plot.set_ylabel("distance (cm)")
        ax_plot.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"uwb_signal_check_{'_'.join(gestures)}.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gestures", required=True, help="comma-separated, e.g. push,pull")
    parser.add_argument("--num-trials", type=int, default=5)
    parser.add_argument("--sensor", default="imu", choices=["imu", "uwb"])
    args = parser.parse_args()

    gestures = args.gestures.split(",")
    if args.sensor == "imu":
        plot_imu(gestures, args.num_trials)
    else:
        plot_uwb(gestures, args.num_trials)


if __name__ == "__main__":
    main()
