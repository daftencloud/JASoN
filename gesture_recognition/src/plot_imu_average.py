"""
plot_imu_average.py
----------------------------------------------------
Plots the AVERAGE signal (across multiple trials) for two gestures,
overlaid on the same axes with clear labels -- much easier to
interpret than overlapping individual trial lines, good for a poster
figure specifically.

Since trials have slightly different lengths, this resamples each
trial to a common length (100 points) before averaging, so trials of
different durations can be combined meaningfully.

Usage:
    python plot_imu_average.py --gestures push,pull --num-trials 5 --column ax
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "figures")

COLORS = {0: "#2774AE", 1: "#FFD100"}  # UCLA blue, gold -- matches poster


def resample(values, target_len=100):
    x_old = np.linspace(0, 1, len(values))
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gestures", required=True, help="comma-separated, e.g. push,pull")
    parser.add_argument("--num-trials", type=int, default=5)
    parser.add_argument("--column", default="ax")
    args = parser.parse_args()

    gestures = args.gestures.split(",")

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, gesture in enumerate(gestures):
        gesture_dir = os.path.join(DATA_RAW_DIR, gesture)
        trial_files = sorted(glob.glob(os.path.join(gesture_dir, "*_imu.csv")))[:args.num_trials]

        resampled_trials = []
        for trial_path in trial_files:
            df = pd.read_csv(trial_path)
            if args.column not in df.columns or df.empty:
                continue
            resampled_trials.append(resample(df[args.column].values))

        if not resampled_trials:
            print(f"No valid trials found for '{gesture}'")
            continue

        stacked = np.array(resampled_trials)
        mean_signal = stacked.mean(axis=0)
        std_signal = stacked.std(axis=0)
        x = np.linspace(0, 100, 100)

        color = COLORS.get(i, None)
        ax.plot(x, mean_signal, label=f"{gesture} (avg of {len(resampled_trials)} trials)",
                color=color, linewidth=2.5)
        ax.fill_between(x, mean_signal - std_signal, mean_signal + std_signal,
                         color=color, alpha=0.15)

    ax.set_title(f"Average {args.column} signal: {' vs '.join(gestures)}", fontsize=14)
    ax.set_xlabel("Normalized time (% of trial)")
    ax.set_ylabel(f"{args.column} (g)")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.legend(fontsize=11)
    fig.tight_layout()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"imu_average_{args.column}_{'_'.join(gestures)}.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
