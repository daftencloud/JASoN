#!/usr/bin/env python3

import argparse
import subprocess
import threading
import time
from pathlib import Path

import joblib
import numpy as np

from train_uwb_gestures import extract_features, valid_distances
from uwb_lab_common import (
    LAB_DIR,
    RangeLogParser,
    process_env,
    stop_process,
    twr_command,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run interactive live UWB gesture predictions."
    )

    parser.add_argument("--controller-port", required=True)
    parser.add_argument("--controlee-port", required=True)
    parser.add_argument("--model", required=True)

    parser.add_argument("--preamble-code", type=int, default=9)
    parser.add_argument("--channel", type=int, choices=[5, 9], default=5)

    parser.add_argument("--slot-span", type=int, default=2400)
    parser.add_argument("--slots-per-rr", type=int, default=6)
    parser.add_argument("--ranging-span", type=int, default=20)

    parser.add_argument("--window-seconds", type=float, default=3.0)
    parser.add_argument("--countdown", type=int, default=2)
    parser.add_argument("--startup-delay", type=float, default=2.0)
    parser.add_argument("--session-duration", type=int, default=3600)

    return parser.parse_args()


def start_process(command, capture_output=False):
    return subprocess.Popen(
        command,
        cwd=LAB_DIR,
        stdout=subprocess.PIPE if capture_output else subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=process_env(),
        start_new_session=True,
    )


def main():
    args = parse_args()

    model_path = Path(args.model).resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    bundle = joblib.load(model_path)

    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError("The model file does not contain the expected model bundle.")

    model = bundle["model"]
    params = bundle["feature_params"]

    feature_set = bundle["feature_set"]
    resample_points = int(params["resample_points"])
    feature_fs = float(params["feature_fs"])
    close_threshold_cm = float(params["close_threshold_cm"])
    max_distance_cm = float(params["max_distance_cm"])
    mad_z = float(params["mad_z"])

    parser = RangeLogParser()
    samples = []
    samples_lock = threading.Lock()

    controlee_command = twr_command(
        "python",
        args.controlee_port,
        args.preamble_code,
        args.session_duration + 10,
        args.slot_span,
        args.slots_per_rr,
        args.ranging_span,
        channel=args.channel,
        controlee=True,
        stats=True,
    )

    controller_command = twr_command(
        "python",
        args.controller_port,
        args.preamble_code,
        args.session_duration,
        args.slot_span,
        args.slots_per_rr,
        args.ranging_span,
        channel=args.channel,
        controlee=False,
        stats=True,
    )

    controlee_process = None
    controller_process = None

    try:
        print("Starting controlee...")
        controlee_process = start_process(controlee_command)

        time.sleep(args.startup_delay)

        print("Starting controller...")
        controller_process = start_process(
            controller_command,
            capture_output=True,
        )

        def read_controller():
            assert controller_process.stdout is not None

            for line in controller_process.stdout:
                sample = parser.feed(line)

                if sample:
                    sample = dict(sample)
                    sample["captured_monotonic"] = time.monotonic()

                    with samples_lock:
                        samples.append(sample)

        reader_thread = threading.Thread(
            target=read_controller,
            name="uwb-controller-reader",
            daemon=True,
        )
        reader_thread.start()

        print()
        print("Live UWB gesture predictor is ready.")
        print("Press Enter to record a gesture.")
        print("Type q and press Enter to quit.")

        while True:
            command = input("\nPress Enter to test a gesture: ").strip().lower()

            if command in {"q", "quit", "exit"}:
                break

            for remaining in range(args.countdown, 0, -1):
                print(f"Starting in {remaining}...")
                time.sleep(1)

            start_time = time.monotonic()
            end_time = start_time + args.window_seconds

            print("\nSTART GESTURE")

            while time.monotonic() < end_time:
                time.sleep(0.01)

            print("FINISH GESTURE")

            with samples_lock:
                captured = [
                    sample
                    for sample in samples
                    if start_time
                    <= sample.get("captured_monotonic", 0.0)
                    <= end_time
                ]

            distances = valid_distances(
                captured,
                max_distance_cm=max_distance_cm,
                mad_z=mad_z,
            )

            print(
                f"Captured {len(distances)} valid samples "
                f"from {len(captured)} total samples."
            )

            if len(distances) < 20:
                print("Not enough valid samples. Please try again.")
                continue

            features = extract_features(
                values=distances,
                points=resample_points,
                feature_set=feature_set,
                fs=feature_fs,
                close_threshold_cm=close_threshold_cm,
            )

            feature_array = np.asarray(features, dtype=float).reshape(1, -1)
            prediction = str(model.predict(feature_array)[0])

            print()
            print("=" * 42)
            print(f"PREDICTED GESTURE: {prediction}")
            
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(feature_array)[0]
                classes = model.classes_

                ranked = sorted(
                    zip(classes, probabilities),
                    key=lambda item: item[1],
                    reverse=True,
                )

                print(f"CONFIDENCE: {ranked[0][1] * 100:.1f}%")
                print()
                print("All probabilities:")

                for label, probability in ranked:
                    print(f"  {label:18s} {probability * 100:6.1f}%")

            print("=" * 42)

    except KeyboardInterrupt:
        print("\nStopping live prediction...")

    finally:
        if controller_process is not None:
            stop_process(controller_process)

        if controlee_process is not None:
            stop_process(controlee_process)

        print("UWB session stopped.")


if __name__ == "__main__":
    raise SystemExit(main())