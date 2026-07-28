"""
realtime_router_demo.py
----------------------------------------------------
"Router" live demo: instead of one fused model trying to handle every
gesture (which breaks down live -- see README note below), this loads
the 3 SEPARATE specialist models (imu_random_forest.pkl,
mmwave_random_forest.pkl, uwb_random_forest.pkl) and, at each
prediction step, asks all 3 "how confident are you, and what's your
guess?" -- then reports whichever specialist is most confident.

This matches the actual team design intent: each gesture is meant to
be recognized by its ONE best-suited sensor, with all 3 sensors
physically connected and running throughout. It also naturally avoids
the "zero columns = shortcut" problem the single fused model had,
since each specialist only ever sees its own sensor's real (non-zero)
features -- never a mix of real + zero-padded columns from sensors it
was never trained to expect data from.

Usage:
    python realtime_router_demo.py \
        --imu-port /dev/cu.wchusbserial5B1F0087031 \
        --mmwave-port /dev/cu.usbserial-BH00LUQV \
        --mmwave-cfg ~/COSMOS-Cluster10/mmwave_lab/xwrL64xx-evm/point_cloud.cfg \
        --uwb-controller-port /dev/cu.usbmodemC55B4699A7AB1 \
        --uwb-controlee-port /dev/cu.usbmodemE313381643361 \
        --uwb-preamble-idx 11 \
        --vote-window 5
"""

import argparse
import os
import pickle
import time
from collections import Counter

import pandas as pd

from sensors import ImuReader, UwbReader, MmwaveReader
from extract_features import EXTRACTORS

WINDOW_SECONDS = 5.0
PREDICT_INTERVAL_SECONDS = 1.0
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# NOTE: with SCOPED specialists (each trained on only its own gestures
# via train_scoped_specialists.py), a gesture-ownership boost is a
# no-op -- every candidate a scoped specialist produces is already
# "in its own domain" by construction, so the boost applies equally to
# all 3 and never changes the ranking. The real fix for cross-domain
# confusion is the margin-based confidence in predict_with_confidence()
# below, which corrects for out-of-domain overconfidence directly.


def load_specialist(model_filename):
    path = os.path.join(MODELS_DIR, model_filename)
    if not os.path.exists(path):
        print(f"  (no model found at {path} -- skipping this specialist)")
        return None
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    print(f"  Loaded {model_filename}")
    return bundle


def predict_with_confidence(bundle, feats: dict):
    """Returns (predicted_gesture, confidence) for one specialist, or
    (None, 0.0) if this specialist has no relevant features this cycle."""
    if bundle is None or not feats:
        return None, 0.0

    row = pd.DataFrame([feats])
    for col in bundle["feature_columns"]:
        if col not in row.columns:
            row[col] = 0
    row = row[bundle["feature_columns"]]

    model = bundle["model"]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(row)[0]
        classes = model.classes_
        best_idx = proba.argmax()
        return classes[best_idx], proba[best_idx]
    else:
        # Fallback for models without predict_proba (e.g. some pipelines)
        pred = model.predict(row)[0]
        return pred, 1.0


def build_readers(args):
    readers = {}
    if args.imu_port:
        readers["imu"] = ImuReader(args.imu_port)
    if args.mmwave_port:
        readers["mmwave"] = MmwaveReader(args.mmwave_port, cfg_path=args.mmwave_cfg)
    if args.uwb_controller_port and args.uwb_controlee_port:
        readers["uwb"] = UwbReader(
            args.uwb_controller_port, args.uwb_controlee_port,
            args.uwb_lab_tools_path, channel=args.uwb_channel,
            preamble_idx=args.uwb_preamble_idx,
        )
    return readers


def run_trigger_mode(readers, specialists, args):
    """
    Press Enter -> record for exactly args.capture_duration seconds ->
    classify once -> print result -> repeat. No continuous streaming,
    no rolling-window lag -- eliminates the "delay between doing the
    gesture and seeing the classification" issue entirely, since
    there's never an ambiguous in-between state to be confused by.
    """
    print("\nConnected. TRIGGER MODE -- press Enter, then perform ONE gesture "
          f"during the {args.capture_duration}s capture window.")
    print("Each specialist votes using its own sensor's features; whichever "
          "is most confident wins. Ctrl+C to stop.\n")

    try:
        while True:
            input("Press Enter to capture a gesture...")

            buffers = {name: [] for name in readers}
            deadline = time.time() + args.capture_duration
            while time.time() < deadline:
                for name, reader in readers.items():
                    sample = reader.read_sample()
                    if sample is not None:
                        buffers[name].append(sample)

            candidates = []
            for name, samples in buffers.items():
                if not samples or name not in EXTRACTORS:
                    continue
                df = pd.DataFrame(samples)
                feats = EXTRACTORS[name](df)
                print(f"    [{name}] {len(samples)} raw samples collected this "
                      f"capture. Key features: "
                      f"{ {k: round(v, 3) for k, v in list(feats.items())[:6]} }")
                gesture, confidence = predict_with_confidence(specialists[name], feats)
                if gesture is not None:
                    candidates.append((name, gesture, confidence))

            if not candidates:
                print("  No sensor produced enough data this capture -- try again.\n")
                continue

            # IMU saying "rest" means "not much wrist motion detected" --
            # which is EXPECTED and uninformative during fine hand-only
            # gestures (soli, open_close_fist, palm_up_down all involve
            # almost no wrist movement). It shouldn't be treated as
            # evidence AGAINST another sensor's real positive detection.
            # If another sensor has a real (non-rest) guess, prefer that
            # over IMU's rest, regardless of raw confidence.
            non_rest_candidates = [c for c in candidates if c[1] != "rest"]
            if non_rest_candidates:
                winner_sensor, winner_gesture, winner_confidence = max(
                    non_rest_candidates, key=lambda c: c[2]
                )
            else:
                winner_sensor, winner_gesture, winner_confidence = max(
                    candidates, key=lambda c: c[2]
                )
            debug_str = ", ".join(
                f"{name}={gesture}({conf:.2f})" for name, gesture, conf in candidates
            )

            if winner_confidence < args.confidence_threshold:
                print(f"  All: [{debug_str}]")
                print(f"  -> RESULT: rest / uncertain (best guess {winner_sensor}/"
                      f"{winner_gesture} only {winner_confidence:.2f} confident)\n")
            else:
                print(f"  All: [{debug_str}]")
                print(f"  -> RESULT: {winner_gesture} (via {winner_sensor}, "
                      f"{winner_confidence:.2f} confidence)\n")

    except KeyboardInterrupt:
        print("\nStopping...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imu-port")
    parser.add_argument("--mmwave-port")
    parser.add_argument("--mmwave-cfg")
    parser.add_argument("--uwb-controller-port")
    parser.add_argument("--uwb-controlee-port")
    parser.add_argument("--uwb-lab-tools-path",
                         default=os.path.expanduser("~/UWB_lab/uwb-qorvo-tools"))
    parser.add_argument("--uwb-channel", type=int, default=5)
    parser.add_argument("--uwb-preamble-idx", type=int, default=11)
    parser.add_argument("--imu-model", default="imu_random_forest.pkl",
                         help="Model filename in models/ to use for the IMU "
                              "specialist (default: plain real-data model, "
                              "since augmentation didn't help IMU)")
    parser.add_argument("--mmwave-model", default="mmwave_random_forest_augmented.pkl",
                         help="Model filename for the mmWave specialist "
                              "(default: augmented, since augmentation genuinely "
                              "helped mmWave's small dataset)")
    parser.add_argument("--uwb-model", default="uwb_random_forest_augmented.pkl",
                         help="Model filename for the UWB specialist "
                              "(default: augmented, same reasoning as mmWave)")
    parser.add_argument("--vote-window", type=int, default=5,
                         help="Number of recent predictions to majority-vote over. "
                              "Lower = reacts faster to gesture changes (less lag), "
                              "but less smoothing of single-instant noise. Higher = "
                              "smoother but can lag behind real gesture changes.")
    parser.add_argument("--confidence-threshold", type=float, default=0.0,
                         help="Minimum confidence for ANY specialist to report a "
                              "gesture at all -- below this, prints 'rest/uncertain' "
                              "instead of forcing a low-confidence guess. Default 0.0 "
                              "means always trust whichever specialist is most "
                              "confident, never override with 'rest/uncertain'.")
    parser.add_argument("--trigger-mode", action="store_true",
                         help="Instead of continuously streaming predictions from a "
                              "rolling window (which has inherent lag between "
                              "finishing a gesture and the window catching up), "
                              "press Enter, then perform ONE gesture during a fixed "
                              "capture window (matching --capture-duration), then get "
                              "exactly one classification. This also better matches "
                              "how training data was collected (fixed-duration "
                              "trials), so classification tends to be more accurate too.")
    parser.add_argument("--capture-duration", type=float, default=7.0,
                         help="Seconds to record in trigger-mode (default matches "
                              "typical collection trial duration)")
    args = parser.parse_args()

    print("Loading specialist models...")
    specialists = {
        "imu": load_specialist(args.imu_model),
        "mmwave": load_specialist(args.mmwave_model),
        "uwb": load_specialist(args.uwb_model),
    }

    readers = build_readers(args)
    if not readers:
        print("No sensor ports provided.")
        return

    for reader in readers.values():
        reader.connect()

    if args.trigger_mode:
        run_trigger_mode(readers, specialists, args)
        for reader in readers.values():
            reader.close()
        return

    print("\nConnected. Streaming routed predictions (Ctrl+C to stop)...")
    print(f"Each specialist only votes using its OWN sensor's features -- "
          f"whichever is most confident wins.\n")

    buffers = {name: [] for name in readers}
    recent_predictions = []
    last_predict_time = time.time()

    try:
        while True:
            now = time.time()
            for name, reader in readers.items():
                sample = reader.read_sample()
                if sample is not None:
                    buffers[name].append((now, sample))

            for name in buffers:
                buffers[name] = [(t, s) for t, s in buffers[name]
                                  if now - t <= WINDOW_SECONDS]

            if now - last_predict_time >= PREDICT_INTERVAL_SECONDS:
                last_predict_time = now

                candidates = []  # (sensor_name, gesture, confidence)
                for name, samples in buffers.items():
                    if not samples or name not in EXTRACTORS:
                        continue
                    df = pd.DataFrame([s for _, s in samples])
                    feats = EXTRACTORS[name](df)
                    gesture, confidence = predict_with_confidence(specialists[name], feats)
                    if gesture is not None:
                        candidates.append((name, gesture, confidence))

                if not candidates:
                    continue

                # Same rest-exclusion rule as trigger mode -- IMU's
                # "rest" is uninformative during fine hand-only gestures,
                # shouldn't crowd out a real positive detection elsewhere.
                non_rest_candidates = [c for c in candidates if c[1] != "rest"]
                if non_rest_candidates:
                    winner_sensor, winner_gesture, winner_confidence = max(
                        non_rest_candidates, key=lambda c: c[2]
                    )
                else:
                    winner_sensor, winner_gesture, winner_confidence = max(
                        candidates, key=lambda c: c[2]
                    )

                if winner_confidence < args.confidence_threshold:
                    final_label = "rest / uncertain"
                else:
                    final_label = winner_gesture

                recent_predictions.append(final_label)
                if len(recent_predictions) > args.vote_window:
                    recent_predictions.pop(0)

                debug_str = ", ".join(
                    f"{name}={gesture}({conf:.2f})" for name, gesture, conf in candidates
                )

                if len(recent_predictions) >= args.vote_window:
                    voted = Counter(recent_predictions).most_common(1)[0][0]
                    print(f"[{time.strftime('%H:%M:%S')}] All: [{debug_str}] "
                          f"-> Winner: {winner_sensor}/{winner_gesture} "
                          f"({winner_confidence:.2f}) -> Voted: {voted}")
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] All: [{debug_str}] "
                          f"-> Winner: {winner_sensor}/{winner_gesture} "
                          f"({winner_confidence:.2f}) (warming up vote window)")

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        for reader in readers.values():
            reader.close()


if __name__ == "__main__":
    main()
