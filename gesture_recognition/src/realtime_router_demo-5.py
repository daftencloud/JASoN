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


def load_specialist(feature_set, model_name="random_forest"):
    path = os.path.join(MODELS_DIR, f"{feature_set}_{model_name}.pkl")
    if not os.path.exists(path):
        print(f"  (no {feature_set} model found at {path} -- skipping this specialist)")
        return None
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    print(f"  Loaded {feature_set} specialist ({model_name})")
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
    parser.add_argument("--vote-window", type=int, default=5)
    parser.add_argument("--confidence-threshold", type=float, default=0.4,
                         help="Minimum confidence for ANY specialist to report a "
                              "gesture at all -- below this, prints 'rest/uncertain' "
                              "instead of forcing a low-confidence guess.")
    args = parser.parse_args()

    print("Loading specialist models...")
    specialists = {
        "imu": load_specialist("imu"),
        "mmwave": load_specialist("mmwave"),
        "uwb": load_specialist("uwb"),
    }

    readers = build_readers(args)
    if not readers:
        print("No sensor ports provided.")
        return

    for reader in readers.values():
        reader.connect()

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

                # Whichever specialist is most confident wins this round.
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
