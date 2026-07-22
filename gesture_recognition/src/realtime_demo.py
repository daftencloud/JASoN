"""
realtime_demo.py
----------------------------------------------------
Live gesture classification demo: connects to whichever sensors you
specify, buffers a rolling window of samples, extracts the same
features extract_features.py would compute, and runs them through a
trained model from models/ to print live gesture predictions.

Usage:
    python src/realtime_demo.py --model models/fused_random_forest.pkl \
        --imu-port /dev/tty.usbserial-AAA \
        --uwb-port /dev/tty.usbserial-BBB

Only pass ports for sensors that were part of the feature set the
chosen model was trained on -- check the model's feature_set name
(e.g. "fused", "imu", "mmwave") to know which sensors it expects.
"""

import argparse
import pickle
import time

import pandas as pd

from sensors import ImuReader, UwbReader, MmwaveReader, WifiReader, RfidReader
from extract_features import EXTRACTORS

WINDOW_SECONDS = 1.5
PREDICT_INTERVAL_SECONDS = 0.3


def build_readers(args):
    readers = {}
    if args.imu_port:
        readers["imu"] = ImuReader(args.imu_port)
    if args.uwb_port:
        readers["uwb"] = UwbReader(args.uwb_port)
    if args.mmwave_port:
        readers["mmwave"] = MmwaveReader(args.mmwave_port)
    if args.wifi_port:
        readers["wifi"] = WifiReader(args.wifi_port)
    if args.rfid_port:
        tag_ids = args.rfid_tags.split(",") if args.rfid_tags else []
        readers["rfid"] = RfidReader(args.rfid_port, tracked_tag_ids=tag_ids)
    return readers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="path to a .pkl file from models/")
    parser.add_argument("--imu-port")
    parser.add_argument("--uwb-port")
    parser.add_argument("--mmwave-port")
    parser.add_argument("--wifi-port")
    parser.add_argument("--rfid-port")
    parser.add_argument("--rfid-tags")
    args = parser.parse_args()

    with open(args.model, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    feature_set = bundle["feature_set"]

    print(f"Loaded model trained on feature set: '{feature_set}'")
    print(f"Expects these feature columns: {feature_columns}")

    readers = build_readers(args)
    if not readers:
        print("No sensor ports provided.")
        return

    for reader in readers.values():
        reader.connect()

    print("Connected. Streaming live predictions (Ctrl+C to stop)...")

    buffers = {name: [] for name in readers}
    last_predict_time = time.time()

    try:
        while True:
            now = time.time()
            for name, reader in readers.items():
                sample = reader.read_sample()
                if sample is not None:
                    buffers[name].append((now, sample))

            # Drop samples older than the window
            for name in buffers:
                buffers[name] = [(t, s) for t, s in buffers[name]
                                  if now - t <= WINDOW_SECONDS]

            if now - last_predict_time >= PREDICT_INTERVAL_SECONDS:
                last_predict_time = now
                feats = {}
                for name, samples in buffers.items():
                    if not samples or name not in EXTRACTORS:
                        continue
                    df = pd.DataFrame([s for _, s in samples])
                    feats.update(EXTRACTORS[name](df))

                if feats:
                    row = pd.DataFrame([feats])
                    # Ensure every expected column is present, fill
                    # missing ones with 0 (e.g. a sensor briefly quiet).
                    for col in feature_columns:
                        if col not in row.columns:
                            row[col] = 0
                    row = row[feature_columns]

                    prediction = model.predict(row)[0]
                    print(f"[{time.strftime('%H:%M:%S')}] Predicted gesture: {prediction}")

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        for reader in readers.values():
            reader.close()


if __name__ == "__main__":
    main()
