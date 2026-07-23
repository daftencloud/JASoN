"""
collect.py
----------------------------------------------------
Records one labeled gesture trial at a time, reading from however many
sensors are currently connected, and writes one raw CSV per sensor per
trial to data/raw/<gesture>/<person>_<trial>_<sensor>.csv

Every ACTIVE sensor is recorded for EVERY gesture trial -- don't
restrict which sensors run per gesture. gestures.py's
EXPECTED_STRONGEST_SENSOR table is a hypothesis to test in evaluate.py,
not a collection-time restriction; restricting collection would
fragment your dataset and prevent training one fused model across all
15 gestures.

Usage:
    python src/collect.py --person alex --gesture push --duration 2.5 \
        --imu-port /dev/tty.usbserial-AAA \
        --uwb-port /dev/tty.usbserial-BBB \
        --mmwave-port /dev/tty.usbserial-CCC \
        --rfid-port /dev/tty.usbserial-DDD \
        --rfid-tags THUMB_EPC,INDEX_EPC,MIDDLE_EPC,RING_EPC

Only pass the ports for sensors you actually have connected this
session -- any sensor without a --X-port flag is simply skipped for
this trial.
"""

import argparse
import csv
import os
import time

from gestures import GESTURES
from sensors import ImuReader, UwbReader, MmwaveReader, WifiReader, RfidReader

DATA_RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def build_readers(args):
    readers = {}
    if args.imu_port:
        readers["imu"] = ImuReader(args.imu_port)
    if args.uwb_port:
        readers["uwb"] = UwbReader(args.uwb_port)
    if args.mmwave_port:
        readers["mmwave"] = MmwaveReader(args.mmwave_port, cfg_path=args.mmwave_cfg)
    if args.wifi_port:
        readers["wifi"] = WifiReader(args.wifi_port)
    if args.rfid_port:
        tag_ids = args.rfid_tags.split(",") if args.rfid_tags else []
        readers["rfid"] = RfidReader(args.rfid_port, tracked_tag_ids=tag_ids)
    return readers


def next_trial_number(gesture_dir: str, person: str) -> int:
    if not os.path.isdir(gesture_dir):
        return 1
    existing = [f for f in os.listdir(gesture_dir) if f.startswith(person + "_")]
    trial_nums = set()
    for f in existing:
        parts = f.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            trial_nums.add(int(parts[1]))
    return (max(trial_nums) + 1) if trial_nums else 1


def record_trial(readers: dict, gesture: str, person: str, duration: float):
    if gesture not in GESTURES:
        print(f"WARNING: '{gesture}' is not in the canonical GESTURES list "
              f"in gestures.py -- continuing anyway, but double check spelling.")

    gesture_dir = os.path.join(DATA_RAW_DIR, gesture)
    os.makedirs(gesture_dir, exist_ok=True)
    trial_num = next_trial_number(gesture_dir, person)

    for reader in readers.values():
        reader.connect()

    input(f"Ready to record '{gesture}' trial {trial_num} for {person} "
          f"using sensors: {list(readers.keys())}. Press Enter, then perform "
          f"the gesture...")

    buffers = {name: [] for name in readers}
    deadline = time.time() + duration
    while time.time() < deadline:
        for name, reader in readers.items():
            sample = reader.read_sample()
            if sample is not None:
                buffers[name].append(sample)

    for reader in readers.values():
        reader.close()

    for name, samples in buffers.items():
        out_path = os.path.join(gesture_dir, f"{person}_{trial_num:03d}_{name}.csv")
        _write_samples_csv(samples, out_path)
        print(f"  {name}: {len(samples)} samples -> {out_path}")


def _write_samples_csv(samples: list, out_path: str):
    """
    Writes a list of sample dicts to CSV. Handles the two shapes that
    show up in this project: flat dicts (imu, uwb, wifi -- mostly) and
    nested dicts (mmwave's point list, rfid's per-tag dict) by
    serializing nested values as a single JSON-ish string column so
    nothing gets silently dropped; extract_features.py knows how to
    parse each sensor's specific shape back out.
    """
    import json

    if not samples:
        with open(out_path, "w", newline="") as f:
            f.write("")  # empty file -- extract_features.py should skip these
        return

    fieldnames = sorted({key for s in samples for key in s.keys()})
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in samples:
            row = {}
            for k in fieldnames:
                v = s.get(k)
                row[k] = json.dumps(v) if isinstance(v, (list, dict)) else v
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--person", required=True)
    parser.add_argument("--gesture", required=True, choices=GESTURES)
    parser.add_argument("--duration", type=float, default=2.5)
    parser.add_argument("--imu-port")
    parser.add_argument("--uwb-port")
    parser.add_argument("--mmwave-port")
    parser.add_argument("--mmwave-cfg", help="Path to a .cfg file from your "
                         "cloned mmwave_lab repo, e.g. "
                         "~/COSMOS-Cluster10/mmwave_lab/xwrL64xx-evm/point_cloud.cfg")
    parser.add_argument("--wifi-port")
    parser.add_argument("--rfid-port")
    parser.add_argument("--rfid-tags", help="comma-separated EPC hex strings, "
                                              "thumb first")
    args = parser.parse_args()

    readers = build_readers(args)
    if not readers:
        print("No sensor ports provided -- pass at least one of --imu-port, "
              "--uwb-port, --mmwave-port, --wifi-port, --rfid-port.")
        return

    record_trial(readers, args.gesture, args.person, args.duration)


if __name__ == "__main__":
    main()
