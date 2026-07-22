"""
collect_data.py
----------------------------------------------------
Records one labeled gesture trial at a time from the IMU (Core2) and
UWB tag serial ports, saving each trial as its own CSV under
data/raw/<gesture>/<person>_<trial_number>.csv

Usage:
    python collect_data.py --imu-port /dev/tty.usbserial-XXXX \
                            --uwb-port /dev/tty.usbserial-YYYY \
                            --person alex --gesture push --duration 2.5

Run once per trial. Repeat for every gesture x every group member,
aiming for a balanced number of trials per gesture (e.g. 20-30 each
to start).
"""

import argparse
import csv
import os
import time
from collections import deque

import serial


def read_serial_lines(ser, deadline, buffer):
    """Read whatever lines are available before `deadline` (time.time())."""
    while time.time() < deadline:
        if ser.in_waiting:
            line = ser.readline().decode(errors="ignore").strip()
            if line and not line.startswith("timestamp"):  # skip header
                buffer.append(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imu-port", required=True)
    parser.add_argument("--uwb-port", required=True)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--person", required=True, help="group member name/id")
    parser.add_argument("--gesture", required=True, help="gesture class label")
    parser.add_argument("--duration", type=float, default=2.5,
                         help="seconds to record per trial")
    parser.add_argument("--out-dir", default="data/raw")
    args = parser.parse_args()

    gesture_dir = os.path.join(args.out_dir, args.gesture)
    os.makedirs(gesture_dir, exist_ok=True)

    # figure out the next trial number for this person+gesture so we
    # don't overwrite previous trials
    existing = [f for f in os.listdir(gesture_dir) if f.startswith(args.person)]
    trial_num = len(existing) + 1
    out_path = os.path.join(gesture_dir, f"{args.person}_{trial_num:03d}.csv")

    imu_ser = serial.Serial(args.imu_port, args.baud, timeout=0.05)
    uwb_ser = serial.Serial(args.uwb_port, args.baud, timeout=0.05)
    time.sleep(2)  # let the serial connections settle
    imu_ser.reset_input_buffer()
    uwb_ser.reset_input_buffer()

    input(f"Ready to record '{args.gesture}' trial {trial_num} for "
          f"{args.person}. Press Enter, then perform the gesture...")

    imu_lines = deque()
    uwb_lines = deque()

    start = time.time()
    deadline = start + args.duration
    while time.time() < deadline:
        read_serial_lines(imu_ser, deadline, imu_lines)
        read_serial_lines(uwb_ser, deadline, uwb_lines)

    imu_ser.close()
    uwb_ser.close()

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "raw_line"])
        for line in imu_lines:
            writer.writerow(["imu", line])
        for line in uwb_lines:
            writer.writerow(["uwb", line])

    print(f"Saved {len(imu_lines)} IMU samples and {len(uwb_lines)} "
          f"UWB samples to {out_path}")


if __name__ == "__main__":
    main()
