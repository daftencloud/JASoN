"""
test_uwb_raw.py
----------------------------------------------------
Minimal raw connectivity test for a DWM3001C UWB module. Listens
passively first (in case it streams data on its own), then sends a
few common things (Enter keypresses, "help") to see if it's running an
interactive shell.

Usage:
    python src/test_uwb_raw.py --port /dev/tty.usbmodem0007602254181
"""

import argparse
import time

import serial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--baud", type=int, default=115200)
    args = parser.parse_args()

    print(f"Opening {args.port} at {args.baud} baud...")
    ser = serial.Serial(args.port, args.baud, timeout=0.1)
    time.sleep(1)
    ser.reset_input_buffer()

    print("\n--- Phase 1: passive listen (5 seconds, no input sent) ---")
    start = time.time()
    total = b""
    while time.time() - start < 5:
        if ser.in_waiting:
            total += ser.read(ser.in_waiting)
        time.sleep(0.05)

    if total:
        print(f"RECEIVED {len(total)} bytes passively:")
        print(f"  Hex:  {total.hex()}")
        print(f"  Text: {total.decode(errors='replace')}")
    else:
        print("Nothing received passively.")

    print("\n--- Phase 2: sending Enter keypresses ---")
    for _ in range(3):
        ser.write(b"\r\n")
        time.sleep(0.3)

    time.sleep(1)
    response = b""
    start = time.time()
    while time.time() - start < 2:
        if ser.in_waiting:
            response += ser.read(ser.in_waiting)
        time.sleep(0.05)

    if response:
        print(f"RECEIVED {len(response)} bytes after Enter:")
        print(f"  Hex:  {response.hex()}")
        print(f"  Text: {response.decode(errors='replace')}")
    else:
        print("Nothing received after Enter either.")

    print("\n--- Phase 3: sending 'help' command ---")
    ser.write(b"help\r\n")
    time.sleep(1)
    response = b""
    start = time.time()
    while time.time() - start < 2:
        if ser.in_waiting:
            response += ser.read(ser.in_waiting)
        time.sleep(0.05)

    if response:
        print(f"RECEIVED {len(response)} bytes after 'help':")
        print(f"  Hex:  {response.hex()}")
        print(f"  Text: {response.decode(errors='replace')}")
    else:
        print("Nothing received after 'help' either.")

    ser.close()
    print("\n--- Done ---")


if __name__ == "__main__":
    main()
