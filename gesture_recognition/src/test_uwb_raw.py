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

BAUD_RATES_TO_TRY = [115200, 9600, 19200, 38400, 460800, 921600]


def test_one_baud(port, baud):
    print(f"\n=== Testing baud {baud} ===")
    ser = serial.Serial(port, baud, timeout=0.1)
    time.sleep(1)
    ser.reset_input_buffer()

    print("--- Phase 1: passive listen (2 seconds) ---")
    start = time.time()
    total = b""
    while time.time() - start < 2:
        if ser.in_waiting:
            total += ser.read(ser.in_waiting)
        time.sleep(0.05)

    if total:
        print(f"RECEIVED {len(total)} bytes passively:")
        print(f"  Hex:  {total.hex()}")
        print(f"  Text: {total.decode(errors='replace')}")
        ser.close()
        return True

    print("Nothing passively. Trying Enter + help...")
    ser.write(b"\r\n")
    time.sleep(0.3)
    ser.write(b"help\r\n")
    time.sleep(1)

    response = b""
    start = time.time()
    while time.time() - start < 1.5:
        if ser.in_waiting:
            response += ser.read(ser.in_waiting)
        time.sleep(0.05)

    ser.close()

    if response:
        print(f"RECEIVED {len(response)} bytes after input:")
        print(f"  Hex:  {response.hex()}")
        print(f"  Text: {response.decode(errors='replace')}")
        return True

    print("Nothing at this baud rate.")
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    args = parser.parse_args()

    print(f"Testing {args.port} across {len(BAUD_RATES_TO_TRY)} baud rates...")

    for baud in BAUD_RATES_TO_TRY:
        try:
            got_data = test_one_baud(args.port, baud)
        except serial.SerialException as e:
            print(f"  Could not open port at {baud}: {e}")
            continue
        if got_data:
            print(f"\n*** SUCCESS at baud {baud} -- use this baud rate! ***")
            return

    print("\n--- Done ---")
    print("No response at ANY baud rate. This strongly suggests you're "
          "connected to the wrong USB port on the board -- DWM3001CDK "
          "boards typically have a separate debug/J-Link USB port distinct "
          "from the application/shell port. Check for a second USB "
          "connector on the board and try that one instead.")


if __name__ == "__main__":
    main()
