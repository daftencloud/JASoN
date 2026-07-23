"""
test_mmwave_raw.py
----------------------------------------------------
Raw connectivity test for the TI IWRL6432FSPEVM's DATA UART port.
Listens passively across a few common baud rates and reports whether
it sees TI's standard magic word (0201040306050807), which confirms
both the port AND the baud rate are correct.

Usage:
    python src/test_mmwave_raw.py --port /dev/tty.usbmodemXXXX
"""

import argparse
import time

import serial

MAGIC_WORD_HEX = "0201040306050807"
BAUD_RATES_TO_TRY = [921600, 115200, 460800, 230400]


def test_baud(port, baud, listen_seconds=3):
    print(f"\n=== Testing baud {baud} ===")
    try:
        ser = serial.Serial(port, baud, timeout=0.1)
    except serial.SerialException as e:
        print(f"  Could not open port: {e}")
        return False

    time.sleep(1)
    ser.reset_input_buffer()

    start = time.time()
    total = b""
    while time.time() - start < listen_seconds:
        if ser.in_waiting:
            total += ser.read(ser.in_waiting)
        time.sleep(0.05)
    ser.close()

    if not total:
        print("  No bytes received.")
        return False

    print(f"  Received {len(total)} bytes.")
    hex_str = total.hex()
    if MAGIC_WORD_HEX in hex_str:
        idx = hex_str.index(MAGIC_WORD_HEX)
        print(f"  *** FOUND TI MAGIC WORD at byte offset {idx // 2} -- "
              f"this baud rate is correct! ***")
        print(f"  Sample: {hex_str[max(0, idx-20):idx+100]}")
        return True
    else:
        print(f"  Bytes received, but no TI magic word found -- likely "
              f"wrong baud (garbled data). First 40 hex chars: {hex_str[:40]}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    args = parser.parse_args()

    print(f"Testing {args.port} across {len(BAUD_RATES_TO_TRY)} baud rates.")
    print("Make sure the radar is powered and running its demo firmware, "
          "and that this port is the DATA port, not the CLI/config port.\n")

    for baud in BAUD_RATES_TO_TRY:
        if test_baud(args.port, baud):
            print(f"\nUse --mmwave-port {args.port} with this baud rate "
                  f"({baud}) confirmed working.")
            return

    print("\n--- Done ---")
    print("No baud rate showed the magic word. Check: this is the DATA port "
          "(not CLI port -- IWRL6432 typically exposes two), the radar demo "
          "firmware is actually running (not just powered on), and the "
          "board/cable connection.")


if __name__ == "__main__":
    main()
