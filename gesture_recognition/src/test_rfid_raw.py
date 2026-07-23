"""
test_rfid_raw.py
----------------------------------------------------
Minimal raw connectivity test -- bypasses rfid_reader.py's higher-level
logic. Two modes:

  1. Passive listen (default): just opens the port at a few common baud
     rates and listens for any unsolicited bytes (some modules print a
     boot banner).
  2. Active command test (--send-command): sends the real GET VERSION
     command our rfid_reader.py uses, and checks for a response. This
     is the more definitive test, since many RFID readers stay silent
     until queried -- a quiet passive listen doesn't necessarily mean
     the module is broken.

Usage:
    python src/test_rfid_raw.py --port /dev/tty.wchusbserial110
    python src/test_rfid_raw.py --port /dev/tty.wchusbserial110 --send-command
"""

import argparse
import sys
import time

import serial

BAUD_RATES_TO_TRY = [38400, 115200, 9600, 19200]


def listen_passively(port, baud, listen_seconds=3):
    print(f"\n--- Passive listen at baud {baud} ---")
    try:
        ser = serial.Serial(port, baud, timeout=0.1)
    except serial.SerialException as e:
        print(f"  Could not open port: {e}")
        return

    time.sleep(1)
    ser.reset_input_buffer()

    start = time.time()
    total_bytes = b""
    while time.time() - start < listen_seconds:
        if ser.in_waiting:
            total_bytes += ser.read(ser.in_waiting)
        time.sleep(0.05)

    ser.close()

    if total_bytes:
        print(f"  RECEIVED {len(total_bytes)} bytes (unsolicited)")
        print(f"  Hex: {total_bytes.hex()}")
    else:
        print("  No bytes received passively (this is normal for many "
              "readers -- try --send-command for a more definitive test).")


def send_command_test(port, baud, listen_seconds=2):
    sys.path.insert(0, ".")
    from sensors.rfid_reader import _build_command, OPCODE_VERSION

    print(f"\n--- Sending GET VERSION command at baud {baud} ---")
    try:
        ser = serial.Serial(port, baud, timeout=0.1)
    except serial.SerialException as e:
        print(f"  Could not open port: {e}")
        return

    time.sleep(1)
    ser.reset_input_buffer()

    cmd = _build_command(OPCODE_VERSION)
    print(f"  Sending bytes: {cmd.hex()}")
    ser.write(cmd)

    start = time.time()
    response = b""
    while time.time() - start < listen_seconds:
        if ser.in_waiting:
            response += ser.read(ser.in_waiting)
        time.sleep(0.05)

    ser.close()

    if response:
        print(f"  GOT RESPONSE: {response.hex()}")
    else:
        print("  No response to this command either.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--send-command", action="store_true",
                         help="Send the real GET VERSION command instead of "
                              "just listening passively.")
    args = parser.parse_args()

    print(f"Testing {args.port} at several baud rates.\n")

    for baud in BAUD_RATES_TO_TRY:
        if args.send_command:
            send_command_test(args.port, baud)
        else:
            listen_passively(args.port, baud)

    print("\n--- Done ---")
    if not args.send_command:
        print("Nothing received passively across all baud rates is common "
              "and not necessarily a problem -- re-run with --send-command "
              "for a definitive test.")
    else:
        print("If NONE of the baud rates got a response, check: the USB/SER "
              "switch is on USB, the board has power (any LED lit?), and "
              "try a different USB cable.")
        print("If ONE baud rate got a response, tell me the hex and which "
              "baud -- that confirms the real settings to use.")


if __name__ == "__main__":
    main()
