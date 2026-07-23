"""
test_rfid.py
----------------------------------------------------
Standalone debug script -- connects to the RFID reader and prints
EVERY tag it sees (EPC + RSSI), regardless of tracked_tag_ids. Use
this FIRST, before running collect.py, to:
  1. Confirm the reader connects and responds at all.
  2. Discover your tags' actual EPC strings (you'll need these for
     collect.py's --rfid-tags argument).
  3. Sanity-check that RSSI values look reasonable (roughly -20 to -60
     dBm range is typical; wildly different numbers suggest the byte
     offsets in rfid_reader.py's _parse_tag_response() need adjusting).

Usage:
    python src/test_rfid.py --port /dev/tty.usbmodemXXXX
"""

import argparse
import time

from sensors.rfid_reader import RfidReader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--duration", type=float, default=15.0)
    args = parser.parse_args()

    print(f"Connecting to {args.port}...")
    reader = RfidReader(args.port, tracked_tag_ids=[])  # empty = we'll patch below

    reader.connect()
    print("Connected. Listening for tags for", args.duration, "seconds...")
    print("Move your tags in front of the reader now.\n")

    seen_tags = {}  # epc -> list of rssi values
    start = time.time()
    raw_byte_count = 0
    unparsed_sample_dump = None

    # Bypass the tracked-tag filtering in read_sample() -- for this
    # debug script we want to see EVERY tag, not just pre-registered
    # ones, so we call the internal parse loop directly instead.
    while time.time() - start < args.duration:
        if reader._ser.in_waiting:
            raw = reader._ser.read(reader._ser.in_waiting)
            raw_byte_count += len(raw)
            frames = raw.split(bytes([0xFF]))
            for frame in frames:
                if not frame:
                    continue
                parsed = reader._parse_tag_response(bytes([0xFF]) + frame)
                if parsed:
                    epc = parsed["epc"]
                    rssi = parsed["rssi"]
                    if epc not in seen_tags:
                        print(f"  NEW TAG: EPC={epc}  RSSI={rssi}")
                    seen_tags.setdefault(epc, []).append(rssi)
                elif unparsed_sample_dump is None and len(frame) > 3:
                    # Save the first unparsed-but-nonzero frame so we can
                    # show it below -- this tells us whether the reader
                    # IS sending data that our parser just isn't matching.
                    unparsed_sample_dump = (bytes([0xFF]) + frame).hex()
        time.sleep(0.01)

    reader.close()

    print("\n--- Summary ---")
    print(f"Total raw bytes received from reader: {raw_byte_count}")
    if unparsed_sample_dump:
        print(f"Sample UNPARSED frame (hex): {unparsed_sample_dump}")
        print("^ The reader IS sending data, but the parser didn't recognize "
              "it as a tag response -- send me this hex dump, this tells us "
              "exactly what byte offsets to fix.")
    if not seen_tags:
        print("No tags detected. Check: USB/SER switch on USB, tags close "
              "enough to the antenna (a few inches to ~1ft typically), and "
              "that the reader responded to the initial version check "
              "(look for connection errors above).")
    else:
        for epc, rssi_list in seen_tags.items():
            avg_rssi = sum(rssi_list) / len(rssi_list)
            print(f"EPC={epc}  reads={len(rssi_list)}  avg_rssi={avg_rssi:.1f}")
        print(f"\nCopy these EPC strings for collect.py's --rfid-tags argument, "
              f"thumb tag first.")


if __name__ == "__main__":
    main()
