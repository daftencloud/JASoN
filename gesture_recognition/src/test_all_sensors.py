"""
test_all_sensors.py
----------------------------------------------------
Quick status check across whichever sensors you have connected right
now -- tries to connect and read a few samples from each, and reports
pass/fail per sensor. Use this as your go-to "is everything working"
script once individual sensors are debugged.

Usage:
    python src/test_all_sensors.py \
        --imu-port /dev/tty.usbserial-AAA \
        --uwb-port /dev/tty.usbmodemBBB \
        --mmwave-port /dev/tty.usbmodemCCC \
        --rfid-port /dev/tty.wchusbserial110

Only pass ports for sensors you have connected right now.
"""

import argparse
import time

from sensors import ImuReader, UwbReader, MmwaveReader, WifiReader, RfidReader


def test_reader(name, reader, duration=3.0):
    print(f"\n=== {name} ===")
    try:
        reader.connect()
    except Exception as e:
        print(f"  FAILED to connect: {e}")
        return False

    samples = []
    start = time.time()
    while time.time() - start < duration:
        try:
            sample = reader.read_sample()
        except Exception as e:
            print(f"  ERROR during read_sample(): {e}")
            reader.close()
            return False
        if sample is not None:
            samples.append(sample)

    reader.close()

    if samples:
        print(f"  PASS -- received {len(samples)} samples in {duration}s")
        print(f"  Example sample: {samples[0]}")
        return True
    else:
        print(f"  NO DATA -- connected fine, but received 0 samples in {duration}s")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imu-port")
    parser.add_argument("--uwb-port")
    parser.add_argument("--mmwave-port")
    parser.add_argument("--wifi-port")
    parser.add_argument("--rfid-port")
    parser.add_argument("--rfid-tags", help="comma-separated EPCs to track, for the RFID test")
    args = parser.parse_args()

    results = {}

    if args.imu_port:
        results["imu"] = test_reader("IMU", ImuReader(args.imu_port))
    if args.uwb_port:
        results["uwb"] = test_reader("UWB", UwbReader(args.uwb_port))
    if args.mmwave_port:
        results["mmwave"] = test_reader("mmWave", MmwaveReader(args.mmwave_port))
    if args.wifi_port:
        results["wifi"] = test_reader("WiFi CSI", WifiReader(args.wifi_port))
    if args.rfid_port:
        tags = args.rfid_tags.split(",") if args.rfid_tags else []
        results["rfid"] = test_reader("RFID", RfidReader(args.rfid_port, tracked_tag_ids=tags))

    if not results:
        print("No ports provided -- pass at least one of --imu-port, --uwb-port, "
              "--mmwave-port, --wifi-port, --rfid-port.")
        return

    print("\n=== SUMMARY ===")
    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
