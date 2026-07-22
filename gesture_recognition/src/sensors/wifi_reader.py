"""
wifi_reader.py
----------------------------------------------------
Reads WiFi Channel State Information (CSI) amplitude lines from the
sniffer board (ESP32-CAM or ESP32-C3), per your earlier WiFi CSI lab
setup (softAP + injector + sniffer). Expects the sniffer to stream
lines like:

    timestamp_ms,amp_0,amp_1,amp_2,...,amp_N

(one CSI amplitude value per subcarrier), matching the format used by
`tools/plot_csi_serial.py` and `tools/csi_presence_detect.py` from
that lab. Reuse the sniffer firmware from that lab as-is -- no changes
needed there.

NOTE: WiFi CSI sensing needs THREE boards (softAP + injector +
sniffer) to work, not just this reader's one serial connection. Make
sure all three are powered/configured per the earlier lab before
relying on this reader's output -- this class only reads the
sniffer's output stream.
"""

from .base_reader import BaseSensorReader


class WifiReader(BaseSensorReader):
    def __init__(self, port: str, baud: int = 115200):
        super().__init__(port, baud, name="wifi")

    def read_sample(self):
        line = self.readline_raw()
        if not line or line.lower().startswith("timestamp"):
            return None

        parts = line.split(",")
        if len(parts) < 3:
            return None  # need at least a timestamp + a couple subcarriers

        try:
            values = [float(p) for p in parts]
        except ValueError:
            return None

        return {
            "timestamp_ms": values[0],
            "csi_amplitudes": values[1:],  # variable length, subcarrier count
                                             # depends on your sniffer config
        }
