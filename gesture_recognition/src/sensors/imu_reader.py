"""
imu_reader.py
----------------------------------------------------
Reads accelerometer + gyroscope samples from a Core2 (or any ESP32 +
6-axis IMU) streaming plain CSV lines over serial:

    timestamp_ms,ax,ay,az,gx,gy,gz

This assumes the board is running simple streaming firmware (not a
custom gesture classifier -- all gesture logic now lives in Python per
your TA's structure). If your board isn't already streaming this
format, flash a minimal loop that just prints
`millis(),ax,ay,az,gx,gy,gz` every sample -- no feature extraction or
classification needed on the board side anymore.
"""

from .base_reader import BaseSensorReader


class ImuReader(BaseSensorReader):
    def __init__(self, port: str, baud: int = 115200):
        super().__init__(port, baud, name="imu")

    def read_sample(self):
        line = self.readline_raw()
        if not line or line.lower().startswith("timestamp"):
            return None

        parts = line.split(",")
        if len(parts) != 7:
            return None  # malformed/partial line -- skip rather than crash

        try:
            t, ax, ay, az, gx, gy, gz = (float(p) for p in parts)
        except ValueError:
            return None

        return {
            "timestamp_ms": t,
            "ax": ax, "ay": ay, "az": az,
            "gx": gx, "gy": gy, "gz": gz,
        }
