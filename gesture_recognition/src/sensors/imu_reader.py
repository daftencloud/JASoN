"""
imu_reader.py
----------------------------------------------------
Reads accelerometer + gyroscope samples from the course-provided ESP32
+ BMI270 IMU board. IMPORTANT: this board's firmware is already
flashed by the course (per the IMU lab, wshanmu/IMU_lab_students) --
no custom firmware needed, unlike the earlier draft of this project.

The firmware streams lines in this exact format:

    accel[g] x= 0.012 y=-0.034 z= 0.998 | gyro[dps] x= 0.10 y=-0.20 z= 0.05

Note the numbers can have a leading space instead of a minus sign
(e.g. "x= 0.012" for positive, "y=-0.034" for negative) -- the parser
below handles both.

Port note (macOS): use /dev/cu.* not /dev/tty.* -- this matches the
course's own IMU lab instructions and avoids the connection issues we
ran into elsewhere in this project.
"""

import re
import time

from .base_reader import BaseSensorReader

# Matches: accel[g] x= 0.012 y=-0.034 z= 0.998 | gyro[dps] x= 0.10 y=-0.20 z= 0.05
LINE_PATTERN = re.compile(
    r"accel\[g\]\s*x=\s*(-?[\d.]+)\s*y=\s*(-?[\d.]+)\s*z=\s*(-?[\d.]+)\s*"
    r"\|\s*gyro\[dps\]\s*x=\s*(-?[\d.]+)\s*y=\s*(-?[\d.]+)\s*z=\s*(-?[\d.]+)"
)


class ImuReader(BaseSensorReader):
    def __init__(self, port: str, baud: int = 115200):
        super().__init__(port, baud, name="imu")

    def read_sample(self):
        line = self.readline_raw()
        if not line:
            return None

        match = LINE_PATTERN.search(line)
        if not match:
            return None  # not an accel/gyro line (could be a boot message, etc.)

        ax, ay, az, gx, gy, gz = (float(v) for v in match.groups())

        # The firmware doesn't send its own timestamp in this format,
        # so we timestamp at receipt time instead. Fine for windowed
        # feature extraction since sample order is preserved and the
        # stream rate is steady (~100 Hz per the lab spec).
        return {
            "timestamp_ms": time.time() * 1000,
            "ax": ax, "ay": ay, "az": az,
            "gx": gx, "gy": gy, "gz": gz,
        }
