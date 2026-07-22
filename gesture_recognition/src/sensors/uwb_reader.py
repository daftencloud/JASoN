"""
uwb_reader.py
----------------------------------------------------
Reads range measurements from your DWM3001C UWB tag module.

IMPORTANT HARDWARE NOTE: DWM3001C modules ship pre-flashed with
Qorvo's ranging demo firmware and are controlled over a UART
shell/AT-command interface -- the exact command/output format depends
on which firmware image is flashed on your specific boards. I don't
have a verified format for your boards, so `read_sample()` below has
ONE clearly marked TODO. Everything else in this project (feature
extraction, training, evaluation) is complete and works the moment
this returns real data in the documented shape.

TO FILL THIS IN:
  1. Connect one UWB module directly via USB, open a serial terminal
     at 115200 baud (`screen /dev/tty.usbmodemXXXX 115200` on Mac, or
     the Arduino Serial Monitor).
  2. Power on the two anchor modules and observe what the tag module
     prints once ranging is active.
  3. Update `read_sample()` below to parse whatever you actually see.
  4. If you need to SEND a command to trigger each reading (rather
     than it streaming automatically), send it once in `connect()`
     via `self._ser.write(b"<command>\\n")`.

Expected output shape once implemented -- a dict with:
    {"timestamp_ms": <float>, "range_anchor1": <float|None>,
     "range_anchor2": <float|None>}
"""

import time

from .base_reader import BaseSensorReader


class UwbReader(BaseSensorReader):
    def __init__(self, port: str, baud: int = 115200):
        super().__init__(port, baud, name="uwb")

    def read_sample(self):
        line = self.readline_raw()
        if not line:
            return None

        # ============================================================
        # TODO (hardware-specific): replace this placeholder parse with
        # your DWM3001C's actual output format. This is a starting
        # guess at a common Qorvo demo pattern (e.g. "AN0:1.23,AN1:2.01")
        # -- UNVERIFIED until you check it against real Serial output.
        # ============================================================
        range_anchor1 = None
        range_anchor2 = None

        for part in line.split(","):
            part = part.strip()
            if part.startswith("AN0:"):
                try:
                    range_anchor1 = float(part[4:])
                except ValueError:
                    pass
            elif part.startswith("AN1:"):
                try:
                    range_anchor2 = float(part[4:])
                except ValueError:
                    pass

        if range_anchor1 is None and range_anchor2 is None:
            return None  # this line wasn't a ranging result -- likely
                          # a log/status line from the module, ignore it

        return {
            "timestamp_ms": time.time() * 1000,
            "range_anchor1": range_anchor1,
            "range_anchor2": range_anchor2,
        }
