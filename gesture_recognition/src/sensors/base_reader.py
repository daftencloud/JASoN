"""
base_reader.py
----------------------------------------------------
Abstract base class every sensor reader implements. Defines one
consistent interface (connect, read_sample, close) so collect.py can
treat all five sensors identically without knowing their internal
protocols.

Every reader ultimately produces samples as plain Python dicts with:
    {"timestamp_ms": <float>, ...sensor-specific numeric fields...}
which get written straight to CSV by collect.py -- no sensor-specific
logic lives outside each reader's own file.
"""

from abc import ABC, abstractmethod
import time

import serial


class BaseSensorReader(ABC):
    """
    Common serial connection handling. Subclasses implement
    `read_sample()` with their own protocol parsing and return either
    a dict of parsed values, or None if no new sample is available yet
    (non-blocking -- callers should call this in a tight loop and
    handle None gracefully).
    """

    def __init__(self, port: str, baud: int, name: str):
        self.port = port
        self.baud = baud
        self.name = name
        self._ser = None

    def connect(self):
        self._ser = serial.Serial(self.port, self.baud, timeout=0.05)
        time.sleep(2)  # let the board's bootloader/reset settle
        self._ser.reset_input_buffer()

    def close(self):
        if self._ser is not None and self._ser.is_open:
            self._ser.close()

    @property
    def is_connected(self) -> bool:
        return self._ser is not None and self._ser.is_open

    @abstractmethod
    def read_sample(self):
        """
        Non-blocking. Returns one parsed sample dict, or None if no
        complete new sample is available this call. Must never block
        waiting for data -- callers rely on this to poll many sensors
        in one loop without one slow sensor stalling the others.
        """
        raise NotImplementedError

    def readline_raw(self) -> str:
        """
        Helper for line-based protocols (IMU, WiFi CSI): attempts to
        read one line using the port's own read timeout, returning ""
        if nothing arrived in time.

        NOTE: this deliberately does NOT gate on `self._ser.in_waiting`
        first. Some USB-serial chip drivers (notably certain WCH/CH340
        drivers on macOS) report in_waiting as always 0 even while
        real data is actively streaming -- gating on it silently
        starves the reader of data forever. A direct readline() call
        respects the port's configured timeout (set in connect()) and
        works correctly regardless of whether in_waiting is reliable
        on a given driver.
        """
        if self._ser is None:
            return ""
        line = self._ser.readline().decode(errors="ignore").strip()
        return line
