"""
uwb_reader.py
----------------------------------------------------
Reads live ranging distance between two DWM3001CDK boards, using the
COURSE'S OWN VERIFIED SCRIPT (run_fira_twr.py from wshanmu/UWB_lab) as
a subprocess, rather than reimplementing Qorvo's binary UCI protocol
from scratch. That protocol is genuinely complex (structured binary
commands, not plain text), and the course already has a working,
tested implementation -- reusing it directly is far more reliable than
a from-scratch reimplementation.

REQUIRES: the UWB_lab repo cloned and set up per the lab instructions
(conda env "py39", uwb-qorvo-tools installed via pip install -e .).
Point `uwb_lab_tools_path` at your cloned repo's `uwb-qorvo-tools`
folder, e.g. ~/UWB_lab/uwb-qorvo-tools.

HOW IT WORKS: connect() launches the controlee (the board that just
responds) and the controller (the board that prints ranging results)
as two background subprocesses, both running run_fira_twr.py with a
long duration. A background thread continuously reads the
controller's stdout and queues up parsed distance readings.
read_sample() drains that queue non-blocking, matching every other
reader's interface.

NOTE: this reader launches run_fira_twr.py using the SAME Python
interpreter/environment that's currently running (sys.executable) --
if you're running your gesture project's collect.py etc. from the
"cosmos-ds" conda env rather than "py39", this will likely fail since
the required uwb-uci/uqt-utils packages are only installed in "py39".
Either run your gesture project from the py39 env too, or see the
README for a note on reconciling the two environments.
"""

import queue
import re
import subprocess
import sys
import threading
import time

from .base_reader import BaseSensorReader

DISTANCE_PATTERN = re.compile(r"distance:\s*([\d.]+)\s*cm")
STATUS_PATTERN = re.compile(r"status:\s*(\w+)")


class UwbReader(BaseSensorReader):
    def __init__(self, controller_port: str, controlee_port: str,
                 uwb_lab_tools_path: str, channel: int = 5,
                 preamble_idx: int = 9, name: str = "uwb"):
        # NOTE: this reader's constructor shape differs from the other
        # readers (needs two ports, not one) -- collect.py/etc. need a
        # small adjustment to pass both. See README for the updated
        # --uwb-controller-port / --uwb-controlee-port flags.
        super().__init__(controller_port, baud=None, name=name)
        self.controller_port = controller_port
        self.controlee_port = controlee_port
        self.uwb_lab_tools_path = uwb_lab_tools_path
        self.channel = channel
        self.preamble_idx = preamble_idx

        self._controlee_proc = None
        self._controller_proc = None
        self._line_queue = queue.Queue()
        self._reader_thread = None
        self._last_status = None

    def connect(self):
        run_script = f"{self.uwb_lab_tools_path}/scripts/fira/run_fira_twr/run_fira_twr.py"

        # Start the controlee first (it just waits and responds).
        self._controlee_proc = subprocess.Popen(
            [sys.executable, run_script,
             "-p", self.controlee_port,
             "--controlee",
             "--channel", str(self.channel),
             "--preamble-idx", str(self.preamble_idx),
             "--aoa-report", "all-disabled",
             "-t", "3600"],  # long duration -- we control the actual
                              # session length from collect.py, not here
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1.0)  # give the controlee a moment to be ready

        # Start the controller (this one prints ranging results).
        self._controller_proc = subprocess.Popen(
            [sys.executable, run_script,
             "-p", self.controller_port,
             "--channel", str(self.channel),
             "--preamble-idx", str(self.preamble_idx),
             "--aoa-report", "all-disabled",
             "-t", "3600"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

        self._reader_thread = threading.Thread(
            target=self._read_stdout_loop, daemon=True
        )
        self._reader_thread.start()

        time.sleep(2.0)  # let ranging actually start before returning

    def _read_stdout_loop(self):
        for line in self._controller_proc.stdout:
            self._line_queue.put(line)

    def close(self):
        for proc in (self._controller_proc, self._controlee_proc):
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()

    @property
    def is_connected(self) -> bool:
        return self._controller_proc is not None and self._controller_proc.poll() is None

    def read_sample(self):
        got_distance = None
        # Drain whatever lines are currently queued, tracking the most
        # recent status so we only report distances that came with a
        # real "Ok" status (matching what the lab's own ranging logic
        # treats as a valid measurement -- RangingRxTimeout means no
        # measurement, not zero distance).
        try:
            while True:
                line = self._line_queue.get_nowait()

                status_match = STATUS_PATTERN.search(line)
                if status_match:
                    self._last_status = status_match.group(1)

                distance_match = DISTANCE_PATTERN.search(line)
                if distance_match and self._last_status == "Ok":
                    got_distance = float(distance_match.group(1))
        except queue.Empty:
            pass

        if got_distance is None:
            return None

        return {
            "timestamp_ms": time.time() * 1000,
            "distance_cm": got_distance,
        }
