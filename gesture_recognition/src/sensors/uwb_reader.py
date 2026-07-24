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

import atexit
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
                 preamble_idx: int = 11, name: str = "uwb"):
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
        """
        Starts continuous ranging, matching the lab's own design:
        collect_dataset.py starts ranging ONCE and keeps it running
        for an entire session, only marking trial boundaries around
        the continuous stream -- it does NOT restart ranging per
        trial. Our earlier version killed and relaunched these
        subprocesses on every single trial, which is almost certainly
        what caused the flaky 0-sample trials: each relaunch is a
        fresh race condition on the USB ports.

        Now: if ranging is already running (from a previous trial in
        this same session), this is a no-op -- we just keep using the
        same live connection. Only the FIRST call in a session
        actually launches the subprocesses.
        """
        if self.is_connected:
            # Already ranging from a previous trial -- reuse the live
            # connection, but clear out any readings that piled up
            # during the idle time between trials (e.g. while you were
            # getting ready for the next gesture), so this trial
            # starts clean instead of possibly returning a stale
            # leftover distance as its first sample.
            try:
                while True:
                    self._line_queue.get_nowait()
            except queue.Empty:
                pass
            return

        # If we get here on anything other than the very first call in
        # a session, the previous ranging subprocess died unexpectedly
        # -- that's the real cause of the repeated relaunch/lock-on
        # cycle you're seeing every trial instead of just once. Print
        # diagnostics so we can see WHY instead of guessing.
        if self._controller_proc is not None:
            print(f"  [UWB] NOTE: previous controller process is no longer "
                  f"alive (exit code: {self._controller_proc.poll()}) -- "
                  f"relaunching. This is why lock-on is happening again "
                  f"instead of reusing the prior connection.")
            try:
                stderr_output = self._controller_proc.stderr.read()
                if stderr_output:
                    print(f"  [UWB] Controller stderr: {stderr_output[-500:]}")
            except Exception:
                pass
        if self._controlee_proc is not None:
            print(f"  [UWB] NOTE: previous controlee process exit code: "
                  f"{self._controlee_proc.poll()}")
            try:
                stderr_output = self._controlee_proc.stderr.read()
                if stderr_output:
                    print(f"  [UWB] Controlee stderr: {stderr_output[-500:]}")
            except Exception:
                pass

        run_script = f"{self.uwb_lab_tools_path}/scripts/fira/run_fira_twr/run_fira_twr.py"

        # Start the controlee first (it just waits and responds).
        self._controlee_proc = subprocess.Popen(
            [sys.executable, run_script,
             "-p", self.controlee_port,
             "--controlee",
             "--channel", str(self.channel),
             "--preamble-idx", str(self.preamble_idx),
             "--aoa-report", "all-disabled",
             "--slot-span", "2400",
             "--slots-per-rr", "6",
             "--ranging-span", "20",  # ~50Hz -- matches the teammate's
                                       # proven working config. Without
                                       # this, the script falls back to
                                       # its slow default (~5Hz), giving
                                       # far fewer samples per trial and
                                       # more time spent on initial
                                       # RangingRxTimeout noise before lock-on.
             "-t", "3600"],  # long duration -- covers a full multi-trial
                              # collection session, not just one trial
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(1.0)  # give the controlee a moment to be ready

        # Start the controller (this one prints ranging results).
        self._controller_proc = subprocess.Popen(
            [sys.executable, run_script,
             "-p", self.controller_port,
             "--channel", str(self.channel),
             "--preamble-idx", str(self.preamble_idx),
             "--aoa-report", "all-disabled",
             "--slot-span", "2400",
             "--slots-per-rr", "6",
             "--ranging-span", "20",
             "-t", "3600"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        self._reader_thread = threading.Thread(
            target=self._read_stdout_loop, daemon=True
        )
        self._reader_thread.start()

        # Register a real cleanup to run when the whole Python process
        # exits (e.g. collect.py finishes all --trials), regardless of
        # how many times close() gets called mid-session. Registered
        # only once, the first time we actually connect.
        atexit.register(self._real_close)

        # DON'T just sleep a fixed amount and hope ranging is ready --
        # real observed behavior is that the boards print a long
        # stretch of RangingRxTimeout failures (sometimes 20-30+
        # seconds worth) before ranging actually locks on and starts
        # reporting real "status: Ok" measurements. A short fixed
        # sleep here was the actual cause of trial 1 consistently
        # getting 0 samples -- recording started before ranging had
        # locked on at all. Instead, actively wait for the first real
        # measurement to appear, with a generous timeout.
        print("  [UWB] Waiting for ranging to lock on (this can take up to "
              "30s on first connect)...")
        locked_on = False
        wait_deadline = time.time() + 40.0
        last_status = None
        while time.time() < wait_deadline:
            try:
                line = self._line_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            status_match = STATUS_PATTERN.search(line)
            if status_match:
                last_status = status_match.group(1)
            if last_status == "Ok" and DISTANCE_PATTERN.search(line):
                locked_on = True
                break

        if locked_on:
            print("  [UWB] Ranging locked on.")
        else:
            print("  [UWB] WARNING: ranging did not lock on within 40s -- "
                  "proceeding anyway, but this trial and possibly the next "
                  "few may get 0 samples. Check board positioning/distance.")

    def _read_stdout_loop(self):
        for line in self._controller_proc.stdout:
            self._line_queue.put(line)

    def close(self):
        """
        Soft close -- does NOT actually kill the ranging subprocesses.
        collect.py calls close() after every trial for every sensor
        (matching the other readers' per-trial lifecycle), but for
        UWB specifically we want the ranging session to stay alive
        across trials -- see the note in connect(). Real cleanup
        happens automatically at program exit via _real_close(),
        registered through atexit.
        """
        pass

    def _real_close(self):
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
