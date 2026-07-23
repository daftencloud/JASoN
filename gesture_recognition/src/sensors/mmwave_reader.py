"""
mmwave_reader.py
----------------------------------------------------
Reads point-cloud frames from the TI IWRL6432FSPEVM, using the SAME
protocol confirmed working by the course's own `mmwave_lab` repo
(wshanmu/mmwave_lab, get_range_profile.py). Ported directly from that
verified source rather than reconstructed independently.

KEY CORRECTIONS from an earlier draft of this file:
  1. This is a SINGLE-PORT protocol -- the same serial connection is
     used for the text CLI configuration handshake AND the binary
     frame data afterward. There is no separate high-baud "DATA port."
  2. Default baud is 115200 (the CLI's default), not 921600.
  3. Before any binary frames arrive, you must send the radar its
     configuration commands (from a .cfg file) over the CLI and wait
     for text acknowledgements -- the demo firmware sends nothing in
     binary until this handshake completes and `sensorStart` is sent.

CFG FILE: point a `cfg_path` at one of the .cfg files from your cloned
mmwave_lab repo, e.g. `xwrL64xx-evm/point_cloud.cfg` for point-cloud
data (what gesture recognition needs) or `xwrL64xx-evm/hand_distance.cfg`
for range-profile-only data.

The binary frame header/TLV parsing logic itself (magic word, 40-byte
header, TLV type+length+payload) matches what get_range_profile.py
uses and is confirmed correct against real hardware -- only the missing
CLI handshake and wrong default baud were bugs in the earlier version
of this file.
"""

import math
import struct
import time

from .base_reader import BaseSensorReader

MAGIC_WORD = bytes([0x02, 0x01, 0x04, 0x03, 0x06, 0x05, 0x08, 0x07])
HEADER_SIZE = 40  # 8 magic + 8 x uint32 fields
TLV_HEADER_SIZE = 8  # type(4) + length(4)

# Point-cloud TLVs come in TWO possible formats, both confirmed real by
# point_cloud_viewer.py in the course's mmwave_lab repo. This demo
# firmware apparently uses the FIXED-point format (301/1020), not the
# float format (1) -- an earlier version of this file only handled
# type 1 and silently discarded every real point as a result.
TLV_TYPE_POINT_CLOUD_FLOAT = 1
TLV_TYPES_POINT_CLOUD_FIXED = {301, 1020}
BYTES_PER_FLOAT_POINT = 16  # x, y, z, doppler as 4x float32
MAX_PACKET_SIZE = 8192

def _decode_fixed_points(payload: bytes) -> list:
    """
    Decodes the fixed-point point-cloud TLV format (types 301/1020),
    ported directly from point_cloud_viewer.py's decode_fixed_points().
    """
    if len(payload) < 20:
        return []

    xyz_unit, doppler_unit, _snr_unit, _noise_unit = struct.unpack_from(
        "<ffff", payload, 0
    )
    num_major_points, _num_minor_points = struct.unpack_from("<HH", payload, 16)

    points = []
    offset = 20
    for _ in range(num_major_points):
        if offset + 10 > len(payload):
            break
        x, y, z, doppler, _snr, _noise = struct.unpack_from(
            "<hhhhBB", payload, offset
        )
        real_x = x * xyz_unit
        real_y = y * xyz_unit
        real_z = z * xyz_unit
        real_doppler = doppler * doppler_unit

        r = math.sqrt(real_x * real_x + real_y * real_y + real_z * real_z)
        horiz = math.sqrt(real_x * real_x + real_y * real_y)
        angle = math.degrees(math.atan2(real_x, real_y))
        elevation = math.degrees(math.atan2(real_z, horiz)) if horiz > 0 else 0.0

        points.append({
            "range": r, "doppler": real_doppler,
            "angle": angle, "elevation": elevation,
        })
        offset += 10

    return points


def _decode_float_points(payload: bytes) -> list:
    """Decodes the float point-cloud TLV format (type 1)."""
    points = []
    num_points = len(payload) // BYTES_PER_FLOAT_POINT
    for p in range(num_points):
        po = p * BYTES_PER_FLOAT_POINT
        x, y, z, doppler = struct.unpack_from("<ffff", payload, po)
        r = math.sqrt(x * x + y * y + z * z)
        horiz = math.sqrt(x * x + y * y)
        angle = math.degrees(math.atan2(x, y))
        elevation = math.degrees(math.atan2(z, horiz)) if horiz > 0 else 0.0
        points.append({
            "range": r, "doppler": doppler,
            "angle": angle, "elevation": elevation,
        })
    return points


CLI_FAILURE_PATTERNS = ("error", "not recognized", "invalid", "failed")
CLI_OK_PATTERNS = ("done", "mmwdemo:", "skipped")


def _load_configuration(cfg_path):
    commands = []
    with open(cfg_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith(("%", "#")):
                continue
            commands.append(line)
    if not any(line.startswith("sensorStart") for line in commands):
        raise ValueError(f"{cfg_path} has no sensorStart command.")
    return commands


class MmwaveReader(BaseSensorReader):
    def __init__(self, port: str, baud: int = 115200, cfg_path: str = None):
        super().__init__(port, baud, name="mmwave")
        if cfg_path is None:
            raise ValueError(
                "MmwaveReader requires cfg_path -- point it at a .cfg file "
                "from your cloned mmwave_lab repo, e.g. "
                "'xwrL64xx-evm/point_cloud.cfg' for gesture/point-cloud data."
            )
        self.cfg_path = cfg_path
        self._buffer = bytearray()

    def connect(self):
        """
        Opens the port and runs the CLI configuration handshake before
        returning -- this is the step that was missing before. After
        this completes, the radar is streaming binary frames on this
        same port.
        """
        super().connect()  # opens self._ser at self.baud
        time.sleep(0.5)

        commands = _load_configuration(self.cfg_path)
        start_command = None

        for command in commands:
            if command.startswith("sensorStart"):
                start_command = command
                continue
            if command.split()[0] == "baudRate":
                # Skip baudRate commands -- keep the host at self.baud,
                # matching get_range_profile.py's default behavior
                # (macOS USB-serial adapters are more reliable staying
                # at one fixed rate).
                continue

            self._ser.write((command + "\n").encode("ascii"))
            self._ser.flush()
            reply = self._read_text_until_quiet()

            reply_lower = reply.lower()
            if any(p in reply_lower for p in CLI_FAILURE_PATTERNS):
                raise RuntimeError(f"Radar rejected command {command!r}: {reply}")

        if start_command is None:
            raise ValueError("No sensorStart command found in cfg.")

        self._ser.reset_input_buffer()
        self._ser.write((start_command + "\n").encode("ascii"))
        self._ser.flush()
        time.sleep(0.2)

    def _read_text_until_quiet(self, quiet_time=0.15, max_time=2.0):
        start = time.monotonic()
        last_rx = start
        chunks = []
        while time.monotonic() - start < max_time:
            waiting = self._ser.in_waiting
            if waiting:
                chunks.append(self._ser.read(waiting))
                last_rx = time.monotonic()
                continue
            if time.monotonic() - last_rx >= quiet_time:
                break
            time.sleep(0.01)
        return b"".join(chunks).decode("ascii", errors="ignore")

    def _find_magic_word(self) -> int:
        return self._buffer.find(MAGIC_WORD)

    def _try_parse_packet(self):
        idx = self._find_magic_word()
        if idx < 0:
            if len(self._buffer) > MAX_PACKET_SIZE:
                self._buffer = self._buffer[-MAX_PACKET_SIZE // 2:]
            return None

        if idx > 0:
            del self._buffer[:idx]

        if len(self._buffer) < HEADER_SIZE:
            return None

        total_packet_len = struct.unpack_from("<I", self._buffer, 12)[0]
        num_tlvs = struct.unpack_from("<I", self._buffer, 28)[0]

        if not (HEADER_SIZE <= total_packet_len <= MAX_PACKET_SIZE) or num_tlvs > 64:
            del self._buffer[:8]
            return None

        if len(self._buffer) < total_packet_len:
            return None

        points = []
        offset = HEADER_SIZE
        for _ in range(num_tlvs):
            if offset + TLV_HEADER_SIZE > total_packet_len:
                break
            tlv_type, tlv_length = struct.unpack_from("<II", self._buffer, offset)
            payload_offset = offset + TLV_HEADER_SIZE

            if payload_offset + tlv_length > total_packet_len:
                break

            if tlv_type == TLV_TYPE_POINT_CLOUD_FLOAT:
                points.extend(_decode_float_points(self._buffer[payload_offset:payload_offset + tlv_length]))
            elif tlv_type in TLV_TYPES_POINT_CLOUD_FIXED:
                points.extend(_decode_fixed_points(self._buffer[payload_offset:payload_offset + tlv_length]))

            offset = payload_offset + tlv_length

        del self._buffer[:total_packet_len]
        return {"points": points}

    def read_sample(self):
        if self._ser is not None and self._ser.in_waiting:
            self._buffer += self._ser.read(self._ser.in_waiting)

        parsed = self._try_parse_packet()
        if parsed is None:
            return None

        return {
            "timestamp_ms": time.time() * 1000,
            "num_points": len(parsed["points"]),
            "points": parsed["points"],
        }
