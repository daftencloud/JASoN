"""
rfid_reader.py
----------------------------------------------------
Reads tag inventory (EPC + RSSI) from the SparkFun M7E-HECTO reader,
talking directly to the ThingMagic/Mercury API binary serial protocol
in pure Python (no Arduino library needed, since this project is
Python-only per your TA's structure).

PROTOCOL: the M7E uses a binary command/response framing:
    [0xFF][Length][Opcode][Data...][CRC_hi][CRC_lo]
This is the module's own hardware-level UART protocol (not tied to any
particular language/library), documented across ThingMagic's Mercury
API and mirrored in SparkFun's open-source Arduino library source.

⚠️ CONFIDENCE NOTE: the opcodes and general framing below reflect the
Mercury API protocol as I understand it, but I don't have 100%
certainty on exact byte-for-byte values for your specific module/
firmware version. VERIFY this against SparkFun's open-source library
source (github.com/sparkfun/SparkFun_UHF_RFID_Reader_Arduino_Library --
look at `SparkFun_UHF_RFID_Reader.cpp`, which defines these same
opcodes/framing in C++ even though we're not using that library here;
the protocol bytes are hardware-defined, so they transfer directly to
this Python implementation) if reads don't work as expected. This is
NOT a stub -- it's a real protocol implementation -- but double-check
it against the reference source before trusting it fully, same as the
mmWave/UWB notes elsewhere in this project.
"""

import struct
import time

from .base_reader import BaseSensorReader

START_BYTE = 0xFF

OPCODE_VERSION = 0x03
OPCODE_SET_REGION = 0x97
OPCODE_SET_READ_POWER = 0x92
OPCODE_START_READING = 0x22   # "Read Tag ID multiple"
OPCODE_STOP_READING = 0x29

REGION_NORTHAMERICA = 0x01


def _crc16_ccitt(data: bytes) -> int:
    """CRC-16/CCITT-FALSE, the checksum variant used by the Mercury API."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def _build_command(opcode: int, data: bytes = b"") -> bytes:
    length = len(data)
    body = bytes([length, opcode]) + data
    crc = _crc16_ccitt(body)
    return bytes([START_BYTE]) + body + struct.pack(">H", crc)


class RfidReader(BaseSensorReader):
    def __init__(self, port: str, baud: int = 38400, tracked_tag_ids=None):
        super().__init__(port, baud, name="rfid")
        # Order matters: index 0 is assumed to be the thumb tag
        # throughout extract_features.py's Soli-detection logic.
        self.tracked_tag_ids = tracked_tag_ids or []
        self._last_seen = {}   # tag_id -> (timestamp_ms, rssi)
        self._visibility_window_ms = 200

    def connect(self):
        super().connect()
        time.sleep(0.2)

        # Basic bring-up sequence -- confirm the module responds, then
        # configure it and start continuous background reading.
        self._ser.write(_build_command(OPCODE_VERSION))
        time.sleep(0.1)
        _ = self._ser.read(self._ser.in_waiting or 1)  # drain the version response

        self._ser.write(_build_command(OPCODE_SET_REGION, bytes([REGION_NORTHAMERICA])))
        time.sleep(0.05)
        self._ser.write(_build_command(OPCODE_SET_READ_POWER, struct.pack(">H", 500)))  # 5.00 dBm
        time.sleep(0.05)
        self._ser.write(_build_command(OPCODE_START_READING))
        time.sleep(0.05)
        self._ser.reset_input_buffer()

    def close(self):
        if self.is_connected:
            self._ser.write(_build_command(OPCODE_STOP_READING))
            time.sleep(0.05)
        super().close()

    def _parse_tag_response(self, raw: bytes):
        """
        Parses one tag-found response packet. Response layout for
        opcode 0x22 responses generally includes RSSI and the EPC
        bytes toward the end of the payload -- exact offsets can vary
        by firmware version (see module docstring's confidence note).
        This implementation looks for the EPC as the trailing bytes of
        the payload and RSSI as a signed byte a few positions in, which
        matches the common Mercury API tag-found layout.
        """
        if len(raw) < 8:
            return None
        try:
            length = raw[1]
            payload = raw[3:3 + length - 1]  # skip start/length/opcode framing
            if len(payload) < 3:
                return None
            rssi_raw = payload[4] if len(payload) > 4 else payload[0]
            rssi = rssi_raw - 256 if rssi_raw > 127 else rssi_raw  # signed byte
            epc_bytes = payload[-12:] if len(payload) >= 12 else payload
            epc_hex = epc_bytes.hex().upper()
            return {"epc": epc_hex, "rssi": float(rssi)}
        except (IndexError, ValueError):
            return None

    def read_sample(self):
        if self._ser is None or not self._ser.in_waiting:
            return None

        raw = self._ser.read(self._ser.in_waiting)
        # Frames start with START_BYTE -- split on that if multiple
        # frames arrived in one read.
        frames = raw.split(bytes([START_BYTE]))
        now_ms = time.time() * 1000

        for frame in frames:
            if not frame:
                continue
            parsed = self._parse_tag_response(bytes([START_BYTE]) + frame)
            if parsed and parsed["epc"] in self.tracked_tag_ids:
                self._last_seen[parsed["epc"]] = (now_ms, parsed["rssi"])

        # Build a snapshot across all tracked tags based on recent
        # visibility, same "rolling state" approach as the earlier
        # Arduino version -- necessary because tag reads arrive
        # asynchronously, not in lockstep with any fixed poll rate.
        tags = {}
        for tag_id in self.tracked_tag_ids:
            last = self._last_seen.get(tag_id)
            if last and (now_ms - last[0]) < self._visibility_window_ms:
                tags[tag_id] = {"visible": True, "rssi": last[1]}
            else:
                tags[tag_id] = {"visible": False, "rssi": None}

        return {"timestamp_ms": now_ms, "tags": tags}
