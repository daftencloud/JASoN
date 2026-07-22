"""
mmwave_reader.py
----------------------------------------------------
Reads point-cloud frames from the TI IWRL6432FSPEVM's DATA UART port,
using TI's standard binary TLV (type-length-value) frame format --
the same format used across TI's mmWave Demo Visualizer ecosystem.

IMPORTANT: connect this reader's `port` to the radar's DATA port, not
the CLI/config port -- they're two separate UARTs typically exposed
over the same USB connection. Check your board or `mmwave_lab` course
materials for which one is which, and the DATA port's baud rate
(921600 is a common default for TI mmWave demos -- update `baud` if
yours differs).

FORMAT NOTE: this implements the header/TLV layout used consistently
across TI's classic mmWave SDK demos and generally preserved in the
newer L-SDK for visualizer compatibility. Some specific demo firmware
builds reorder or drop header fields -- verify against your actual
frame data (see README "Calibration Procedure": numTLVs and
totalPacketLen should look sane; garbage values mean your firmware
build uses a different header layout).
"""

import math
import struct

from .base_reader import BaseSensorReader

MAGIC_WORD = bytes([0x02, 0x01, 0x04, 0x03, 0x06, 0x05, 0x08, 0x07])
HEADER_SIZE = 40  # 8 magic + 8 x uint32 fields
TLV_HEADER_SIZE = 8  # type(4) + length(4)
TLV_TYPE_DETECTED_POINTS = 1
BYTES_PER_POINT = 16  # x, y, z, doppler as 4x float32
MAX_PACKET_SIZE = 8192


class MmwaveReader(BaseSensorReader):
    def __init__(self, port: str, baud: int = 921600):
        super().__init__(port, baud, name="mmwave")
        self._buffer = bytearray()

    def _find_magic_word(self) -> int:
        return self._buffer.find(MAGIC_WORD)

    def _try_parse_packet(self):
        idx = self._find_magic_word()
        if idx < 0:
            # No magic word anywhere in the buffer -- avoid unbounded
            # growth on garbage/pre-stream noise.
            if len(self._buffer) > MAX_PACKET_SIZE:
                self._buffer = self._buffer[-MAX_PACKET_SIZE // 2:]
            return None

        if idx > 0:
            del self._buffer[:idx]  # discard bytes before the magic word

        if len(self._buffer) < HEADER_SIZE:
            return None  # header not fully received yet

        total_packet_len = struct.unpack_from("<I", self._buffer, 12)[0]
        num_detected_obj = struct.unpack_from("<I", self._buffer, 24)[0]
        num_tlvs = struct.unpack_from("<I", self._buffer, 28)[0]

        # Sanity bounds -- reject an implausible "match" and resync
        # past this magic word rather than getting stuck (see README
        # note on header layout mismatches).
        if not (HEADER_SIZE <= total_packet_len <= MAX_PACKET_SIZE) or num_tlvs > 32:
            del self._buffer[:8]
            return None

        if len(self._buffer) < total_packet_len:
            return None  # full packet not received yet

        points = []
        offset = HEADER_SIZE
        for _ in range(num_tlvs):
            if offset + TLV_HEADER_SIZE > total_packet_len:
                break
            tlv_type, tlv_length = struct.unpack_from("<II", self._buffer, offset)
            payload_offset = offset + TLV_HEADER_SIZE

            if payload_offset + tlv_length > total_packet_len:
                break  # malformed/truncated TLV

            if tlv_type == TLV_TYPE_DETECTED_POINTS:
                num_points = tlv_length // BYTES_PER_POINT
                for p in range(num_points):
                    po = payload_offset + p * BYTES_PER_POINT
                    x, y, z, doppler = struct.unpack_from("<ffff", self._buffer, po)
                    r = (x * x + y * y + z * z) ** 0.5
                    horiz = (x * x + y * y) ** 0.5
                    angle = math.degrees(math.atan2(x, y))       # azimuth
                    elevation = math.degrees(math.atan2(z, horiz)) if horiz > 0 else 0.0
                    points.append({
                        "range": r, "doppler": doppler,
                        "angle": angle, "elevation": elevation,
                    })

            offset = payload_offset + tlv_length

        # Consume this packet.
        del self._buffer[:total_packet_len]

        return {
            "num_detected_obj": num_detected_obj,
            "points": points,
        }

    def read_sample(self):
        if self._ser is not None and self._ser.in_waiting:
            self._buffer += self._ser.read(self._ser.in_waiting)

        parsed = self._try_parse_packet()
        if parsed is None:
            return None

        # Flatten the point cloud into a fixed-shape-friendly dict --
        # extract_features.py aggregates across whatever points are
        # present, so a variable point count per frame is fine.
        return {
            "timestamp_ms": None,  # filled in by collect.py at receipt time
            "num_points": len(parsed["points"]),
            "points": parsed["points"],  # list of {range, doppler, angle, elevation}
        }
