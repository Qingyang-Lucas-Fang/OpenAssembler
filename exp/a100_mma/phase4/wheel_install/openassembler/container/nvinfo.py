"""`.nv.info` / `.nv.info.<kernel>` attribute records.

Every record starts with a 4-byte header: format byte, attribute byte, u16 field.
  0x01 EIFMT_NVAL  u16 unused (0)
  0x02 EIFMT_BVAL  u16 holds the byte value
  0x03 EIFMT_HVAL  u16 holds the value
  0x04 EIFMT_SVAL  u16 = payload size, followed by `size` bytes
Measured on smoke.cubin ("01 35 00 00", "03 19 18 00" = CBANK_PARAM_SIZE 0x18,
"03 1b ff 00" = MAXREG_COUNT 0xff), round-trip-checked on the whole corpus.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

EIATTR = {0x04: "CTAIDZ_USED", 0x0A: "PARAM_CBANK", 0x12: "FRAME_SIZE", 0x17: "KPARAM_INFO",
          0x19: "CBANK_PARAM_SIZE", 0x1B: "MAXREG_COUNT", 0x1C: "EXIT_INSTR_OFFSETS",
          0x1D: "S2RCTAID_INSTR_OFFSETS", 0x23: "MAX_STACK_SIZE", 0x2F: "REGCOUNT",
          0x36: "SW1850030_WAR", 0x37: "CUDA_API_VERSION", 0x50: "SW2861232_WAR", 0x5F: "MIN_STACK_SIZE"}
NAME_TO_CODE = {v: k for k, v in EIATTR.items()}


@dataclass
class Rec:
    fmt: int
    attr: int
    data: bytes   # SVAL payload; for N/B/HVAL the raw 2-byte header field

    @property
    def name(self) -> str:
        return EIATTR.get(self.attr, f"0x{self.attr:02x}")


def parse(b: bytes) -> list[Rec]:
    out, i = [], 0
    while i < len(b):
        fmt, attr, v = b[i], b[i + 1], struct.unpack_from("<H", b, i + 2)[0]
        i += 4
        if fmt == 4:
            data, i = b[i:i + v], i + v
        elif fmt in (1, 2, 3):
            data = b[i - 2:i]
        else:
            raise ValueError(f"bad nv.info format byte {fmt:#x} at {i - 4}")
        out.append(Rec(fmt, attr, data))
    return out


def serialize(recs: list[Rec]) -> bytes:
    out = bytearray()
    for r in recs:
        out += bytes([r.fmt, r.attr])
        if r.fmt == 4:
            out += struct.pack("<H", len(r.data)) + r.data
        else:
            out += r.data
    return bytes(out)


def value(r: Rec) -> int:
    return struct.unpack("<H", r.data)[0] if r.fmt != 4 else int.from_bytes(r.data, "little")


def u32s(r: Rec) -> list[int]:
    return list(struct.unpack(f"<{len(r.data) // 4}I", r.data))
