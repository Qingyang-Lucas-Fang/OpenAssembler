"""Reallocated arithmetic compared with recorded ptxas code on adversarial bits.

Every selected instruction must also encode. Poisoned registers catch accidental
reads of old physical registers or private-fragment inputs clobbering their caller.
"""
import random
import struct

import pytest

from openassembler.backend.fragments import library
from tests.backend_harness import compile_and_diff

EDGES = [0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000,
         0x3f000000, 0x3f800000, 0xbf800000, 0x40000000, 0x7f7fffff,
         0x7f800000, 0xff800000, 0x7fc00001, 0x7fffffff, 0xffffffff]


@pytest.mark.parametrize('key', sorted(library()))
def test_arithmetic_matches_ptxas_listing(key):
    f = library()[key]
    rng = random.Random(935)
    pairs = [(a, b) for a in EDGES for b in EDGES]
    pairs += [(rng.getrandbits(32), rng.getrandbits(32)) for _ in range(64)]
    data = b''.join(struct.pack('<4I', a, b, 0x13579bdf, 0x2468ace0) for a, b in pairs)
    _, _, bad = compile_and_diff(f['ptx'], f['kernel'], {'in': data, 'out': bytes(len(data))},
                                {'k': 0, 'kf': 0}, block=(len(pairs), 1, 1), reference_sass=f['sass'])
    assert not bad, bad[:8]
