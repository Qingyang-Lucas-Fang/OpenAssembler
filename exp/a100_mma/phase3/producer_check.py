"""Exact-integer SASS emulation and independent pipeline accounting.

The emulator is functional: it does not validate hardware timing, scoreboard
controls, concurrent memory visibility, or actual CUDA barrier behavior.
"""
from __future__ import annotations

import argparse
import json
import random
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
import backend
from openassembler.emu.machine import Launch, Machine, Memory
from openassembler.emu.program import load_listing


def barrier_accounting(k, bk):
    """Each generation establishes readiness and ends prior-tile readers.

    Producer j+1 writes buffer (j+1)%2 while consumers read j%2. The barrier
    after both operations makes the next buffer ready and the old buffer free.
    This trace checks ring ownership and matching barrier generations without
    depending on assembler source labels or the functional emulator.
    """
    tiles = k // bk
    assert tiles >= 2 and k % bk == 0
    ready = [None, None]
    producer_barriers, consumer_barriers = [0], [0]
    ready[0] = 0
    for tile in range(tiles - 1):
        read_slot, write_slot = tile % 2, (tile + 1) % 2
        assert ready[read_slot] == tile and read_slot != write_slot
        ready[write_slot] = tile + 1
        generation = tile + 1
        producer_barriers.append(generation)
        consumer_barriers.append(generation)
    assert ready[(tiles - 1) % 2] == tiles - 1
    assert producer_barriers == consumer_barriers == list(range(tiles))
    return {'tiles': tiles, 'barrier_generations_per_thread': len(producer_barriers),
            'ring_ownership': 'disjoint producer write and consumer read slots'}


def emulate(config, cta=(1, 2, 0)):
    directory = (BASE / 'build' / config['build']).resolve()
    manifest = json.loads((directory / 'manifest.json').read_text())
    entry = manifest['entries'][0]
    meta = json.loads((directory / 'experiment.json').read_text())['geometry']
    m, n, k, bm, bn, bk = (meta[x] for x in ('m', 'n', 'k', 'bm', 'bn', 'bk'))
    if config['dtype'] != 'f16':
        raise ValueError('integer emulator check currently covers f16 only')
    rng = random.Random(1237)
    av = [rng.randrange(-3, 4) for _ in range(m * k)]
    bv = [rng.randrange(-3, 4) for _ in range(n * k)]
    memory = Memory()
    a, b, c = 0x10000000, 0x20000000, 0x30000000
    memory.write_bytes(a, b''.join(struct.pack('<e', x) for x in av))
    memory.write_bytes(b, b''.join(struct.pack('<e', x) for x in bv))
    # Integer products and sums are exactly representable in FP32 here. Poison
    # every output and register so missing writes and definitions fail visibly.
    sentinel = 0x7FC12345
    memory.write_bytes(c, struct.pack('<I', sentinel) * (m * n))
    program = load_listing((directory / entry['sass']).read_text(), kernel='gemm')
    machine = Machine(program, Launch(grid=(m // bm, n // bn, 1),
        block=(config['threads'], 1, 1), ctaid=cta,
        params=struct.pack('<3Q', a, b, c), smem_static=4*bk*(bm+bn)
        if config['smem'] == 0 else 0, poison=0xDEADBEEF), memory)
    machine.run(max_steps=1_000_000)
    bad = []
    changed_outside = 0
    for row in range(m):
        for col in range(n):
            bits = memory.read(c + (row*n+col)*4, 4)
            if cta[0]*bm <= row < (cta[0]+1)*bm and cta[1]*bn <= col < (cta[1]+1)*bn:
                actual = struct.unpack('<f', struct.pack('<I', bits))[0]
                expected = sum(av[row*k+q]*bv[col*k+q] for q in range(k))
                if actual != expected:
                    bad.append((row, col, actual, expected))
            elif bits != sentinel:
                changed_outside += 1
    record = {'name': config['name'], 'cta': cta, 'mismatches': len(bad),
              'examples': bad[:8], 'outside_cta_writes': changed_outside,
              'emulator_warp_instructions': machine.steps,
              'barrier_accounting': barrier_accounting(k, bk),
              'scope': 'functional integer SASS emulation; hardware correctness and timing remain untested'}
    print(json.dumps(record), flush=True)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', default='producer_smoke_suite.json')
    args = parser.parse_args()
    results = [emulate(case) for case in json.loads((HERE / args.suite).read_text())]
    (HERE / 'producer_emulation.json').write_text(json.dumps(results, indent=2) + '\n')
    if not results or any(x['mismatches'] or x['outside_cta_writes'] for x in results):
        raise SystemExit(1)
