"""Hardware ground truth, replayed offline: emulator vs recorded A100 outputs.

The .raw.json.gz files hold inputs and per-thread outputs from real A100 runs
(tools/gpu_modal.py). No GPU is needed to run these tests.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "tools")]

import gpu_common  # noqa: E402
import gpu_replay  # noqa: E402

C = ROOT / "corpus" / "sm_80"
# The recorded A100 outputs (*.raw.json.gz) are committed, but replaying them runs
# the emulator on each probe's SASS, which is build output (tools/build_corpus.py).
needs_corpus = pytest.mark.skipif(not (C / "sass").is_dir() or not any((C / "sass").glob("*.sass")),
                                  reason="build the probe corpus first: python tools/build_corpus.py sm_80 (needs the CUDA toolkit)")

# Recorded runs that the current probes/emulator must reproduce.
# v2: full corpus (its st.shared.* used a since-fixed racy probe -> excluded, covered by fix5)
RUNS = {
    "gpu_a100_40gb_v2.raw.json.gz": {"st.shared.b32", "st.shared.b64", "st.shared.v2.b32", "st.shared.v4.b32"},
    "gpu_a100_40gb_tensor.raw.json.gz": {"ldmatrix.trans"},  # old probe read uninitialized smem
    "gpu_a100_40gb_fix5.raw.json.gz": set(),
}


@needs_corpus
@pytest.mark.parametrize("run", sorted(RUNS))
def test_emulator_reproduces_a100(run):
    path = C / run
    if not path.exists():
        pytest.skip(f"{run} not recorded")
    res = gpu_replay.replay(path)
    bad = []
    for name, r in res.items():
        if name in RUNS[run] or gpu_common.expected_reason(name):
            continue  # approx (tolerance-compared elsewhere) / nondeterministic / known-stale
        if r["status"] != "match":
            bad.append((name, r))
    assert not bad, bad
    assert len(res) > 0


@needs_corpus
def test_tensor_cores_bit_exact():
    """mma f16/bf16/tf32/s8/f64, ldmatrix, cp.async: every lane equals the A100."""
    path = C / "gpu_a100_40gb_tensor.raw.json.gz"
    if not path.exists():
        pytest.skip("tensor run not recorded")
    res = gpu_replay.replay(path)
    for name in ("mma.f16", "mma.bf16", "mma.tf32", "mma.s8", "mma.f64", "ldmatrix", "cp.async", "cp.async.ca4"):
        assert res[name]["status"] == "match" and res[name]["lanes"] == 512, (name, res[name])


def test_regcount_rule():
    """Measured on A100 (gpu_a100_40gb_bisect5.json): writing Rd needs regcount >= Rd + 3."""
    from openassembler.container.cubin import required_regcount

    assert required_regcount(["MOV R31, R28"]) == 34
    assert required_regcount(["LDG.E.128 R16, [R12.64]"]) == 22
    assert required_regcount(["IADD3 R4, R2, R3, RZ"]) == 16
    rec = json.loads((C / "gpu_a100_40gb_bisect5.json").read_text())["probes"] if (
        C / "gpu_a100_40gb_bisect5.json").exists() else {}
    for k, v in rec.items():
        d, n = int(k[1:4]), int(k[5:8])
        faulted = v["status"] == "gpu-error"
        if d >= 14:  # below that, ptxas's minimum allocation hides the boundary
            assert faulted == (n < d + 3), (k, v["status"])
