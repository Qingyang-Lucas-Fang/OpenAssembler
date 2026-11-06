"""Explicit direct-pointer mode, validated independently on A100.

The default legacy stream hashes remain in test_backend_tensor.py. These
streams come from the paired normal CLI builds retained by the experiment in
exp/weaknesses/gemm_pointer_homes.py, not a vendor-kernel replacement.
"""
from collections import Counter
import hashlib

import pytest

from openassembler import oracle
from openassembler.backend import ra, tensor
from openassembler.backend.compile import Out
from openassembler.isa.word import control, control_valid
from openassembler.kernels.gemm import GemmPlan, generate
from openassembler.ptx.parse import parse_module


def kernel(shape=(512, 512, 2048), dtype="f16"):
    return parse_module(generate(GemmPlan(*shape, dtype))).kernels["gemm"]


@pytest.mark.parametrize("shape,expected,words,registers", [
    ((128, 128, 128), {
        "f16": "064f835c67c4c76193003e3eb055269866be961caead76905be0f359086236fe",
        "bf16": "172eb4250a020595af5b2e4ccc3d60ef9e04d0c40792df6ac4f598255e4d192c"}, 952, 184),
    ((512, 512, 2048), {
        "f16": "ffe1e38f95f6349bca0035ea28ed0953051f215b351d37f0e7b59dbe04a5aea0",
        "bf16": "dfd5dc46fd41e1c5cd42d5f72d6d1016618849806b1a7fb8bff6d65295e7c18c"}, 1472, 216),
    ((4096, 4096, 4096), {
        "f16": "581dc095ba8842aed0852bbae7bc6b2b5b8b92b239096ff1ad49c1d4c0e65929",
        "bf16": "b0c57e6cda2e1a808457a7b53390b7b59c1c5d76833e5b1d7ad30274538d6b1a"}, 1472, 216),
    ((16384, 4096, 4096), {
        "f16": "581dc095ba8842aed0852bbae7bc6b2b5b8b92b239096ff1ad49c1d4c0e65929",
        "bf16": "b0c57e6cda2e1a808457a7b53390b7b59c1c5d76833e5b1d7ad30274538d6b1a"}, 1472, 216),
])
@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_direct_mode_gpu_validated_stream_hashes(shape, expected, words, registers, dtype):
    code, _, encoded, _, count = tensor.generate(
        kernel(shape, dtype), "sm_80", "safe",
        config=tensor.TensorConfig(pointer_mode="direct"))
    assert hashlib.sha256(oracle.words_to_bytes(encoded)).hexdigest() == expected[dtype]
    assert len(encoded) == words and count == registers
    assert all(control_valid(control(word)) for word in encoded)
    assert all(ctl.stall == 8 for text, ctl in code if text.startswith("HMMA"))
    assert all(ctl.stall == 15 and ctl.yield_ for text, ctl in code if text == "NOP")


def test_direct_mode_removes_copy_homes_but_preserves_aligned_native_operands():
    outputs = {}
    for mode in ("legacy", "direct"):
        selector = tensor.TensorSelector(kernel(), tensor.TensorConfig(pointer_mode=mode))
        ops = ra.coalesce(ra.dce(selector.run()))
        phys = ra.allocate(ops, selector.r1)
        lowered = ra.lower(ops, phys)
        counts = Counter(text.split()[0] for text, *_ in lowered if text)
        outputs[mode] = counts
        if mode == "direct":
            transfers = [op for op in ops if op.fmt.startswith("LDGSTS")]
            assert len(transfers) == 64
            for op in transfers:
                pair = op.uses[1]
                assert pair in selector.homes.values()
                assert phys[pair[0]] % 2 == 0 and phys[pair[1]] == phys[pair[0]] + 1
    assert outputs["legacy"]["MOV"] == 282
    assert outputs["direct"]["MOV"] == 154
    for op in ("HMMA.16816.F32", "LDSM.16.M88.4", "LDGSTS.E.BYPASS.128", "BAR.SYNC.DEFER_BLOCKING"):
        assert outputs["legacy"][op] == outputs["direct"][op]


def test_direct_pointer_update_waits_for_async_address_consumption():
    code = tensor.tensor_schedule([
        Out("LDGSTS.E.BYPASS.128 [R2], [R4.64]", False, [2, 4, 5], []),
        Out("IADD3 R4, P0, R4, 0x80, RZ", False, [4], [4]),
        Out("IADD3.X R5, R5, RZ, RZ, P0, !PT", False, [5], [5]),
    ], {})
    transfer_barrier = code[0][1].rbar
    assert transfer_barrier != 7
    assert code[1][1].wait & (1 << transfer_barrier)


def test_pointer_mode_is_explicit_and_legacy_is_the_default():
    assert tensor.TensorConfig().pointer_mode == "legacy"
    assert tensor.policy()["pointer_mode"] == "legacy"
    assert tensor.policy(tensor.TensorConfig(pointer_mode="direct"))["pointer_mode"] == "direct"
    with pytest.raises(ValueError, match="pointer mode"):
        tensor.TensorConfig(pointer_mode="auto")
    with pytest.raises(TypeError, match="TensorConfig"):
        tensor.policy({"pointer_mode": "direct"})
