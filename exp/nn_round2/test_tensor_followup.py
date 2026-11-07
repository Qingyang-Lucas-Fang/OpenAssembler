import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).with_name("tensor_followup.py")
SPEC = importlib.util.spec_from_file_location("nn_round2_tensor_followup", PATH)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


@pytest.mark.parametrize("k", [128, 1024, 4096])
def test_relocation_preserves_every_ptx_operation_and_operand(k):
    from openassembler.kernels.gemm import GemmPlan, generate
    original = generate(GemmPlan(1024, 4096, k))
    moved = module.relocate_advances(original, "f16", k)
    assert original != moved
    assert module.instruction_multiset(original) == module.instruction_multiset(moved)
    assert moved.count("bar.sync 0") == original.count("bar.sync 0")
    assert moved.count("cp.async") == original.count("cp.async")


@pytest.mark.parametrize("pointer", ["legacy", "direct"])
def test_relocated_tensor_stream_keeps_policy_and_hmma_count(pointer):
    from openassembler.backend.tensor import TensorConfig, generate
    from openassembler.kernels.gemm import GemmPlan, generate as source
    from openassembler.ptx.parse import parse_module
    original = source(GemmPlan(1024, 4096, 1024))
    moved = module.relocate_advances(original, "f16", 1024)
    streams = []
    for text in (original, moved):
        code, _, words, _, registers = generate(parse_module(text).kernels["gemm"], "sm_80", "opt",
                                                config=TensorConfig(pointer_mode=pointer))
        assert registers <= 255
        assert all(ctl.stall == 8 for op, ctl in code if op.startswith("HMMA"))
        assert all(ctl.stall == 15 and ctl.yield_ for op, ctl in code if op == "NOP")
        assert sum(op.startswith("HMMA") for op, _ in code) == 512
        streams.append(words)
    assert streams[0] != streams[1]


def test_malformed_pipeline_fails_closed():
    with pytest.raises(ValueError, match="unexpected stage-pair"):
        module.relocate_advances("not the expected pipeline", "f16", 1024)
