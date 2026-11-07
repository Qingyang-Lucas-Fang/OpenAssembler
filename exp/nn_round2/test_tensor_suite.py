"""CPU gates for the experimental tensor candidate compiler."""
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).with_name("tensor_suite.py")
SPEC = importlib.util.spec_from_file_location("nn_round2_tensor", PATH)
suite = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(suite)


def test_training_matrix_covers_forward_dx_dw_both_precisions():
    matrix = suite.definitions("full")
    assert len(matrix) == 18
    assert {row["dtype"] for row in matrix} == {"f16", "bf16"}
    for size in ("small", "wide"):
        for role in ("forward", "dx", "dw"):
            assert any(f"mlp_{size}_{role}" in row["name"] for row in matrix)
    for row in matrix:
        for candidate in row["candidates"]:
            if candidate["order"] == "morton8":
                assert row["shape"][0] % 1024 == row["shape"][1] % 1024 == 0
            if "_dx_" in row["name"] or "_dw_" in row["name"]:
                assert candidate["epilogue"] == "none"


def test_combination_is_explicit_and_keeps_proven_tensor_policy():
    row = next(row for row in suite.definitions("full", ("f16",)) if row["name"] == "mlp_wide_forward_f16")
    for candidate in row["candidates"]:
        source, config, plan = suite.source_and_config(row, candidate)
        assert config.issue_gap == 8 and config.epoch_drain == 15 and config.reuse
        assert config.pointer_mode == candidate["pointer"]
        assert plan["pointer_mode"] == "legacy"
        assert plan["cta_order"] == candidate["order"]
        assert not source.startswith("// openassembler-gemm-v1")


def test_experiment_does_not_relax_production_contract():
    from openassembler.kernels.gemm import GemmPlan
    with pytest.raises(ValueError, match="direct pointer"):
        GemmPlan(1024, 1024, 1024, cta_order="morton8", pointer_mode="direct")
    with pytest.raises(ValueError, match="direct pointer"):
        GemmPlan(1024, 1024, 1024, epilogue="bias_residual_relu", pointer_mode="direct")


@pytest.mark.parametrize("pointer", ["legacy", "direct"])
def test_same_plan_matches_current_native_backend(pointer):
    from openassembler.backend.tensor import TensorConfig, generate
    from openassembler.kernels.gemm import GemmPlan, generate as source_generate
    from openassembler.ptx.parse import parse_module
    row = suite.definitions("smoke", ("f16",))[0]
    candidate = next(c for c in row["candidates"] if c["name"] == "oa_" + pointer)
    source, config, _ = suite.source_and_config(row, candidate)
    actual = generate(parse_module(source).kernels["gemm"], "sm_80", "opt", config=config)
    expected = generate(parse_module(source_generate(GemmPlan(128, 128, 128))).kernels["gemm"],
                        "sm_80", "opt", config=TensorConfig(pointer_mode=pointer))
    assert actual[2:] == expected[2:]


def test_experimental_compositions_encode_with_unchanged_hmma_work():
    from openassembler.backend.tensor import generate
    from openassembler.ptx.parse import parse_module
    row = next(row for row in suite.definitions("full", ("f16",)) if row["name"] == "mlp_wide_forward_f16")
    counts = {}
    for candidate in row["candidates"]:
        source, config, _ = suite.source_and_config(row, candidate)
        code, _, words, _, registers = generate(parse_module(source).kernels["gemm"], "sm_80", "opt", config=config)
        counts[candidate["name"]] = sum(text.startswith("HMMA") for text, _ in code)
        assert registers <= 255
        assert all(ctl.stall == 8 for text, ctl in code if text.startswith("HMMA"))
        assert all(ctl.stall == 15 and ctl.yield_ for text, ctl in code if text == "NOP")
        assert len(words) % 8 == 0
    assert set(counts.values()) == {512}


def test_invalid_profiles_and_output_escape_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="profile"):
        suite.definitions("unknown")
    with pytest.raises(ValueError, match="dtypes"):
        suite.definitions(dtypes=("fp8",))
    with pytest.raises(ValueError, match="remain under"):
        suite.build_payload("smoke", ("f16",), build_dir=tmp_path)
