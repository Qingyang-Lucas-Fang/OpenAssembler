"""CPU-only checks for the prospective campaign coverage/admission contract."""
import json
from pathlib import Path


def load_coverage():
    return json.loads(Path(__file__).with_name("coverage.json").read_text())


def test_coverage_is_explicitly_prospective_and_covers_training_roles():
    data = load_coverage()
    assert data["schema"] == 1
    assert data["status"] == "planned"
    categories = data["categories"]
    ids = [row["id"] for row in categories]
    assert len(ids) == len(set(ids))
    assert set(ids) >= {
        "dense_linear", "batched_attention_products", "residual_bias_scale",
        "relu", "gelu", "silu_gate", "softmax", "layernorm", "rmsnorm",
        "bias_gradient", "loss_mse", "loss_cross_entropy", "gradient_clipping",
        "adamw", "elementary_math", "integer_addressing", "layout_and_cast",
        "masking_position", "embedding_gather_scatter", "dropout_rng",
        "convolution_pooling", "distributed_collectives",
    }
    for row in categories:
        for key in ("arithmetic", "training_roles", "native_status", "candidate",
                    "gate", "integration_costs"):
            assert row[key], (row["id"], key)
    assert {role for row in categories for role in row["training_roles"]} == {
        "forward", "backward", "input_gradient", "weight_gradient", "optimizer"
    }


def test_comparisons_and_admission_do_not_conflate_compilers():
    data = load_coverage()
    assert set(data["measurement_scopes"]) == {
        "native_kernel", "vendor_operator", "hybrid_workload", "compiled_application"
    }
    native = data["measurement_scopes"]["native_kernel"]
    assert "identical generated PTX" in native
    application = data["measurement_scopes"]["compiled_application"]
    assert "stock compiled PyTorch" in application
    assert "never attribute" in application
    admission = data["admission"]
    assert "held-out" in admission["candidate_selection"]
    assert "No timing admission" in admission["failure"]
    assert any("poisoned after" in item for item in admission["kernel_checks"])
    assert any("raw parameter gradients" in item for item in admission["workload_checks"])
    assert any("optimizer tensors" in item for item in admission["workload_checks"])
    assert any("stage instrumentation separate" in item for item in admission["timing"])
