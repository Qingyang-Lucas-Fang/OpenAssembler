"""CPU cohort-shape and actual custom-autograd derivative contract checks.

These do not substitute for full-state A100 admission. The derivative test
executes the Product class extracted from the real worker with tiny exact
matrices so non-square transpose mistakes fail before a remote launch.
"""
import ast
from contextlib import nullcontext
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from exp.nn_round2.whole_suite import CASES, GATE, shapes


def test_cohort_names_and_all_logical_training_products_are_launchable():
    names = [case["name"] for case in CASES]
    assert len(names) == len(set(names)) >= 5
    assert {case["family"] for case in CASES} >= {"mlp", "gated", "transformer"}
    available = set(shapes())
    for case in CASES:
        m, d, h = (case[key] for key in ("tokens", "width", "hidden"))
        assert min(m, d, h, case["layers"]) > 0
        linear_products = [(m, h, d), (m, d, h)]
        if case["family"] == "transformer":
            assert case["batch"] * case["seq"] == m
            assert d % case["heads"] == 0
            s, head = case["seq"], d // case["heads"]
            linear_products += [(m, 3*d, d), (m, d, d), (m, 256, d),
                                (s, s, head), (s, head, s)]
        for rows, outputs, inputs in linear_products:
            required = {(rows, outputs, inputs), (rows, inputs, outputs),
                        (outputs, inputs, rows)}
            assert required <= available, (case["name"], required - available)
    assert all(len(shape) == 3 and all(dim > 0 and dim % 128 == 0 for dim in shape)
               for shape in available)


def test_gates_cover_the_complete_snapshot_categories_without_relaxing():
    assert set(GATE) == {"output", "gradient", "parameter", "optimizer"}
    for name, gate in GATE.items():
        assert set(gate) == {"rtol", "atol"}
        assert math.isfinite(gate["rtol"]) and 0 <= gate["rtol"] <= 2e-3
        assert math.isfinite(gate["atol"]) and 0 <= gate["atol"] <= 2e-5


class Matrix:
    """Minimal exact arithmetic stand-in for the worker's FP16 operand handles."""
    def __init__(self, data):
        self.data = [list(row) for row in data]
        assert self.data and self.data[0]
        assert len({len(row) for row in self.data}) == 1
        self.shape = (len(self.data), len(self.data[0]))

    def to(self, _dtype):
        return self

    def contiguous(self):
        return self

    @property
    def T(self):
        return Matrix(zip(*self.data))


class Backend:
    def pack_weight(self, tensor):
        return tensor
    def __init__(self):
        self.calls = []

    def stage(self, _name):
        return nullcontext()

    def matmul(self, a, b, name):
        self.calls.append((name, a.shape, b.shape))
        assert a.shape[1] == b.shape[1]
        return Matrix([[sum(x*y for x, y in zip(row, col)) for col in b.data]
                       for row in a.data])


def worker_product_class():
    source = Path(__file__).with_name("whole_worker.py").read_text()
    parsed = ast.parse(source)
    run = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    product = next(node for node in run.body if isinstance(node, ast.ClassDef) and node.name == "Product")
    namespace = {"torch": SimpleNamespace(autograd=SimpleNamespace(Function=object), float16="f16")}
    module = ast.fix_missing_locations(ast.Module(body=[product], type_ignores=[]))
    exec(compile(module, "whole_worker.py:Product", "exec"), namespace)
    return namespace["Product"]


def loss_value(a, b, upstream):
    result = Backend().matmul(Matrix(a), Matrix(b), "loss")
    return sum(value*grad for row, grad_row in zip(result.data, upstream)
               for value, grad in zip(row, grad_row))


def test_actual_product_backward_matches_finite_difference_for_nonsquare_operands():
    Product = worker_product_class()
    a = [[1., -2.], [.5, 3.], [-1., .25]]
    b = [[2., 1.], [-3., .5], [0., 4.], [.125, -.5], [1., 1.]]
    upstream = [[.25*(i+1)-.1*j for j in range(5)] for i in range(3)]
    backend = Backend()
    ctx = SimpleNamespace()
    ctx.save_for_backward = lambda *values: setattr(ctx, "saved_tensors", values)
    out = Product.forward(ctx, Matrix(a), Matrix(b), backend, "nonsquare")
    assert out.shape == (3, 5)
    da, db, dbkend, dname = Product.backward(ctx, Matrix(upstream))
    assert da.shape == (3, 2) and db.shape == (5, 2)
    assert dbkend is dname is None
    assert backend.calls == [
        ("nonsquare/gemm_forward", (3, 2), (5, 2)),
        ("nonsquare/gemm_dx", (3, 5), (2, 5)),
        ("nonsquare/gemm_dw", (5, 3), (2, 3)),
    ]
    step = 1e-5
    for operand, actual in ((a, da.data), (b, db.data)):
        for row in range(len(operand)):
            for col in range(len(operand[0])):
                old = operand[row][col]
                operand[row][col] = old + step
                plus = loss_value(a, b, upstream)
                operand[row][col] = old - step
                minus = loss_value(a, b, upstream)
                operand[row][col] = old
                assert actual[row][col] == pytest.approx((plus-minus)/(2*step), rel=1e-8, abs=1e-8)
