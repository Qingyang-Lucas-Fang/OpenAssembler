"""Validate state diagnostics on pinned PyTorch and the unchanged BF16 A100 case."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import modal

_SCRIPT = Path(__file__).resolve()
ROOT = _SCRIPT.parents[2] if len(_SCRIPT.parents) > 2 else Path.cwd()
APP = modal.App('openassembler-bf16-verification-guard')
IMAGE = (modal.Image.from_registry('pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel')
         .pip_install('pytest==8.4.2')
         .add_local_dir(str(ROOT / 'openassembler'), '/root/openassembler', copy=True,
                        ignore=['**/__pycache__', '**/*.pyc'])
         .add_local_file(str(ROOT / 'tests/test_torchfront_numerics.py'), '/root/tests/test_torchfront_numerics.py', copy=True)
         .add_local_file(str(ROOT / 'exp/weaknesses/bf16_verification.py'), '/root/exp/weaknesses/bf16_verification.py', copy=True)
         .add_local_file(str(ROOT / 'exp/whole_stack/model.py'), '/root/whole_stack/model.py', copy=True))


@APP.function(image=IMAGE, gpu='A100-40GB', timeout=600, max_containers=1, retries=0)
def execute():
    import contextlib
    import io
    import os
    import subprocess
    import sys
    import tempfile
    import time
    import torch

    sys.path.insert(0, '/root')
    from openassembler.torchfront.worker import StateVerificationError, _child, compare_states, runtime_info
    tests = subprocess.run([sys.executable, '-m', 'pytest', '-q', '/root/tests/test_torchfront_numerics.py'],
                           cwd='/root', env={**os.environ, 'PYTHONPATH': '/root'},
                           capture_output=True, text=True, timeout=60)
    if tests.returncode:
        raise RuntimeError(tests.stdout + tests.stderr)
    scratch = Path(tempfile.mkdtemp(prefix='oa-bf16-guard-'))
    os.environ.update(TORCHINDUCTOR_CACHE_DIR=str(scratch / 'inductor'),
                      TRITON_CACHE_DIR=str(scratch / 'triton'),
                      TORCHINDUCTOR_COMPILE_THREADS='1', PYTHONHASHSEED='0')
    cfg = {'entrypoint': 'model.py', 'args': ['--mode', 'train', '--batch', '8', '--seq', '128',
           '--width', '512', '--heads', '8', '--layers', '2', '--vocab', '512', '--dtype', 'bf16',
           '--steps', '20', '--warmup', '8', '--profile'], 'seed': 1729, 'mode': 'compile',
           'rtol': .05, 'atol': .005, 'state_rtol': .05, 'state_atol': .005, 'requirements': []}
    started = time.monotonic()
    with contextlib.redirect_stdout(io.StringIO()):
        eager, reference = _child(Path('/root/whole_stack'), cfg, scratch, 'eager', False, 180)
        compiled, actual = _child(Path('/root/whole_stack'), cfg, scratch, 'compiled', True, 360)
    expected = torch.load(io.BytesIO(reference), map_location='cpu', weights_only=True)
    got = torch.load(io.BytesIO(actual), map_location='cpu', weights_only=True)
    original_failures = []
    for key, tensor in expected.items():
        if isinstance(tensor, torch.Tensor):
            try:
                torch.testing.assert_close(got[key], tensor, rtol=.05, atol=.005, equal_nan=False)
            except AssertionError:
                original_failures.append(key)
        elif got[key] != tensor:
            original_failures.append(key)
    try:
        verification = compare_states(reference, actual, .05, .005, .05, .005)
    except StateVerificationError as exc:
        verification = exc.diagnostics
    assert bool(original_failures) == (verification['status'] == 'failed')
    if original_failures:
        assert verification['tensors_failed'] + verification['non_tensor_values_failed'] == len(original_failures)
    return {'schema': 1, 'runtime': runtime_info(), 'tests': tests.stdout.strip(),
            'seconds': time.monotonic() - started, 'config': cfg,
            'optimizer_steps': {'eager': eager['optimizer_steps'], 'compiled': compiled['optimizer_steps']},
            'original_assert_close_failed_keys': original_failures, 'verification': verification,
            'gradient_snapshot_semantics': 'retained .grad after application clip_grad_norm_; raw gradients are not observed',
            'scope': 'diagnostic validation only; unchanged BF16 math, AdamW epsilon/defaults and numerical admission gate; no speed claim'}


@APP.local_entrypoint()
def main(out: str = ''):
    result = execute.remote()
    result['timestamp_utc'] = datetime.now(timezone.utc).isoformat()
    output = Path(out) if out else Path('exp/weaknesses/bf16_verification_a100.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(output)
