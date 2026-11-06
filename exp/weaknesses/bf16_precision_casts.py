"""A100 test of eager cast preservation; unchanged BF16 training/optimizer gates."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import modal

_SCRIPT = Path(__file__).resolve()
ROOT = _SCRIPT.parents[2] if len(_SCRIPT.parents) > 2 else Path.cwd()
APP = modal.App('openassembler-bf16-precision-casts')
IMAGE = (modal.Image.from_registry('pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel')
         .add_local_dir(str(ROOT / 'openassembler'), '/root/openassembler', copy=True,
                        ignore=['**/__pycache__', '**/*.pyc'])
         .add_local_file(str(ROOT / 'exp/whole_stack/model.py'), '/root/whole_stack/model.py', copy=True))


@APP.function(image=IMAGE, gpu='A100-40GB', timeout=900, max_containers=1, retries=0)
def execute(reverse: bool = False):
    import contextlib
    import io
    import os
    import statistics
    import sys
    import tempfile
    import time
    import torch

    sys.path.insert(0, '/root')
    from openassembler.torchfront.worker import StateVerificationError, _child, compare_states, runtime_info
    scratch = Path(tempfile.mkdtemp(prefix='oa-bf16-cast-'))
    cfg = {'entrypoint': 'model.py', 'args': ['--mode', 'train', '--batch', '8', '--seq', '128',
           '--width', '512', '--heads', '8', '--layers', '2', '--vocab', '512', '--dtype', 'bf16',
           '--steps', '20', '--warmup', '8', '--profile'], 'seed': 1729, 'mode': 'compile',
           'rtol': .05, 'atol': .005, 'state_rtol': .05, 'state_atol': .005, 'requirements': []}
    started = time.monotonic()
    records = []
    for preserve in ((True, False) if reverse else (False, True)):
        workspace = scratch / str(int(preserve))
        workspace.mkdir()
        os.environ.update(TORCHINDUCTOR_CACHE_DIR=str(workspace / 'inductor'),
                          TRITON_CACHE_DIR=str(workspace / 'triton'),
                          TORCHINDUCTOR_COMPILE_THREADS='1', PYTHONHASHSEED='0',
                          TORCHINDUCTOR_EMULATE_PRECISION_CASTS=str(int(preserve)))
        with contextlib.redirect_stdout(io.StringIO()):
            eager, reference = _child(Path('/root/whole_stack'), cfg, workspace, 'eager', False, 180)
            compiled, actual = _child(Path('/root/whole_stack'), cfg, workspace, 'compiled', True, 360)
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
        stage_timings = {}
        for method, report in (('eager', eager), ('compiled', compiled)):
            timing = report.get('timing', {})
            # The model's separate profiler step is not part of measured timing.
            stage_timings[method] = {key: value for key, value in timing.items() if key != 'profiler'}
        records.append({'emulate_precision_casts': preserve, 'verification': verification,
                        'original_assert_close_failed_keys': original_failures,
                        'optimizer_steps': {'eager': eager['optimizer_steps'], 'compiled': compiled['optimizer_steps']},
                        'timings': stage_timings,
                        'compiled_graphs': len(compiled.get('graphs', [])),
                        'eager_state_sha256': hashlib.sha256(reference).hexdigest(),
                        'compiled_state_sha256': hashlib.sha256(actual).hexdigest()})
        print(json.dumps({'preserve': preserve, 'verification': verification['status'],
                          'failed_keys': original_failures}), flush=True)
    return {'schema': 1, 'runtime': runtime_info(), 'config': cfg, 'records': records,
            'process_order': 'true then false' if reverse else 'false then true',
            'seconds': time.monotonic() - started,
            'worker_sha256': hashlib.sha256(Path('/root/openassembler/torchfront/worker.py').read_bytes()).hexdigest(),
            'model_sha256': hashlib.sha256(Path('/root/whole_stack/model.py').read_bytes()).hexdigest(),
            'primary_source': 'https://github.com/pytorch/pytorch/blob/v2.7.1/torch/_inductor/config.py#L531-L541',
            'scope': 'fresh seeded eager/compiled subprocesses; separate caches per setting; unchanged29updateBF16model,AdamWandgates; prototypeonly'}


@APP.local_entrypoint()
def main(reverse: bool = False):
    result = execute.remote(reverse)
    result['timestamp_utc'] = datetime.now(timezone.utc).isoformat()
    result['harness_sha256'] = hashlib.sha256(_SCRIPT.read_bytes()).hexdigest()
    output = ROOT / 'exp/weaknesses' / ('bf16_precision_casts_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '.json')
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(output)
