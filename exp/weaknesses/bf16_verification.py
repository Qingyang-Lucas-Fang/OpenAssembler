"""CPU fixtures and bounded A100 checks for fail-closed state diagnostics.

This does not change model arithmetic, optimizer policies or tolerances.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def fixtures(compare, error_type):
    import torch
    from openassembler.torchfront.worker import _save_state

    expected = {
        'outputs/model/0': torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
        'parameters/model/qkv.bias': torch.zeros(1536),
        'gradients/model/qkv.bias': torch.full((1536,), 1e-8),
        'optimizer/0/state/exp_avg': torch.zeros(1536),
        'optimizer/0/steps': 29,
    }
    got = {name: value.clone() if isinstance(value, torch.Tensor) else value
           for name, value in expected.items()}
    got['parameters/model/qkv.bias'][512:516] = 0.007
    got['gradients/model/qkv.bias'][512:516] = -1e-8
    got['optimizer/0/state/exp_avg'][0] = 0.007
    try:
        compare(_save_state(expected), _save_state(got), .05, .005, .05, .005)
    except error_type as exc:
        report = exc.diagnostics
        assert report['status'] == 'failed'
        assert report['tensors_checked'] == 4
        assert report['tensors_failed'] == 2
        assert report['groups']['parameters']['tensors_failed'] == 1
        assert report['groups']['optimizer']['tensors_failed'] == 1
        assert report['groups']['gradients']['tensors_failed'] == 0
        assert report['failures'][0]['mismatched_elements'] == 4
        assert report['failures'][0]['mismatched_index_sample'] == [[512], [513], [514], [515]]
        assert report['failures'][0]['reference_l2'] == 0.0
        assert report['failures'][0]['relative_l2_difference'] is None
        assert report['failures'][0]['absolute_tolerance'] == .005
        json.dumps(report, allow_nan=False)
    else:
        raise AssertionError('original BF16-like parameter failure was incorrectly admitted')
    clean = compare(_save_state(expected), _save_state(expected), .05, .005, .05, .005)
    assert clean['status'] == 'ok' and clean['tensors_checked'] == 4
    return report


def main():
    import sys
    import torch
    from openassembler.torchfront.worker import StateVerificationError, compare_states
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = {'schema': 1, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'runtime': {'python': sys.version.split()[0], 'torch': torch.__version__},
              'scope': 'CPU diagnostic fixtures; no GPU performance or training-equivalence claim',
              'diagnostic': fixtures(compare_states, StateVerificationError)}
    rendered = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
    print(rendered)


if __name__ == '__main__':
    main()
