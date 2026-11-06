"""State-verification diagnostics retain the original elementwise admission gate."""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip('torch')

from openassembler.torchfront.worker import _save_state
from openassembler.torchfront.worker import StateVerificationError, compare_states


def compare(reference, actual, *, rtol=.05, atol=.005, state_rtol=.05, state_atol=.005):
    return compare_states(_save_state(reference), _save_state(actual), rtol, atol, state_rtol, state_atol)


def test_bf16_like_bias_failure_does_not_hide_second_tensor_group():
    from exp.weaknesses.bf16_verification import fixtures
    result = fixtures(compare_states, StateVerificationError)
    assert 'raw versus clipped is not inferred' in result['gradient_scope']


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16, torch.int64, torch.complex64, torch.bool])
def test_failure_diagnostic_matches_torch_elementwise_gate(dtype):
    expected = torch.tensor([1, 2, 0, 1], dtype=dtype)
    got = torch.tensor([1, 0, 1, 0], dtype=dtype)
    with pytest.raises(StateVerificationError) as caught:
        compare({'outputs/model/0': expected}, {'outputs/model/0': got})
    detail = caught.value.diagnostics['failures'][0]
    assert detail['mismatched_elements'] == int((~torch.isclose(got, expected, rtol=.05, atol=.005)).sum())
    assert detail['mismatched_elements'] == 3
    assert detail['mismatched_index_sample'] == [[1], [2], [3]]
    json.dumps(caught.value.diagnostics, allow_nan=False)


def test_parameter_and_output_tolerances_are_still_separate():
    reference = {f'{group}/model': torch.zeros(1) for group in ('outputs', 'gradients', 'parameters', 'optimizer')}
    actual = {key: value + .02 for key, value in reference.items()}
    with pytest.raises(StateVerificationError) as caught:
        compare(reference, actual, rtol=0, atol=.01, state_rtol=0, state_atol=.03)
    report = caught.value.diagnostics
    assert report['tensors_checked'] == 4 and report['tensors_failed'] == 2
    assert [item['key'] for item in report['failures']] == ['outputs/model', 'gradients/model']
    assert all(item['absolute_tolerance'] == .01 for item in report['failures'])


@pytest.mark.parametrize('actual', [torch.zeros(2), torch.zeros(1, dtype=torch.float64), 'not a tensor'])
def test_metadata_and_type_mismatches_still_fail(actual):
    with pytest.raises(StateVerificationError) as caught:
        compare({'parameters/x': torch.zeros(1)}, {'parameters/x': actual})
    assert caught.value.diagnostics['tensors_failed'] == 1
    assert caught.value.diagnostics['failures'][0]['kind'] in ('metadata', 'type')


def test_matching_infinities_are_allowed_but_nans_are_not():
    expected = torch.tensor([float('inf'), float('-inf'), float('nan'), 1.])
    actual = torch.tensor([float('inf'), float('-inf'), float('nan'), 2.])
    with pytest.raises(StateVerificationError) as caught:
        compare({'outputs/x': expected}, {'outputs/x': actual})
    detail = caught.value.diagnostics['failures'][0]
    assert detail['mismatched_elements'] == 2
    assert detail['reference_nonfinite_elements'] == detail['actual_nonfinite_elements'] == 3
    assert detail['finite_pair_elements'] == 1 and detail['max_abs_difference'] == 1
    json.dumps(caught.value.diagnostics, allow_nan=False)
    assert compare({'outputs/x': expected[:2]}, {'outputs/x': actual[:2]})['status'] == 'ok'


def test_noncontiguous_tensor_index_samples_use_logical_shape():
    expected = torch.zeros(3, 2).T
    actual = expected.clone()
    actual[1, 2] = 1
    with pytest.raises(StateVerificationError) as caught:
        compare({'outputs/x': expected}, {'outputs/x': actual})
    assert caught.value.diagnostics['failures'][0]['mismatched_index_sample'] == [[1, 2]]


def test_diagnostic_details_are_bounded_but_all_failures_are_counted():
    expected = {f'parameters/x{i}': torch.zeros(1) for i in range(12)}
    actual = {key: torch.ones(1) for key in expected}
    expected['optimizer/steps'], actual['optimizer/steps'] = 29, 28
    with pytest.raises(StateVerificationError, match='verification failed at parameters/x0') as caught:
        compare(expected, actual)
    report = caught.value.diagnostics
    assert report['tensors_checked'] == report['tensors_failed'] == 12
    assert report['non_tensor_values_failed'] == 1
    assert len(report['failures']) == 8 and report['failure_details_omitted'] == 5


def test_missing_state_keys_and_empty_tensor_scope_still_fail():
    with pytest.raises(StateVerificationError, match='different model or optimizer states') as caught:
        compare({'outputs/x': torch.zeros(1)}, {'outputs/y': torch.zeros(1)})
    assert caught.value.diagnostics['missing_keys'] == ['outputs/x']
    assert caught.value.diagnostics['unexpected_keys'] == ['outputs/y']
    with pytest.raises(RuntimeError, match='no model tensors'):
        compare({'optimizer/steps': 2}, {'optimizer/steps': 2})


def test_success_does_not_mutate_saved_tensors_or_require_nonempty_values():
    state = {'outputs/x': torch.empty(0), 'parameters/x': torch.tensor([1., 2.]), 'optimizer/steps': 2}
    before = {key: value.clone() for key, value in state.items() if isinstance(value, torch.Tensor)}
    result = compare(state, state)
    assert result['status'] == 'ok' and result['tensors_checked'] == 2
    assert result['groups']['outputs']['tensors_checked'] == 1
    for key, value in before.items():
        torch.testing.assert_close(state[key], value, rtol=0, atol=0)


def test_diagnostic_errors_cannot_turn_mismatches_into_success(monkeypatch):
    import openassembler.torchfront.worker as worker
    def unavailable(*_args):
        raise ArithmeticError('diagnostic failed')
    monkeypatch.setattr(worker, '_tensor_diagnostic', unavailable)
    with pytest.raises(StateVerificationError) as caught:
        compare({'parameters/x': torch.zeros(1)}, {'parameters/x': torch.ones(1)})
    assert caught.value.diagnostics['failures'][0]['kind'] == 'statistics-unavailable'
    assert caught.value.diagnostics['tensors_failed'] == 1


def test_thousands_of_arbitrary_names_have_a_bounded_error_response():
    expected = {f'custom_{i}/' + 'x' * 512: torch.zeros(1) for i in range(1024)}
    actual = {key: torch.ones(1) for key in expected}
    with pytest.raises(StateVerificationError) as caught:
        compare(expected, actual)
    report = caught.value.diagnostics
    assert report['tensors_checked'] == report['tensors_failed'] == 1024
    assert set(report['groups']) == {'other'}
    assert all(item['key_truncated'] for item in report['failures'])
    assert len(json.dumps(report, sort_keys=True)) <= 12000
    assert len(str(caught.value)) < 14000


def test_worker_error_json_retains_diagnostics_and_publishes_no_artifacts(tmp_path, monkeypatch):
    import sys
    import openassembler.torchfront.worker as worker
    request, output = tmp_path / 'request.json', tmp_path / 'result.json'
    request.write_text('{}')
    def failed_process(_request):
        compare({'parameters/x': torch.zeros(1)}, {'parameters/x': torch.ones(1)})
    monkeypatch.setattr(worker, 'process', failed_process)
    monkeypatch.setattr(sys, 'argv', ['worker', '--request', str(request), '--result', str(output)])
    assert worker.main() == 1
    result = json.loads(output.read_text())
    assert 'verification failed at parameters/x' in result['error']
    assert result['verification']['tensors_failed'] == 1
    assert 'report' not in result and 'artifacts' not in result
