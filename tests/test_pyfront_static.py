"""Compilation discovers functions statically without executing a user's module."""
from pathlib import Path
import pytest
from openassembler.pyfront import compile_file, NotSupported
from openassembler.frontend import to_ptx_module
from openassembler.ptx.parse import parse_module


def test_multiple_kernels_and_import_aliases_without_top_level_execution(tmp_path):
    marker = tmp_path / 'must_not_exist'
    p = tmp_path / 'kernels.py'
    p.write_text(f'''from openassembler.pyfront import kernel as gpu, Array, f32, tid_x
from pathlib import Path
Path({str(marker)!r}).write_text("executed")
raise RuntimeError("never execute source")
@gpu
def first(a: Array(f32)):
    a[tid_x()] = 2.0
@gpu
def second(a: Array(f32)):
    a[tid_x()] = 3.0
''')
    ptx, names, _ = to_ptx_module(p, arch='sm_80')
    assert names == ['first', 'second']
    assert set(parse_module(ptx).kernels) == set(names)
    assert not marker.exists()


def test_divergent_barrier_has_actionable_error(tmp_path):
    p = tmp_path / 'bad.py'
    p.write_text('''from openassembler.pyfront import *
@kernel
def bad(a: Array(f32)):
    if tid_x() < 3:
        syncthreads()
''')
    with pytest.raises(NotSupported, match='barrier|syncthreads|divergen'):
        compile_file(p)
