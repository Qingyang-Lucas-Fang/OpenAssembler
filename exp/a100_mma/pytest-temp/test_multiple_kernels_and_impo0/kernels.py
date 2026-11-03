from openassembler.pyfront import kernel as gpu, Array, f32, tid_x
from pathlib import Path
Path('/Users/lucasfang/Desktop/OpenAssembler/exp/a100_mma/pytest-temp/test_multiple_kernels_and_impo0/must_not_exist').write_text("executed")
raise RuntimeError("never execute source")
@gpu
def first(a: Array(f32)):
    a[tid_x()] = 2.0
@gpu
def second(a: Array(f32)):
    a[tid_x()] = 3.0
