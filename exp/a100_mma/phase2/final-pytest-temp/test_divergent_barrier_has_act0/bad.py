from openassembler.pyfront import *
@kernel
def bad(a: Array(f32)):
    if tid_x() < 3:
        syncthreads()
