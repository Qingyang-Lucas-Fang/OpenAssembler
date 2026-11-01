"""Dispatches `openassembler compile` to the right frontend by file extension, so
the backend (openassembler.backend.ra) always starts from PTX text regardless of
whether the source was already PTX, a Python kernel (openassembler.pyfront), or
CUDA C/C++ (openassembler.cppfront, which shells out to nvcc once -- see its
docstring for why that's not "cheating" the same way ptxas's code generation
would be).
"""

from __future__ import annotations

from pathlib import Path

PTX_EXTS = (".ptx",)
PY_EXTS = (".py",)
CU_EXTS = (".cu", ".cuh")
CPP_EXTS = (".cpp", ".cc", ".cxx")


class FrontendError(Exception):
    pass


def to_ptx(path: Path, *, entry: str | None, arch: str) -> tuple[str, str]:
    """Returns (ptx_text, kernel_name). `entry` selects which kernel/function
    to compile when the source has more than one; for .ptx with exactly one
    kernel it's optional, same as before this function existed."""
    suf = path.suffix.lower()
    if suf in PTX_EXTS:
        text = path.read_text()
        if entry:
            return text, entry
        # Caller resolves the single-kernel case itself (needs parse_module,
        # which lives in openassembler.ptx -- avoid importing it here just for that).
        return text, None  # type: ignore[return-value]

    if suf in PY_EXTS:
        from openassembler.pyfront.compile import find_kernel

        fn = find_kernel(path, entry)
        from openassembler.pyfront.compile import compile_kernel as py_compile

        return py_compile(fn, arch=arch), fn.__name__

    if suf in CU_EXTS or suf in CPP_EXTS:
        from openassembler.cppfront import compile_to_ptx

        return compile_to_ptx(path, entry=entry, arch=arch)

    raise FrontendError(f"don't know how to compile {path.suffix!r} files "
                         f"(supported: {PTX_EXTS + PY_EXTS + CU_EXTS + CPP_EXTS})")
