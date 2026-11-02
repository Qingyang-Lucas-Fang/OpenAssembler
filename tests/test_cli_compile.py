"""`openassembler compile`: writes a .cubin AND a .sass listing, from .ptx/.py/.cu,
with the output anywhere (not only inside the repo). Plus the pieces that made
that work on both macOS (toolkit in a colima VM) and Linux (toolkit native)."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from openassembler import oracle
from openassembler.cli import main

ROOT = Path(__file__).resolve().parent.parent
needs_toolkit = pytest.mark.skipif(not oracle.available("ptxas") or not oracle.available("nvdisasm"),
                                   reason=oracle.NEEDS_TOOLKIT)


@needs_toolkit
def test_compile_ptx_writes_cubin_and_sass(tmp_path, capsys):
    out = tmp_path / "k.cubin"
    assert main(["compile", str(ROOT / "examples" / "div_by_const.ptx"), "-o", str(out), "--verify"]) == 0
    sass = out.with_suffix(".sass")
    assert out.stat().st_size > 0 and sass.exists()
    text = sass.read_text()
    assert ".text.div_by_const" in text and "IMAD.HI" in text   # real nvdisasm output of our code
    assert "bit-exact" in capsys.readouterr().out


@needs_toolkit
def test_compile_python_source_and_custom_sass_path(tmp_path):
    out, sass = tmp_path / "v.cubin", tmp_path / "listing" / "v.sass"
    assert main(["compile", str(ROOT / "examples" / "vector_add.py"), "-o", str(out), "--sass", str(sass)]) == 0
    assert out.exists() and sass.exists() and "FADD" in sass.read_text()


@needs_toolkit
def test_compile_cuda_source_leaves_no_files_next_to_it(tmp_path):
    src = tmp_path / "k.cu"
    src.write_text((ROOT / "examples" / "vector_add.cu").read_text())
    out = tmp_path / "out" / "k.cubin"
    assert main(["compile", str(src), "-o", str(out)]) == 0
    assert out.exists() and out.with_suffix(".sass").exists()
    assert not src.with_suffix(".ptx").exists()        # nvcc's intermediate goes to scratch, not beside the source


@needs_toolkit
def test_no_sass_flag_skips_disassembly(tmp_path):
    out = tmp_path / "k.cubin"
    assert main(["compile", str(ROOT / "examples" / "div_by_const.ptx"), "-o", str(out), "--no-sass"]) == 0
    assert out.exists() and not out.with_suffix(".sass").exists()


def test_backend_rejection_is_a_clean_error_not_a_traceback(tmp_path, capsys):
    src = tmp_path / "bad.py"
    src.write_text("from openassembler.pyfront import Array, f32, i32, kernel\n\n@kernel\n"
                   "def bad(a: Array(f32), out: Array(f32), n: i32):\n    i = tid_x()\n"
                   "    if i < n:\n        out[i] = float(i) + a[i]\n")
    assert main(["compile", str(src), "-o", str(tmp_path / "b.cubin"), "--no-sass",
                 "--base-cubin", str(ROOT / "specs" / "does-not-matter")]) == 1
    err = capsys.readouterr().err
    assert "can't compile this kernel yet" in err and "cvt" in err


def test_tool_path_relative_inside_repo_absolute_outside(tmp_path):
    assert oracle.tool_path(ROOT / "examples" / "div_by_const.ptx") == "examples/div_by_const.ptx"
    outside = tmp_path / "x.cubin"
    if oracle.platform_is_macos():
        # the colima VM only mounts $HOME; pytest's tmp_path lives under /private/var on macOS
        with pytest.raises(oracle.OracleError, match="outside your home"):
            oracle.tool_path(outside)
    else:
        assert oracle.tool_path(outside) == str(outside.resolve())


def test_generic_verifier_aligns_pointer_after_a_4_byte_scalar():
    """(float a, float* x, ...): the pointer sits at offset 8, not 4. The harness used to
    concatenate without padding and the kernel read garbage (found compiling saxpy)."""
    from openassembler.verify import _align

    chunks = [struct.pack("<f", 2.0)]
    _align(chunks, 8)
    assert sum(len(c) for c in chunks) == 8


def test_encoder_loads_without_the_toolkit_thanks_to_the_fieldmap_cache(monkeypatch):
    """Encoder() used to call nvdisasm ~60 times at every start (a minute on macOS).
    The probe results are cached in specs/sm_80/synth_fieldmaps.json, so loading is
    instant and offline."""
    def boom(*a, **k):
        raise AssertionError("Encoder() must not need nvdisasm when the cache is complete")

    monkeypatch.setattr(oracle, "run_tool", boom)
    from openassembler.isa.encode import Encoder

    assert len(Encoder("sm_80").templates) > 400


@needs_toolkit
def test_failed_verify_removes_the_bad_cubin(tmp_path, monkeypatch):
    import openassembler.cli as cli

    monkeypatch.setattr(cli, "_verify_compiled", lambda *a, **k: 1)       # pretend --verify found a mismatch
    out = tmp_path / "k.cubin"
    assert main(["compile", str(ROOT / "examples" / "vector_add.py"), "-o", str(out), "--verify"]) == 1
    assert not out.exists() and not out.with_suffix(".sass").exists()


@needs_toolkit
def test_output_directory_and_sass_collision(tmp_path, capsys):
    d = tmp_path / "outdir"
    d.mkdir()
    assert main(["compile", str(ROOT / "examples" / "vector_add.py"), "-o", str(d)]) == 0
    assert (d / "vector_add.cubin").exists() and (d / "vector_add.sass").exists()
    capsys.readouterr()
    assert main(["compile", str(ROOT / "examples" / "vector_add.py"), "-o", str(tmp_path / "x.sass")]) == 1
    assert "both be written" in capsys.readouterr().err


def test_unknown_arch_is_a_clean_error(tmp_path, capsys):
    assert main(["compile", str(ROOT / "examples" / "vector_add.py"), "--arch", "sm_90", "-o", str(tmp_path / "a.cubin")]) == 1
    assert "no SASS encoder data for 'sm_90'" in capsys.readouterr().err
