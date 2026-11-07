"""CPU-only native compiler on Modal; execution uses the existing A100 runner."""
from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App('openassembler-native-build')
image = modal.Image.from_registry('nvidia/cuda:12.8.1-devel-ubuntu22.04', add_python='3.12')


@app.function(image=image, timeout=1860, max_containers=1, retries=0)
def build(request_text: str, runtime: dict[str, bytes]) -> str:
    import os
    import subprocess
    import sys
    import tempfile
    with tempfile.TemporaryDirectory(prefix='oa-compiler-') as d:
        for name, data in runtime.items():
            path = Path(d) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        (Path(d) / 'env/cuda.sh').chmod(0o700)
        request, result = Path(d) / 'request.json', Path(d) / 'result.json'
        request.write_text(request_text)
        # Fresh interpreter per request prevents cached oracle roots/encoder data
        # from a prior invocation surviving a warm Modal container.
        subprocess.run([sys.executable, '-m', 'openassembler.remote_build',
                        '--request', str(request), '--result', str(result)],
                       cwd=d, env=dict(os.environ, PYTHONPATH=d), check=True)
        return result.read_text()


@app.local_entrypoint()
def main(request: str, result: str, budget: float = 900):
    import math
    from openassembler.transport import runtime_files
    response = build.with_options(timeout=math.ceil(budget) + 60).remote(
        Path(request).read_text(), runtime_files(native=True))
    Path(result).write_text(response)
