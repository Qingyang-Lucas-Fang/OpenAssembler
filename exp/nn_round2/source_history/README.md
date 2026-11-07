# Exact experiment-source snapshots

This directory preserves source bytes for the November 7 NN experiment round.
It is a provenance archive, not another performance report. The current
experiment files remain in their ordinary locations and were not replaced.

Each `<sha256>.py.gz` file contains the exact UTF-8 bytes whose SHA256 appears
in its filename. Gzip timestamps are zero for deterministic archives. The
[snapshot index](snapshot-index.json) maps current files and each historical
hash recorded in a raw run to its archive, related result records and recovered
immutable Modal image IDs. Every archive is decompressed and rehashed before
it is admitted to that index.

## What was recovered

All eight recorded `whole_worker.py` versions and all three recorded
`counter_worker.py` versions are archived. The original operator worker and
frozen tensor validator are archived as well. This includes the worker sources
for successful whole-network runs, numerical failures, graph-capture failures
and the earlier counter parser. A failed run's source is not relabeled as a
passing implementation.

As of the final recovery check, **48 of 57 historical harness-source hashes**
recorded across the round's raw JSON files have exact archived bytes. The nine
remaining hashes belong to superseded local builders, analyses, launchers or
tests. Their names, hashes and referencing records are explicitly marked
`archived: false` in the index. They are not reconstructed by guesswork. The
complete historical GPU worker versions are available, but this is not a claim
that every earlier local orchestration file has been recovered.

The tool also snapshots all current top-level experiment `.py` files. That
current-file count can grow as documentation/verification helpers are added;
the generated index is authoritative.

## Recovery method and resource boundary

The installed Modal 1.6.2 client exposes `Image.from_id`. Image IDs were taken
from the actual retained launch logs. Recovery started short **CPU-only**
sandboxes using those already-built immutable images, with `gpu=None`,
0.25 requested CPU cores, 256 MiB requested memory, a sixty-second timeout and
outbound network blocked. The script read only named top-level experiment
source files, verified SHA256 locally, saved the compressed bytes and terminated
each sandbox. It did not import PyTorch, invoke a compiler, run a kernel or
allocate a GPU. It did not print credentials or environment variables.

Individual `recovery-im-*.json` records retain the exact image-to-source mapping,
resource intent and recovery timestamp. An image can contain a different
version from a locally recorded harness hash, so only **matching recovered
bytes** satisfy a requested historical version; proximity in time is not used
as evidence. Images may later expire; the verified local archives no longer
depend on their continued availability.

Two otherwise missing local follow-up versions were recovered from their known
recorded edits and accepted only after their full bytes matched the already
recorded SHA256. These are labeled separately in
[local edit recovery](local-edit-recovery.json), not attributed to an image.
One is the initial failed path-metadata builder; the other is the rejected
serialized Modal launcher. Neither was a successful GPU execution.

## Reproduction and verification

From the repository root, snapshot and verify current files without remote work:

```sh
python3 exp/nn_round2/recover_sources.py
```

Optional recovery requires the authenticated Modal client. It uses CPU
sandboxes only and reuses completed recovery records:

```sh
python3 exp/nn_round2/recover_sources.py --recover --workers 3 --max-images 24
python3 exp/nn_round2/recover_sources.py --recover --include-helpers \
  --image-id im-7LiIocjSDR2H5eGAty6kpl \
  --image-id im-CqWTIsJpx16983e5uaanGm \
  --image-id im-o2wmXUOdCi9QbQk0Q9Xz0o --workers 3
```

Inspect or extract a selected archive into a separate experiment location;
do not overwrite the current worker merely to inspect history. A standalone
verification of every content-addressed file is:

```sh
python3 - <<'PY'
import gzip, hashlib
from pathlib import Path
files = sorted(Path('exp/nn_round2/source_history').glob('*.py.gz'))
for path in files:
    data = gzip.decompress(path.read_bytes())
    assert hashlib.sha256(data).hexdigest() == path.name[:-6], path
print(f'{len(files)} exact source archives verified')
PY
```

[Recovery driver](../recover_sources.py) and
[shared evidence conventions](../../../docs/DEVELOPMENT.md) describe the
implementation and broader validity requirements. The archive adds no new
GPU performance measurement.
