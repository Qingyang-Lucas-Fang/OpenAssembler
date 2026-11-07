"""Preserve experiment sources and recover recorded versions from Modal images.

Local snapshots use deterministic content-addressed gzip files. Optional image
recovery starts only CPU sandboxes with GPU=None and network blocked, reads
only named experiment source files, verifies SHA256 and immediately terminates
the sandbox. It does not execute GPU experiments or modify current workers.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "exp/nn_round2"
HISTORY = HERE / "source_history"
NAMES = ("whole_worker.py", "counter_worker.py", "operator_worker.py", "tensor_suite.py", "campaign.py")


def archive(data):
    h = hashlib.sha256(data).hexdigest()
    path = HISTORY / (h + ".py.gz")
    if not path.exists():
        path.write_bytes(gzip.compress(data, mtime=0))
    if gzip.decompress(path.read_bytes()) != data:
        raise ValueError(f"source archive does not match its content address: {h}")
    return h, str(path.relative_to(ROOT))


def snapshot():
    HISTORY.mkdir(exist_ok=True)
    rows = []
    for path in sorted(HERE.glob("*.py")):
        data = path.read_bytes()
        h, destination = archive(data)
        rows.append({"source_path": str(path.relative_to(ROOT)), "sha256": h,
                     "bytes": len(data), "archive": destination, "origin": "local_current_snapshot"})
    wanted = {}
    for path in sorted((HERE / "runs").glob("*.json")):
        try:
            record = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        hashes = record.get("harness_sha256", {})
        if not isinstance(hashes, dict):
            continue
        for filename, h in hashes.items():
            if not filename.endswith(".py") or not isinstance(h, str):
                continue
            row = wanted.setdefault(h, {"source_paths": [], "records": []})
            if filename not in row["source_paths"]:
                row["source_paths"].append(filename)
            row["records"].append(str(path.relative_to(ROOT)))
    recoveries = []
    for path in sorted(HISTORY.glob("recovery-im-*.json")):
        recoveries.append(json.loads(path.read_text()))
    for h, row in wanted.items():
        archive_path = HISTORY / (h + ".py.gz")
        row["archived"] = archive_path.exists()
        if row["archived"]:
            data = gzip.decompress(archive_path.read_bytes())
            if hashlib.sha256(data).hexdigest() != h:
                raise AssertionError(f"invalid content-addressed archive {h}")
            row["archive"] = str(archive_path.relative_to(ROOT))
        row["recovered_from_images"] = sorted({recovery["image_id"] for recovery in recoveries
                                               for item in recovery.get("files", []) if item["sha256"] == h})
    result = {"schema": 1, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "snapshots": rows,
              "requested_recorded_sources": wanted, "image_recoveries": recoveries,
              "cpu_only": True, "new_gpu_measurement": False}
    (HISTORY / "snapshot-index.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def recover_image(image_id, include_helpers=False):
    import modal
    destination = HISTORY / ("recovery-" + image_id + ("-helpers" if include_helpers else "") + ".json")
    if destination.exists():
        return json.loads(destination.read_text())
    app = modal.App.lookup("openassembler-nn-round2-source-recovery", create_if_missing=True)
    image = modal.Image.from_id(image_id)
    filenames = tuple(sorted(set(NAMES) | {p.name for p in HERE.glob("*.py")})) if include_helpers else NAMES
    remote = ("import pathlib,base64,hashlib,json\nrows=[]\n"
              f"for name in {filenames!r}:\n"
              " p=pathlib.Path('/root/exp/nn_round2')/name\n"
              " if p.is_file():\n"
              "  data=p.read_bytes();rows.append({'path':str(p),'sha256':hashlib.sha256(data).hexdigest(),"
              "'bytes':len(data),'base64':base64.b64encode(data).decode()})\n"
              "print('OA_SOURCES='+json.dumps(rows))\n")
    sandbox = None
    result = {"schema": 1, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "image_id": image_id,
              "cpu_only": True, "gpu_requested": None, "network_blocked": True,
              "requested_filenames": list(filenames), "files": []}
    try:
        sandbox = modal.Sandbox.create("python", "-c", remote, app=app, image=image,
                                       timeout=60, cpu=0.25, memory=256, gpu=None, block_network=True)
        stdout, stderr = sandbox.stdout.read(), sandbox.stderr.read()
        sandbox.wait()
        marker = "OA_SOURCES="
        if marker not in stdout:
            raise RuntimeError("source result marker absent; stderr tail: " + stderr[-500:])
        rows = json.loads(stdout.rsplit(marker, 1)[1].splitlines()[0])
        for row in rows:
            data = base64.b64decode(row.pop("base64"))
            h, archive_path = archive(data)
            if h != row["sha256"]:
                raise ValueError("image response source hash mismatch")
            result["files"].append({**row, "archive": archive_path,
                                    "origin": "modal_immutable_image_cpu_read", "image_id": image_id})
        result["status"] = "completed"
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        if sandbox is not None:
            sandbox.terminate()
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        destination.write_text(json.dumps(result, indent=2) + "\n")
    return result


def image_ids_from_logs():
    images = []
    for path in sorted((HERE / "runs").glob("*.log")):
        ids = re.findall(r"Built image (im-[A-Za-z0-9]+)", path.read_text())
        # Final built layer of a run includes all preceding COPY layers. An
        # intermediate image may still be queried explicitly with --image-id.
        if ids and ids[-1] not in images:
            images.append(ids[-1])
    return images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recover", action="store_true")
    parser.add_argument("--image-id", action="append", default=[])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-images", type=int, default=24)
    parser.add_argument("--include-helpers", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 4 or args.max_images < 1:
        raise ValueError("use one to four CPU workers and a positive image bound")
    snapshot()
    if args.recover:
        ids = (args.image_id or image_ids_from_logs())[:args.max_images]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(recover_image, image_id, args.include_helpers): image_id for image_id in ids}
            for future in as_completed(futures):
                result = future.result()
                print(json.dumps({"image_id": result["image_id"], "status": result.get("status", "completed"),
                                  "files": [{"path": row["path"], "sha256": row["sha256"]} for row in result["files"]]}), flush=True)
    result = snapshot()
    missing = {h: row for h, row in result["requested_recorded_sources"].items() if not row["archived"]}
    print(json.dumps({"current_snapshots": len(result["snapshots"]),
                      "recorded_versions": len(result["requested_recorded_sources"]),
                      "archived_versions": len(result["requested_recorded_sources"]) - len(missing),
                      "missing_recorded_versions": missing}, indent=2))


if __name__ == "__main__":
    main()
