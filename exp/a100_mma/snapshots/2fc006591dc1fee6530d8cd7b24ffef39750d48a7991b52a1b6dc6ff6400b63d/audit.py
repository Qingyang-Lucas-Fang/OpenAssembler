"""Read-only repository inventory; evidence is written only beside this script."""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def snapshot():
    names = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT
    ).decode().split("\0")
    records = {}
    for name in sorted(set(names)):
        if not name or name.startswith("exp/") or "/__pycache__/" in name:
            continue
        path = ROOT / name
        if not path.is_file():
            continue
        data = path.read_bytes()
        item = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        if path.suffix in (".py", ".cu", ".ptx", ".sh", ".toml", ".md", ".rs", ".cc", ".h"):
            source = data.decode(errors="replace")
            item["lines"] = len(source.splitlines())
            if path.suffix == ".py":
                tree = ast.parse(source, filename=name)
                item["doc"] = ast.get_docstring(tree)
                item["symbols"] = [
                    {"name": node.name, "start": node.lineno, "end": node.end_lineno,
                     "doc": ast.get_docstring(node)}
                    for node in ast.walk(tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                ]
                item["imports"] = sorted({
                    node.module for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom) and node.module
                })
        records[name] = item
    return records


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    dest = HERE / "repository-before.json"
    current = snapshot()
    if args.check:
        before = json.loads(dest.read_text())
        changed = [n for n in sorted(set(before) | set(current))
                   if before.get(n, {}).get("sha256") != current.get(n, {}).get("sha256")]
        print(json.dumps({"outside_exp_changed": changed}, indent=2))
        raise SystemExit(bool(changed))
    if dest.exists():
        raise SystemExit("Snapshot already exists; use --check instead of overwriting it")
    dest.write_text(json.dumps(current, indent=2) + "\n")
    print(json.dumps({"files": len(current), "python_files": sum(n.endswith('.py') for n in current),
                      "source_lines": sum(r.get("lines", 0) for r in current.values())}))
