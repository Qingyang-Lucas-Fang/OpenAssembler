"""Bounded, path-checked archives shared by the frontend and its GPU worker."""
from __future__ import annotations

import io
from pathlib import Path, PurePosixPath
import stat
import zipfile

MAX_BYTES = 256 * 1024 * 1024
MAX_FILES = 20000


def unpack(blob: bytes, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        items = archive.infolist()
        if len(items) > MAX_FILES or sum(i.file_size for i in items) > MAX_BYTES:
            raise ValueError("application archive exceeds the file/size limit")
        names = set()
        for item in items:
            path = PurePosixPath(item.filename)
            mode = item.external_attr >> 16
            normalized = path.as_posix()
            if (path.is_absolute() or not path.parts or '..' in path.parts or '\\' in item.filename
                    or stat.S_ISLNK(mode) or normalized in names):
                raise ValueError(f"unsafe application archive member: {item.filename!r}")
            names.add(normalized)
            target = root.joinpath(*path.parts)
            if not target.resolve().is_relative_to(root):
                raise ValueError(f"application archive member escapes its root: {item.filename!r}")
        for item in items:
            path = root.joinpath(*PurePosixPath(item.filename).parts)
            if item.is_dir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.read(item))


def pack(files: dict[str, bytes]) -> bytes:
    if len(files) > MAX_FILES or sum(map(len, files.values())) > MAX_BYTES:
        raise ValueError("application bundle exceeds the 256 MiB / 20000 file limit")
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            archive.writestr(name, data)
    return output.getvalue()
