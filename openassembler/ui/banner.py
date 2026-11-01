"""The splash banner: the exact OPENASSEMBLER wordmark art (openassembler.ui.logo)
above a bordered status/tips box -- the same two pieces as Claude Code's own
welcome screen, just stacked instead of side by side since our wordmark is
wider than Claude's small glyph.
"""

from __future__ import annotations

import re
import shutil

from openassembler import __version__
from openassembler.ui.colors import supports_color
from openassembler.ui.logo import LINES as LOGO_LINES

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _strip(s: str) -> str:
    return _ANSI.sub("", s)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur:
        lines.append(cur)
    return lines or [""]


TIPS = [
    "Run `compile <file.ptx> -o out.cubin --verify` to compile with our own"
    " backend and check it against the PTX reference interpreter.",
    "`gpu run --backend user@host` runs on any NVIDIA box you can SSH into --"
    " no CUDA toolkit needed there.",
    "`status` prints the headline, reproducible results from docs/STATUS.md.",
    "`help` lists every command; `<command> --help` shows its flags.",
]


def logo_lines(color: bool | None = None) -> list[str]:
    """The literal wordmark art, or its plain (uncolored) text if color is
    disabled -- same visible shape either way."""
    if color is None:
        color = supports_color()
    return list(LOGO_LINES) if color else [_strip(l) for l in LOGO_LINES]


def render(width: int | None = None, include_logo: bool = True) -> str:
    cols = width or shutil.get_terminal_size(fallback=(80, 24)).columns
    cols = max(60, min(cols, 140))
    color = supports_color()

    out = []
    if include_logo:
        out.append("")
        out += logo_lines(color)
        out.append("")

    inner = cols - 2
    header = f"openassembler v{__version__} · PTX → SASS with our own backend"
    out.append(f"╭{'─' * inner}╮")
    out.append(f"│{header.center(inner)}│")
    out.append(f"│{' ' * inner}│")
    out.append(f"│{'Tips for getting started'.ljust(inner)}│")
    for tip in TIPS:
        for line in _wrap(tip, inner - 2):
            out.append(f"│  {line.ljust(inner - 2)}│")
    out.append(f"╰{'─' * inner}╯")
    return "\n".join(out)
