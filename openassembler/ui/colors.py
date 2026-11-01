"""ANSI color helpers. Uses the standard 16-color SGR codes (works in nearly
every terminal, no truecolor detection needed) and a no-color fallback when
color is disabled (NO_COLOR, non-tty, TERM=dumb). Stdlib only -- this
project ships with dependencies = [].
"""

from __future__ import annotations

import os
import sys

RESET = "\033[0m"

# The four bands in the reference logo (figs/OpenAssembler_logo.png), top to
# bottom: ANSI blue -> bright blue -> white -> bright white. Standard 16-color
# SGR codes (30-37 normal, 90-97 bright), not truecolor -- this matches the
# original design exactly and renders correctly in basic terminals too.
BANDS = ("\033[0;34m", "\033[0;94m", "\033[0;37m", "\033[0;97m")


def supports_color(stream=sys.stdout) -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("OPENASSEMBLER_FORCE_COLOR"):
        return True
    if not hasattr(stream, "isatty") or not stream.isatty():
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return True


def band(t: float, enabled: bool = True) -> str:
    """t in [0, 1] (top to bottom) -> one of the 4 SGR color codes above."""
    if not enabled:
        return ""
    i = min(len(BANDS) - 1, int(t * len(BANDS)))
    return BANDS[i]
