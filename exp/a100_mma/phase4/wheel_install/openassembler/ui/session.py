"""Tracks whether the logo has already been shown in the current terminal
session, so `openassembler <command>` shows it on the first command in a fresh
shell and stays quiet on every command after that -- until the terminal is
closed (which ends the shell process) and reopened.

Mechanism: a stamp file keyed by the *parent* process's PID. Each
`openassembler` invocation is a new process; its parent is the shell you typed
the command into, which keeps the same PID for the life of that terminal
session. A new terminal (or a new shell) gets a new PID, so the stamp
naturally "resets" without us having to detect that explicitly. We don't key
off a TTY path because some terminal apps reuse the same TTY slot for a new
session, which would wrongly suppress the banner.
"""

from __future__ import annotations

import os
import tempfile


def _stamp_path() -> str:
    shell_pid = os.getppid()
    return os.path.join(tempfile.gettempdir(), f"openassembler-shown-{shell_pid}")


def consume_show_once() -> bool:
    """Returns True the first time it's called for this shell session (and
    creates the stamp), False every time after. Best-effort: if the stamp
    file can't be written (read-only /tmp, odd permissions), just show the
    banner every time rather than fail the command."""
    path = _stamp_path()
    if os.path.exists(path):
        return False
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False
    except OSError:
        return True
