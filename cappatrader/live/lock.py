"""Single-instance lock: one running engine per state database.

`resume` / `reset` refuse to edit the account while an engine is running, because the
engine keeps the account in memory and would overwrite the change at the next save.
"""

from __future__ import annotations

import os
from pathlib import Path


def lock_path(state_db: str) -> Path:
    return Path(f"{state_db}.lock")


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def running_pid(state_db: str) -> int | None:
    """PID of the engine holding the lock, or None (stale locks are ignored)."""
    p = lock_path(state_db)
    try:
        pid = int(p.read_text().strip())
    except (OSError, ValueError):
        return None
    return pid if pid != os.getpid() and _pid_alive(pid) else None


class RunLock:
    def __init__(self, state_db: str):
        self.path = lock_path(state_db)
        self.held = False

    def acquire(self) -> None:
        if self.path.parent != Path(""):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        other = running_pid(str(self.path)[: -len(".lock")])
        if other is not None:
            raise RuntimeError(f"Bu hesap için zaten çalışan bir bot var (PID {other}). Önce onu durdurun.")
        self.path.write_text(str(os.getpid()))
        self.held = True

    def release(self) -> None:
        if self.held:
            try:
                if self.path.read_text().strip() == str(os.getpid()):
                    self.path.unlink()
            except OSError:
                pass
            self.held = False
