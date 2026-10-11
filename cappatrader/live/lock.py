"""Single-instance lock: one running engine per state database.

`resume` / `reset` refuse to edit the account while an engine is running, because the
engine keeps the account in memory and would overwrite the change at the next save.

The lock is an OS advisory lock (flock) held on `<state_db>.lock` for the engine's lifetime:
the kernel releases it when the process dies, and it works across Docker containers that
share the state volume (where every container's python is PID 1, so PIDs alone cannot tell
two engines apart). The PID written into the file is informational only. Platforms without
fcntl (Windows) fall back to the PID check.
"""

from __future__ import annotations

import errno
import logging
import os
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None


log = logging.getLogger(__name__)

# flock errors that mean "another process holds it"; anything else means the filesystem cannot flock
_HELD = {errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES}


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


def _read_pid(p: Path) -> int | None:
    try:
        return int(p.read_text().strip())
    except (OSError, ValueError):
        return None


def _pid_check(p: Path) -> int | None:
    pid = _read_pid(p)
    return pid if pid is not None and pid != os.getpid() and _pid_alive(pid) else None


def running_pid(state_db: str) -> int | None:
    """PID of the engine holding the lock (0 when held but the PID is unreadable), or None."""
    p = lock_path(state_db)
    if not p.exists():
        return None
    if fcntl is None:
        return _pid_check(p)
    try:
        fd = os.open(p, os.O_RDONLY)
    except OSError:
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)  # conflicts with the engine's exclusive lock
    except OSError as exc:
        if exc.errno in _HELD:
            pid = _read_pid(p)
            return pid if pid is not None else 0
        return _pid_check(p)  # this filesystem cannot flock (e.g. some NFS / CIFS mounts): PID file only
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return None  # a leftover file from an engine that exited: nobody holds it
    finally:
        os.close(fd)


class RunLock:
    def __init__(self, state_db: str):
        self.path = lock_path(state_db)
        self.held = False
        self._fd: int | None = None

    def _refuse(self, pid: int | None) -> None:
        who = f" (PID {pid})" if pid else ""
        raise RuntimeError(f"Bu hesap için zaten çalışan bir bot var{who}. Önce onu durdurun.")

    def acquire(self) -> None:
        if self.path.parent != Path(""):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        if fcntl is None:
            other = running_pid(str(self.path)[: -len(".lock")])
            if other is not None:
                self._refuse(other)
            self.path.write_text(str(os.getpid()))
            self.held = True
            return
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            if exc.errno in _HELD:
                self._refuse(_read_pid(self.path))
            # no flock on this filesystem: fall back to the PID file (not safe across containers)
            log.warning("Kilit dosyası flock desteklemiyor (%s); PID tabanlı kilit kullanılıyor", exc)
            other = _pid_check(self.path)
            if other is not None:
                self._refuse(other)
            self.path.write_text(str(os.getpid()))
            self.held = True
            return
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode())
        os.fsync(fd)
        self._fd, self.held = fd, True

    def release(self) -> None:
        if not self.held:
            return
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None
        else:  # PID fallback
            try:
                if self.path.read_text().strip() == str(os.getpid()):
                    self.path.unlink()
            except OSError:
                pass
        self.held = False
