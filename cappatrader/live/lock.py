"""Single-instance lock: one running engine per state database.

`resume` / `reset` refuse to edit the account while an engine is running, because the
engine keeps the account in memory and would overwrite the change at the next save.

The lock is an OS lock held on `<state_db>.lock` for the engine's lifetime and released by
the OS when the process dies: flock on Linux / macOS (it also works across Docker containers
that share the state volume, where every container's python is PID 1, so PIDs alone cannot
tell two engines apart) and msvcrt.locking on Windows. The PID written into the file is
informational only. Where the filesystem supports neither, a PID file check is the fallback.
"""

from __future__ import annotations

import errno
import logging
import os
from pathlib import Path

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None
try:
    import msvcrt
except ImportError:  # Linux / macOS
    msvcrt = None

log = logging.getLogger(__name__)

# errors that mean "another process holds it"; anything else means this filesystem cannot lock
_HELD = {errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES, getattr(errno, "EDEADLOCK", errno.EDEADLK)}
# Windows locks a byte range: lock one byte well past the PID text so that stays readable
_WIN_OFFSET = 1 << 20


def lock_path(state_db: str) -> Path:
    return Path(f"{state_db}.lock")


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":  # os.kill(pid, 0) would send CTRL_C_EVENT on Windows, never use it as a probe
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5  # ERROR_ACCESS_DENIED: exists, owned by someone else
        try:
            code = ctypes.c_ulong()
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _read_pid(p: Path) -> int | None:
    try:
        return int(p.read_text(encoding="ascii").strip())
    except (OSError, ValueError, UnicodeDecodeError):
        return None


def _pid_check(p: Path) -> int | None:
    pid = _read_pid(p)
    return pid if pid is not None and pid != os.getpid() and _pid_alive(pid) else None


def _os_lock(fd: int, shared: bool = False) -> None:
    """Non-blocking OS lock: BlockingIOError when another process holds it, OSError when unsupported."""
    try:
        if fcntl is not None:
            fcntl.flock(fd, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        elif msvcrt is not None:
            os.lseek(fd, _WIN_OFFSET, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            raise OSError(errno.ENOSYS, "OS file locks are not available")
    except BlockingIOError:
        raise
    except OSError as exc:
        if exc.errno in _HELD:
            raise BlockingIOError(exc.errno, str(exc)) from exc
        raise


def _os_unlock(fd: int) -> None:
    if fcntl is not None:
        fcntl.flock(fd, fcntl.LOCK_UN)
    elif msvcrt is not None:
        os.lseek(fd, _WIN_OFFSET, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)


def running_pid(state_db: str) -> int | None:
    """PID of the engine holding the lock (0 when held but the PID is unreadable), or None."""
    p = lock_path(state_db)
    if not p.exists():
        return None
    try:  # a shared probe works on a read-only descriptor; Windows has only exclusive locks
        fd = os.open(p, os.O_RDONLY if fcntl is not None else os.O_RDWR)
    except OSError:
        return _pid_check(p)
    try:
        _os_lock(fd, shared=True)
    except BlockingIOError:
        pid = _read_pid(p)
        return pid if pid is not None else 0
    except OSError:
        return _pid_check(p)  # this filesystem cannot lock (e.g. some NFS / CIFS mounts): PID file only
    else:
        _os_unlock(fd)
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
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            _os_lock(fd)
        except BlockingIOError:
            os.close(fd)
            self._refuse(_read_pid(self.path))
        except OSError as exc:
            os.close(fd)
            # no OS locks on this filesystem: fall back to the PID file (not safe across containers)
            log.warning("Kilit dosyası işletim sistemi kilidini desteklemiyor (%s); PID tabanlı kilit kullanılıyor",
                        exc)
            other = _pid_check(self.path)
            if other is not None:
                self._refuse(other)
            self.path.write_text(str(os.getpid()), encoding="ascii")
            self.held = True
            return
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode())
        os.fsync(fd)
        self._fd, self.held = fd, True

    def release(self) -> None:
        if not self.held:
            return
        if self._fd is not None:
            try:
                _os_unlock(self._fd)
            except OSError:
                pass
            finally:
                os.close(self._fd)
                self._fd = None
        else:  # PID fallback
            try:
                if self.path.read_text(encoding="ascii").strip() == str(os.getpid()):
                    self.path.unlink()
            except OSError:
                pass
        self.held = False
