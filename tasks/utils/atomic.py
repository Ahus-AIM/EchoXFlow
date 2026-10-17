"""Crash-safe file writes and a per-directory run lock for long, resumable runs.

Every result or checkpoint file is written to a temporary file in the destination directory, flushed and fsynced, and
then moved over the destination with :func:`os.replace`. A crash or ``SIGKILL`` at any moment leaves either the
previous complete file or the new complete file, never a truncated one. Interrupted writes leave only hidden
``.<name>.<pid>.<token>.tmp`` files, which :func:`remove_stale_temporaries` deletes.
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import socket
import time
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import IO, Any

TEMPORARY_SUFFIX = ".tmp"


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}{TEMPORARY_SUFFIX}")


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


@contextlib.contextmanager
def atomic_open(path: str | Path, mode: str = "w", **kwargs: Any) -> Iterator[IO[Any]]:
    """Open a temporary file next to ``path``; on success fsync it and atomically replace ``path``."""
    if mode not in {"w", "wb"}:
        raise ValueError(f"atomic_open supports only 'w' and 'wb', got {mode!r}")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(destination)
    try:
        with open(temporary, mode, **kwargs) as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    with atomic_open(path, "wb") as handle:
        handle.write(data)


def atomic_write_text(path: str | Path, text: str) -> None:
    with atomic_open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def atomic_write_json(path: str | Path, payload: Mapping[str, object]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def atomic_save(path: str | Path, writer: Callable[[IO[bytes]], None]) -> None:
    """Atomically write a binary file produced by ``writer(handle)`` (for example ``torch.save``)."""
    with atomic_open(path, "wb") as handle:
        writer(handle)


def atomic_torch_save(payload: object, path: str | Path) -> None:
    import torch

    atomic_save(path, lambda handle: torch.save(payload, handle))


def read_json(path: str | Path) -> dict[str, Any] | None:
    """Return the JSON mapping in ``path``, or ``None`` when it is missing or unreadable."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def remove_stale_temporaries(directory: str | Path, *, only_dead_writers: bool = True) -> list[Path]:
    """Delete temporary files left in ``directory`` by interrupted atomic writes; return the removed paths.

    With ``only_dead_writers`` a temporary file is kept while the process that created it is still alive.
    """
    root = Path(directory)
    removed: list[Path] = []
    if not root.is_dir():
        return removed
    for candidate in root.glob(f".*{TEMPORARY_SUFFIX}"):
        parts = candidate.name.rsplit(".", 3)
        pid = int(parts[1]) if len(parts) == 4 and parts[1].isdigit() else None
        if only_dead_writers and pid is not None and _process_alive(pid):
            continue
        with contextlib.suppress(OSError):
            candidate.unlink()
            removed.append(candidate)
    return removed


class LockHeldError(RuntimeError):
    """Another live process holds the run lock."""


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class RunLock:
    """Exclusive lock file for one run directory, created atomically with ``O_CREAT | O_EXCL``.

    The file records the owner's host and PID. A lock whose owner on this host is no longer
    running (killed, crashed, rebooted) is stale and is taken over. A lock held by another host cannot be checked and
    is reported with the path to delete once that host is known to be done.
    """

    def __init__(self, path: str | Path, *, label: str | None = None) -> None:
        self.path = Path(path)
        self.label = label or str(self.path.parent)
        self._held = False

    def acquire(self) -> RunLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(3):
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                owner = read_json(self.path)
                if self._is_stale(owner):
                    with contextlib.suppress(FileNotFoundError):
                        self.path.unlink()
                    continue
                raise LockHeldError(self._held_message(owner)) from None
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(self._owner(), handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            self._held = True
            return self
        raise LockHeldError(f"could not acquire {self.path}")

    def release(self) -> None:
        if not self._held:
            return
        owner = read_json(self.path)
        if owner is not None and owner.get("pid") == os.getpid() and owner.get("host") == socket.gethostname():
            with contextlib.suppress(FileNotFoundError):
                self.path.unlink()
        self._held = False

    def __enter__(self) -> RunLock:
        return self.acquire()

    def __exit__(self, *exc_info: object) -> None:
        self.release()

    @staticmethod
    def _owner() -> dict[str, object]:
        pid = os.getpid()
        return {
            "pid": pid,
            "host": socket.gethostname(),
            "acquired_unix": round(time.time(), 3),
        }

    def _is_stale(self, owner: Mapping[str, Any] | None) -> bool:
        if owner is None:
            # Created but not yet written (or unreadable): stale unless it is brand new.
            try:
                return time.time() - self.path.stat().st_mtime > 60.0
            except FileNotFoundError:
                return True
        if owner.get("host") != socket.gethostname():
            return False
        return not _process_alive(int(owner.get("pid", -1)))

    def _held_message(self, owner: Mapping[str, Any] | None) -> str:
        who = "unknown owner" if owner is None else f"PID {owner.get('pid')} on {owner.get('host')}"
        return (
            f"{self.label} is locked by {who} ({self.path}). Another process is running this cell; if it is not "
            "(for example a crashed job on another host), delete the lock file and rerun."
        )


__all__ = [
    "LockHeldError",
    "RunLock",
    "atomic_open",
    "atomic_save",
    "atomic_torch_save",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_text",
    "read_json",
    "remove_stale_temporaries",
]
