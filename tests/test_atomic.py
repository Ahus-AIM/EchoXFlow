from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import pytest

from tasks.utils.atomic import (
    LockHeldError,
    RunLock,
    atomic_write_json,
    atomic_write_text,
    read_json,
    remove_stale_temporaries,
)


def test_atomic_write_replaces_file_and_leaves_no_temporary(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    atomic_write_json(path, {"value": 1})
    atomic_write_json(path, {"value": 2})

    assert read_json(path) == {"value": 2}
    assert sorted(item.name for item in tmp_path.iterdir()) == ["result.json"]


def test_failed_atomic_write_keeps_previous_file(tmp_path: Path) -> None:
    path = tmp_path / "result.txt"
    atomic_write_text(path, "complete")

    with pytest.raises(TypeError):
        atomic_write_text(path, object())  # type: ignore[arg-type]

    assert path.read_text(encoding="utf-8") == "complete"
    assert sorted(item.name for item in tmp_path.iterdir()) == ["result.txt"]


def test_remove_stale_temporaries_keeps_files_of_live_writers(tmp_path: Path) -> None:
    live = tmp_path / f".weights.pt.{os.getpid()}.abcd1234.tmp"
    dead = tmp_path / ".weights.pt.999999999.abcd1234.tmp"
    for path in (live, dead):
        path.write_bytes(b"partial")

    removed = remove_stale_temporaries(tmp_path)

    assert removed == [dead]
    assert live.exists()


def test_run_lock_is_exclusive_and_released(tmp_path: Path) -> None:
    path = tmp_path / ".lock"
    with RunLock(path):
        with pytest.raises(LockHeldError):
            RunLock(path).acquire()
    assert not path.exists()


def test_run_lock_takes_over_lock_of_dead_process(tmp_path: Path) -> None:
    path = tmp_path / ".lock"
    path.write_text(json.dumps({"pid": 999999999, "host": socket.gethostname()}), encoding="utf-8")

    lock = RunLock(path).acquire()

    assert read_json(path)["pid"] == os.getpid()  # type: ignore[index]
    lock.release()


def test_run_lock_of_another_host_is_not_taken_over(tmp_path: Path) -> None:
    path = tmp_path / ".lock"
    path.write_text(json.dumps({"pid": 1, "host": "elsewhere.invalid"}), encoding="utf-8")

    with pytest.raises(LockHeldError, match="elsewhere.invalid"):
        RunLock(path).acquire()
