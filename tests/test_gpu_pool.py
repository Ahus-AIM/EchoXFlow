"""Launchers claim only GPUs nobody is using, never the same GPU twice, and leave no workers behind."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tasks.utils import gpu_pool
from tasks.utils.gpu_pool import GpuPool
from tasks.utils.launcher import Job, Launcher


def _fake_smi(busy_uuid: str, memory: dict[str, int]):
    def run(*args: str) -> list[list[str]]:
        if args[0] == "--query-gpu=index,memory.used":
            return [[index, str(used)] for index, used in memory.items()]
        if args[0] == "--query-gpu=index,uuid":
            return [[index, f"GPU-{index}"] for index in memory]
        return [[busy_uuid, "123"]]

    return run


def test_pool_skips_busy_and_claimed_gpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_pool, "_nvidia_smi", _fake_smi("GPU-0", {"0": 20000, "1": 4000, "2": 5, "3": 5}))
    first, second = GpuPool(tmp_path), GpuPool(tmp_path)

    assert first.acquire() == "2"  # 0 runs a process, 1 holds memory without a listed process
    assert second.acquire() == "3"  # 2 is claimed by the other launcher
    assert second.acquire() is None
    first.release("2")
    assert second.acquire() == "2"


def test_claims_of_dead_processes_are_reclaimed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_pool, "_nvidia_smi", _fake_smi("GPU-9", {"0": 0}))
    (tmp_path / "gpu0").write_text("999999999", encoding="utf-8")

    assert GpuPool(tmp_path).acquire() == "0"


_SPAWNS_A_WORKER = (
    "import subprocess, sys, time; worker = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
    "print(worker.pid, flush=True); time.sleep(60)"
)


def _gone(pid: int) -> bool:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.1)
    return False


def test_launcher_kills_workers_left_behind_by_a_killed_job(tmp_path: Path) -> None:
    """A job with a DataLoader-like worker in its process group, killed with SIGKILL like a colleague's sudo kill."""
    process = subprocess.Popen([sys.executable, "-c", _SPAWNS_A_WORKER], start_new_session=True, stdout=subprocess.PIPE)
    assert process.stdout is not None
    worker = int(process.stdout.readline())
    process.kill()
    job = Job(name="job", commands=lambda job, device: [], log=str(tmp_path / "job.log"), kind="local")
    launcher = Launcher([job], title="t", status_json=tmp_path / "status.json", max_attempts=1, grace_seconds=1.0)
    job.state, job.device, job.attempts = "running", "local", 1
    launcher.running[process.pid] = (job, process, open(job.log, "a", encoding="utf-8"), [])
    try:
        process.wait(timeout=10)
        launcher._reap()

        assert job.state == "failed" and job.exit_code == -signal.SIGKILL
        assert _gone(worker)  # it would otherwise keep its GPU memory and the GPU would look busy
    finally:
        try:
            os.kill(worker, signal.SIGKILL)
        except ProcessLookupError:
            pass
