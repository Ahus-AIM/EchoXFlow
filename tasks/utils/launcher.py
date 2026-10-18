"""Run many long, resumable jobs as subprocesses: slots, retries, dependencies, status files and clean shutdown.

Used by ``scripts/run_full_benchmark.py`` (benchmark cells) and ``scripts/reproduce_video_diffusion.py`` (Task 4
stages). A job is finished when its command exits with code 0 and every done marker exists; rerunning the same
launcher skips such jobs, so an interrupted run resumes where it stopped.

- A failing job is retried up to ``max_attempts`` times (each retry is expected to resume), then marked failed; the
  other jobs continue. A job whose dependency failed is blocked. Exit codes in ``no_retry_codes`` are not retried.
- SIGINT and SIGTERM are forwarded to the running jobs, which get ``grace_seconds`` to stop before they are killed.
  A second signal kills them at once. Interrupted jobs are left pending for the next run.
- ``status_json`` and its ``.txt`` twin are rewritten atomically on every state change.
- GPU jobs run on the given GPUs, or with ``gpus="auto"`` on any GPU nobody uses (see :mod:`tasks.utils.gpu_pool`),
  re-checked every ``GPU_POLL_SECONDS``.
"""

from __future__ import annotations

import math
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path
from types import FrameType
from typing import Any, TextIO

from tasks.utils.atomic import atomic_write_json, atomic_write_text
from tasks.utils.gpu_pool import GpuPool

REPO_ROOT = Path(__file__).resolve().parents[2]
#: Seconds between free-GPU checks with ``gpus="auto"``.
GPU_POLL_SECONDS = 60.0
EXIT_DISK = 4


class SkipJob(Exception):
    """Raised by a job's command builder: the job cannot run and is marked blocked with this reason."""


@dataclass
class Job:
    name: str
    #: ``(job, device) -> commands`` run one after the other; built when the job starts, so they can use its state.
    commands: Callable[[Job, str | None], list[list[str]]]
    log: str
    kind: str = "gpu"  # gpu: needs a GPU slot; cpu: a CPU slot when the launcher has any; local: no slot
    depends_on: tuple[str, ...] = ()
    #: Start once the dependencies finished, whatever their outcome, instead of requiring them to be done.
    after_dependencies: bool = False
    done_markers: tuple[Path, ...] = ()
    group: object = None  # for example the fold, passed to ``on_finish``
    state: str = "pending"  # pending, running, done, failed, blocked
    attempts: int = 0
    device: str | None = None
    started: str | None = None
    ended: str | None = None
    duration_s: float | None = None
    exit_code: int | None = None
    reason: str | None = None
    previously_done: bool = False
    t0: float = field(default=0.0, repr=False)

    def public(self) -> dict[str, Any]:
        skip = {"commands", "kind", "t0", "group", "after_dependencies"}
        return {
            f.name: (
                [str(path) for path in getattr(self, f.name)] if f.name == "done_markers" else getattr(self, f.name)
            )
            for f in fields(self)
            if f.name not in skip
        }

    def finished(self) -> bool:
        return all(marker.is_file() for marker in self.done_markers)


def child_env(device: str | None = None) -> dict[str, str]:
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(REPO_ROOT), env.get("PYTHONPATH"))))
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif device not in {None, "default", "local"}:
        env["CUDA_VISIBLE_DEVICES"] = str(device)
    return env


def duration_text(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(float(seconds)):
        return "n/a"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 6000:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def print_tail(path: Path, *, lines: int) -> None:
    try:
        tail = path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    except OSError:
        return
    print(f"----- last {len(tail)} lines of {path} -----", flush=True)
    for line in tail:
        print(f"  {line.rsplit(chr(13), 1)[-1]}", flush=True)
    print("-----", flush=True)


def disk_preflight(
    output_root: Path, *, final_bytes: int, active_bytes: int, margin_gb: float, ignore: bool, what: str
) -> bool:
    """Print the estimate for ``what`` and return whether ``output_root`` has room for it plus the margin."""
    margin = int(margin_gb * 1e9)
    required = final_bytes + active_bytes + margin
    probe = output_root
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    print(
        f"Disk preflight: {what} keep ~{final_bytes / 1e9:.1f} GB and need up to ~{active_bytes / 1e9:.1f} GB more "
        f"while running; with a {margin / 1e9:.0f} GB margin {required / 1e9:.1f} GB are required, "
        f"{free / 1e9:.1f} GB are free under {probe}.",
        flush=True,
    )
    if free >= required:
        return True
    if ignore:
        print("WARNING: not enough free space for the estimate; continuing (--ignore-disk-check).", flush=True)
        return True
    print(
        "Refusing to start: not enough free space. Use another --output-dir, free space, or --ignore-disk-check.",
        file=sys.stderr,
    )
    return False


class Launcher:
    def __init__(
        self,
        jobs: Sequence[Job],
        *,
        title: str,
        status_json: Path,
        max_attempts: int,
        grace_seconds: float,
        gpus: Sequence[str] | str | None = None,
        max_gpu_jobs: int | None = None,
        cpu_slots: int = 0,
        no_retry_codes: Sequence[int] = (3, 75),
        skip_done: bool = True,
        on_finish: Callable[[Launcher, Job], None] | None = None,
        status_extra: Mapping[str, object] | None = None,
    ) -> None:
        self.jobs = list(jobs)
        self.title = title
        self.status_json = status_json
        self.status_txt = status_json.with_suffix(".txt")
        self.max_attempts = max_attempts
        self.grace_seconds = grace_seconds
        self.cpu_slots = cpu_slots
        self.no_retry_codes = tuple(no_retry_codes)
        self.on_finish = on_finish
        self.status_extra = dict(status_extra or {})
        self.pool: GpuPool | None = GpuPool() if isinstance(gpus, str) else None
        self.free_gpus: list[str] = [] if gpus is None or isinstance(gpus, str) else list(gpus)
        if self.pool is not None:
            self.gpu_slots = max_gpu_jobs or len(self.pool.all_gpus())
        else:
            self.gpu_slots = max_gpu_jobs or max(1, len(self.free_gpus))
            if not self.free_gpus:
                self.free_gpus = ["default"]
        self.gpus = list(self.free_gpus)
        self.running: dict[int, tuple[Job, subprocess.Popen[bytes], TextIO, list[list[str]]]] = {}
        self.stop_signal: int | None = None
        self.started = datetime.now().isoformat(timespec="seconds")
        self.durations: list[float] = []
        self._next_gpu_poll = 0.0
        if skip_done:
            for job in self.jobs:
                if job.done_markers and job.finished():
                    job.state, job.previously_done = "done", True

    def job(self, name: str) -> Job:
        return next(job for job in self.jobs if job.name == name)

    # ------------------------------------------------------------------------------------------------- run
    def run(self) -> int:
        self._write_status()
        handlers = {signum: signal.signal(signum, self._on_signal) for signum in (signal.SIGINT, signal.SIGTERM)}
        try:
            while self.stop_signal is None:
                self._reap()
                self._start_ready()
                if not self.running and not any(job.state == "pending" for job in self.jobs):
                    break
                time.sleep(1.0)
            if self.stop_signal is not None:
                self._stop_running()
        finally:
            if self.pool is not None:
                self.pool.release_all()
            for signum, handler in handlers.items():
                signal.signal(signum, handler)
        self._write_status()
        self.print_summary()
        if self.stop_signal is not None:
            print("Stopped by signal; rerun the same command to resume.", flush=True)
            return 128 + self.stop_signal
        return 0 if all(job.state == "done" for job in self.jobs) else 1

    def _start_ready(self) -> None:
        for job in [job for job in self.jobs if job.state == "pending"]:
            if self.stop_signal is not None:
                return
            dependencies = [self.job(name) for name in job.depends_on]
            if not job.after_dependencies and any(dep.state in {"failed", "blocked"} for dep in dependencies):
                self._block(
                    job, "waiting on failed stage " + ", ".join(d.name for d in dependencies if d.state != "done")
                )
                continue
            if any(dep.state in {"pending", "running"} for dep in dependencies) or (
                not job.after_dependencies and any(dep.state != "done" for dep in dependencies)
            ):
                continue
            device = self._claim_device(job)
            if device is None:
                continue
            self._start(job, device)

    def _claim_device(self, job: Job) -> str | None:
        if job.kind == "local":
            return "local"
        if job.kind == "cpu" and self.cpu_slots > 0:
            busy = sum(1 for running, *_ in self.running.values() if running.device == "cpu")
            return "cpu" if busy < self.cpu_slots else None
        if sum(1 for running, *_ in self.running.values() if running.device not in {"cpu", "local"}) >= self.gpu_slots:
            return None
        if self.pool is None:
            return self.free_gpus.pop(0) if self.free_gpus else None
        if time.monotonic() < self._next_gpu_poll:
            return None
        try:
            device = self.pool.acquire()
        except (OSError, subprocess.CalledProcessError) as exc:
            print(f"GPU check failed ({exc}); retrying in {GPU_POLL_SECONDS:.0f} s", flush=True)
            device = None
        if device is None:
            self._next_gpu_poll = time.monotonic() + GPU_POLL_SECONDS
        return device

    def _release_device(self, device: str | None) -> None:
        if device in {None, "cpu", "local"}:
            return
        if self.pool is not None:
            self.pool.release(str(device))
        else:
            self.free_gpus.append(str(device))
            self.free_gpus.sort(key=lambda gpu: self.gpus.index(gpu))

    def _block(self, job: Job, reason: str) -> None:
        job.state, job.reason = "blocked", reason
        self._changed(f"skipped {job.name}: {reason}")

    def _start(self, job: Job, device: str) -> None:
        job.attempts += 1
        try:
            commands = job.commands(job, device)
        except SkipJob as exc:
            job.attempts -= 1
            self._release_device(device)
            self._block(job, str(exc))
            return
        job.state, job.device = "running", device
        job.started = datetime.now().isoformat(timespec="seconds")
        job.ended = job.duration_s = job.exit_code = None
        job.t0 = time.monotonic()
        Path(job.log).parent.mkdir(parents=True, exist_ok=True)
        log = open(job.log, "a", encoding="utf-8")  # noqa: SIM115 - closed when the job finishes
        log.write(f"\n===== attempt {job.attempts}/{self.max_attempts} on {_device_label(device)} at {job.started} ")
        log.write(f"(launcher pid {os.getpid()} on {socket.gethostname()}) =====\n")
        self._spawn(job, log, commands)
        self._changed(f"started {job.name} on {_device_label(device)} (attempt {job.attempts}/{self.max_attempts})")

    def _spawn(self, job: Job, log: TextIO, commands: list[list[str]]) -> None:
        command = commands[0]
        log.write(f"$ {shlex.join(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            cwd=REPO_ROOT,
            env=child_env(job.device),
            start_new_session=True,  # own process group: signals are forwarded explicitly, workers included
        )
        self.running[process.pid] = (job, process, log, commands[1:])

    def _reap(self) -> None:
        for pid, (job, process, log, rest) in list(self.running.items()):
            code = process.poll()
            if code is None:
                continue
            del self.running[pid]
            _sweep_process_group(pid)  # DataLoader workers left behind when the job was killed (e.g. sudo kill)
            if code == 0 and rest:
                self._spawn(job, log, rest)
                continue
            log.close()
            self._finish(job, int(code))

    def _finish(self, job: Job, code: int) -> None:
        job.exit_code = code
        job.ended = datetime.now().isoformat(timespec="seconds")
        job.duration_s = round(time.monotonic() - job.t0, 1)
        device = _device_label(job.device)
        self._release_device(job.device)
        self._next_gpu_poll = 0.0
        if code == 0 and job.finished():
            job.state, job.reason = "done", job.reason if job.reason and job.reason.startswith("partial") else None
            self.durations.append(float(job.duration_s))
            self._changed(f"completed {job.name} on {device} in {duration_text(job.duration_s)}")
        else:
            if code == 0:
                job.reason = "exited 0 without its output marker"
            elif code < 0:
                job.reason = f"killed by signal {-code}"
            else:
                job.reason = f"exit code {code}"
            retry = code not in self.no_retry_codes and job.attempts < self.max_attempts
            job.state = "pending" if retry else "failed"
            if retry:  # a retry goes behind the other jobs
                self.jobs.remove(job)
                self.jobs.append(job)
            action = f"retrying (attempt {job.attempts + 1}/{self.max_attempts})" if retry else "marked failed"
            self._changed(f"FAILED {job.name} on {device}: {job.reason}; {action}; log: {job.log}")
            print_tail(Path(job.log), lines=40)
        if self.on_finish is not None:
            self.on_finish(self, job)

    # --------------------------------------------------------------------------------------------- signals
    def _on_signal(self, signum: int, frame: FrameType | None) -> None:
        del frame
        if self.stop_signal is not None:
            print(f"Received signal {signum} again; killing running jobs now.", flush=True)
            self._signal_running(signal.SIGKILL)
            return
        self.stop_signal = signum
        print(
            f"Received signal {signum}: stopping {len(self.running)} running job(s) "
            f"(grace {self.grace_seconds:.0f}s); rerun the same command to resume.",
            flush=True,
        )
        self._signal_running(signal.SIGTERM)

    def _signal_running(self, signum: int) -> None:
        for _, process, _, _ in self.running.values():
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signum)
                except ProcessLookupError:
                    pass

    def _stop_running(self) -> None:
        deadline = time.monotonic() + self.grace_seconds
        while time.monotonic() < deadline and any(p.poll() is None for _, p, _, _ in self.running.values()):
            time.sleep(0.5)
        self._signal_running(signal.SIGKILL)
        for job, process, log, _ in self.running.values():
            process.wait()
            log.write(f"===== interrupted by launcher signal {self.stop_signal} =====\n")
            log.close()
            self._release_device(job.device)
            job.state, job.exit_code = "pending", process.returncode
            job.ended = datetime.now().isoformat(timespec="seconds")
            job.duration_s = round(time.monotonic() - job.t0, 1)
            job.reason = f"interrupted by signal {self.stop_signal}; resumes on rerun"
        self.running.clear()

    # ---------------------------------------------------------------------------------------------- status
    def counts(self) -> dict[str, int]:
        counts = {state: 0 for state in ("done", "running", "pending", "failed", "blocked")}
        for job in self.jobs:
            counts[job.state] += 1
        return counts

    def _changed(self, message: str) -> None:
        counts = self.counts()
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        remaining = counts["pending"] + counts["running"] / 2.0
        eta = (
            duration_text(sum(self.durations) / len(self.durations) * remaining / self.gpu_slots)
            if self.durations
            else "n/a"
        )
        summary = " | ".join(f"{state} {count}" for state, count in counts.items() if count or state == "done")
        print(f"[{stamp}] {summary} | eta~{eta} | {message}", flush=True)
        self._write_status()

    def _write_status(self) -> None:
        counts = self.counts()
        updated = datetime.now().isoformat(timespec="seconds")
        atomic_write_json(
            self.status_json,
            {
                **self.status_extra,
                "launcher": {"pid": os.getpid(), "host": socket.gethostname(), "started": self.started},
                "updated": updated,
                "counts": counts,
                "jobs": [job.public() for job in self.jobs],
            },
        )
        lines = [
            f"{self.title} | updated {updated} | pid {os.getpid()} on {socket.gethostname()}",
            "  ".join(f"{state}={count}" for state, count in counts.items()) + f"  total={len(self.jobs)}",
            "",
            f"{'state':<8} {'tries':>5} {'device':<8} {'duration':>9}  {'job':<55} log / reason",
        ]
        order = {"running": 0, "failed": 1, "blocked": 2, "pending": 3, "done": 4}
        for job in sorted(self.jobs, key=lambda item: order[item.state]):
            elapsed = time.monotonic() - job.t0 if job.state == "running" else job.duration_s
            detail = job.log
            if job.previously_done:
                detail = "done before this launch"
            elif job.reason:
                detail = f"{job.reason}; log: {job.log}"
            device = "" if job.device is None else f"gpu{job.device}" if job.device.isdigit() else job.device
            duration = "" if elapsed is None else duration_text(elapsed)
            lines.append(f"{job.state:<8} {job.attempts:>5} {device:<8} {duration:>9}  {job.name:<55} {detail}")
        atomic_write_text(self.status_txt, "\n".join(lines) + "\n")

    def print_summary(self) -> None:
        counts = self.counts()
        print(
            "Summary: "
            + ", ".join(f"{count} {state}" for state, count in counts.items())
            + f". Status: {self.status_txt}",
            flush=True,
        )
        for job in self.jobs:
            if job.state in {"failed", "blocked"}:
                print(
                    f"  {job.state:<7} {job.name}: {job.reason} after {job.attempts} attempt(s); log: {job.log}",
                    flush=True,
                )


def _sweep_process_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _device_label(device: str | None) -> str:
    if device == "cpu":
        return "CPU"
    if device == "local":
        return "this environment"
    if device in {None, "default"}:
        return "default device"
    return f"GPU {device}"
