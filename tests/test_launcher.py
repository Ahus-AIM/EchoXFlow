"""The shared launcher retries failing jobs, blocks their dependents and skips finished jobs on a rerun."""

from __future__ import annotations

import sys
from pathlib import Path

from tasks.utils.launcher import Job, Launcher


def _command(code: int, marker: Path | None = None):
    script = (f"open({str(marker)!r}, 'w').close();" if marker else "") + f"raise SystemExit({code})"

    def commands(job: Job, device: str | None) -> list[list[str]]:
        return [[sys.executable, "-c", script]]

    return commands


def _launcher(tmp_path: Path, jobs: list[Job]) -> Launcher:
    return Launcher(jobs, title="test", status_json=tmp_path / "status.json", max_attempts=2, grace_seconds=1)


def test_failed_job_is_retried_then_blocks_dependents_and_done_jobs_are_skipped(tmp_path: Path) -> None:
    marker = tmp_path / "ok"

    def jobs() -> list[Job]:
        return [
            Job("ok", _command(0, marker), str(tmp_path / "ok.log"), done_markers=(marker,)),
            Job("bad", _command(1), str(tmp_path / "bad.log")),
            Job("child", _command(0), str(tmp_path / "child.log"), depends_on=("bad",)),
        ]

    launcher = _launcher(tmp_path, jobs())
    assert launcher.run() == 1
    assert launcher.counts() == {"done": 1, "running": 0, "pending": 0, "failed": 1, "blocked": 1}
    assert launcher.job("bad").attempts == 2

    rerun = _launcher(tmp_path, jobs())
    assert rerun.job("ok").previously_done
