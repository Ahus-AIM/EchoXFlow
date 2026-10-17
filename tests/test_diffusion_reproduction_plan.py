from __future__ import annotations

from pathlib import Path

import pytest

from scripts.reproduce_video_diffusion import Reproduction, _parse_args


@pytest.mark.parametrize("space", ["beamspace", "cartesian"])
def test_single_space_plan_revalidates_without_comparison_tables(tmp_path: Path, space: str) -> None:
    (tmp_path / "croissant.json").write_text("{}")
    (tmp_path / "camus").mkdir(exist_ok=True)
    (tmp_path / "camus" / "croissant.json").write_text("{}")
    run_dir = tmp_path / "fold4" / f"{space}_0128"
    (run_dir / "evaluation").mkdir(parents=True)
    (run_dir / "training_complete.json").write_text("{}")
    (run_dir / "evaluation" / "metrics.json").write_text("{}")
    (run_dir / "evaluation" / "camus").mkdir()
    (run_dir / "evaluation" / "camus" / "metrics.json").write_text("{}")
    args = _parse_args(
        [
            "--data-root",
            str(tmp_path),
            "--camus-root",
            str(tmp_path / "camus"),
            "--output-dir",
            str(tmp_path),
            "--resolutions",
            "128",
            "--folds",
            "4",
        ]
        + ["--spaces", space]
    )
    reproduction = Reproduction(args)
    assert len(reproduction.launcher.jobs) == 2
    assert all(job.state == "pending" for job in reproduction.launcher.jobs)


def test_two_space_plan_keeps_comparison_tables(tmp_path: Path) -> None:
    (tmp_path / "croissant.json").write_text("{}")
    (tmp_path / "camus").mkdir(exist_ok=True)
    (tmp_path / "camus" / "croissant.json").write_text("{}")
    reproduction = Reproduction(
        _parse_args(
            [
                "--data-root",
                str(tmp_path),
                "--camus-root",
                str(tmp_path / "camus"),
                "--output-dir",
                str(tmp_path),
                "--resolutions",
                "128",
            ]
        )
    )
    for job in reproduction.launcher.jobs:
        if job.name.startswith("evaluate/"):
            job.state = "done"
    tables = reproduction.launcher.job("tables/fold4")
    command = tables.commands(tables, "local")[0]
    assert command[command.index("--pair") + 1 : command.index("--pair") + 4] == [
        "128",
        str(tmp_path / "fold4" / "beamspace_0128" / "evaluation"),
        str(tmp_path / "fold4" / "cartesian_0128" / "evaluation"),
    ]
