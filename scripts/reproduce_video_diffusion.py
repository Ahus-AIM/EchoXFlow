"""Reproduce Task 4 (ECG-conditioned video diffusion) with resumable, independently retried stages.

For every held-out fold, resolution and coordinate space this trains a model (``tasks.video_diffusion.train``) and
evaluates it (``tasks.video_diffusion.evaluate``); per fold it then builds the bootstrap comparison tables
(``tasks.video_diffusion.tables``) from every resolution whose beamspace and Cartesian evaluations both finished. With
more than one fold, a last stage pools the per-fold tables into the cross-validated table
(``tasks.video_diffusion.cv_table``, ``OUTPUT_DIR/tables/comparison_cv.*``).

Stages validate provenance and required artifacts before accepting completed training or evaluation.
Finished stages are skipped on rerun and failing ones are retried (``--max-attempts``); the exit code is non-zero when
any stage failed. ``OUTPUT_DIR/status/video_diffusion.txt`` lists every stage. A disk preflight refuses to start without
enough free space (``--ignore-disk-check``).
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.utils.launcher import EXIT_DISK, Job, Launcher, SkipJob, disk_preflight  # noqa: E402

DEFAULT_EXTRACTORS = ["echoprime_view", "echofine", "echoclip", "inception", "panecho", "echojepa", "echoprime_mvit"]
MODEL_BYTES = 210_000_000  # fp32 size of the diffusion model, and of ema.pt
FEATURES_BYTES_PER_EVALUATION = 300_000_000  # frame and clip features of the seven extractors, real + generated
Commands = Callable[[Job, str | None], list[list[str]]]


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="scripts/reproduce_video_diffusion.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add = parser.add_argument
    add("--data-root", default=os.environ.get("ECHOXFLOW_DATA_ROOT"), help="root with croissant.json")
    add("--split-csv", help="fold CSV (default: <data-root>/splits.csv)")
    add("--output-dir", default="outputs/video_diffusion", help="runs, tables, logs and status")
    add("--resolutions", type=int, nargs="+", default=[128, 256, 320], help="square frame sizes (default 128 256 320)")
    add("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="held-out folds (default 0 1 2 3 4)")
    add(
        "--spaces",
        nargs="+",
        choices=["beamspace", "cartesian"],
        default=["beamspace", "cartesian"],
        help="coordinate spaces to train (comparison tables require both)",
    )
    add("--extractors", nargs="+", default=DEFAULT_EXTRACTORS, help="feature spaces to evaluate")
    add("--replicates", type=int, default=2000, help="bootstrap replicates (default 2000)")
    add("--camus-root", default=os.environ.get("ECHOXFLOW_CAMUS_ROOT"), help="CAMUS EchoXFlow package root")
    add("--max-attempts", type=int, default=2, help="attempts per stage, each resuming (default 2)")
    add("--keep-latest", action="store_true", help="keep latest.pt after training completes")
    add("--keep-samples", action="store_true", help="keep generated clips as samples.npz")
    add("--train-args", default="", help="extra arguments for tasks.video_diffusion.train")
    add("--evaluate-args", default="", help="extra arguments for tasks.video_diffusion.evaluate")
    add(
        "--gpus",
        nargs="+",
        help="GPU ids that run one stage each concurrently, or 'auto': one stage at a time on a GPU nobody uses "
        "(default: one stage at a time on CUDA_VISIBLE_DEVICES)",
    )
    add("--status-name", default="video_diffusion", help="status file name; concurrent launchers need their own")
    add("--skip-tables", action="store_true", help="train and evaluate only; build the tables in a later run")
    add("--ignore-disk-check", action="store_true", help="start even if the disk preflight fails")
    add("--disk-margin-gb", type=float, default=20.0, help="free-space safety margin (default 20)")
    add("--grace-seconds", type=float, default=60.0, help="time a stage gets to stop on a signal")
    args = parser.parse_args(argv)
    if not args.data_root or not (Path(args.data_root) / "croissant.json").is_file():
        parser.error("set --data-root (or ECHOXFLOW_DATA_ROOT) to the EchoXFlow root containing croissant.json")
    if not args.camus_root or not (Path(args.camus_root) / "croissant.json").is_file():
        parser.error(
            "set --camus-root (or ECHOXFLOW_CAMUS_ROOT) to the CAMUS EchoXFlow package containing croissant.json"
        )
    if args.max_attempts < 1:
        parser.error("--max-attempts must be at least 1")
    return args


class Reproduction:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.data_root = Path(args.data_root).expanduser().resolve()
        self.split_csv = Path(args.split_csv or self.data_root / "splits.csv").expanduser().resolve()
        self.output_root = Path(args.output_dir).expanduser().resolve()
        self.camus_root: Path | None = Path(args.camus_root).expanduser().resolve()
        self.resolutions, self.folds, self.spaces = args.resolutions, args.folds, args.spaces
        self.log_dir = self.output_root / "logs"
        self.launcher = Launcher(
            self._plan(),
            title="Task 4 reproduction",
            status_json=self.output_root / "status" / f"{args.status_name}.json",
            max_attempts=args.max_attempts,
            grace_seconds=args.grace_seconds,
            gpus="auto" if args.gpus == ["auto"] else args.gpus,
            max_gpu_jobs=1 if args.gpus in (None, ["auto"]) else len(args.gpus),
            skip_done=False,  # children validate provenance and artifacts before accepting completion
            status_extra={"output_root": str(self.output_root)},
        )

    def _plan(self) -> list[Job]:
        jobs: list[Job] = []
        tables: list[str] = []
        for fold in self.folds:
            pairs: list[str] = []
            for resolution in self.resolutions:
                for space in self.spaces:
                    run_dir = self.output_root / f"fold{fold}" / f"{space}_{resolution:04d}"
                    evaluation_dir = run_dir / "evaluation"
                    train = self._job(
                        f"train/fold{fold}/{space}_{resolution:04d}",
                        self._train(fold, space, resolution, run_dir),
                        done_markers=(run_dir / "training_complete.json",),
                    )
                    markers = [evaluation_dir / "metrics.json"]
                    if self.camus_root is not None:
                        markers.append(evaluation_dir / "camus" / "metrics.json")
                    evaluate = self._job(
                        f"evaluate/fold{fold}/{space}_{resolution:04d}",
                        self._evaluate(run_dir, evaluation_dir),
                        depends_on=(train.name,),
                        done_markers=tuple(markers),
                    )
                    jobs += [train, evaluate]
                    pairs.append(evaluate.name)
            if not self.args.skip_tables and {"beamspace", "cartesian"} <= set(self.spaces):
                # The bootstrap recomputes feature distances on the GPU; without a claim every fold piles onto GPU 0.
                table = self._job(
                    f"tables/fold{fold}", self._tables(fold), depends_on=tuple(pairs), after_dependencies=True
                )
                jobs.append(table)
                tables.append(table.name)
        if len(tables) > 1:
            jobs.append(self._job("tables/cv", self._cv_table(), kind="local", depends_on=tuple(tables)))
        return jobs

    def _job(self, name: str, commands: Commands, **kwargs: object) -> Job:
        return Job(name=name, commands=commands, log=str(self.log_dir / f"{name.replace('/', '_')}.log"), **kwargs)  # type: ignore[arg-type]

    def _train(self, fold: int, space: str, resolution: int, run_dir: Path) -> Commands:
        def commands(job: Job, device: str | None) -> list[list[str]]:
            command = [
                sys.executable,
                "-m",
                "tasks.video_diffusion.train",
                "--data-root",
                str(self.data_root),
                "--split-csv",
                str(self.split_csv),
                "--evaluation-fold",
                str(fold),
                "--coordinate-space",
                space,
                "--resolution",
                str(resolution),
                "--output-dir",
                str(run_dir),
                *(["--keep-latest"] if self.args.keep_latest else []),
                *shlex.split(self.args.train_args),
            ]
            latest = run_dir / "latest.pt"
            return [[*command, *(["--resume", str(latest)] if latest.is_file() else [])]]

        return commands

    def _evaluate(self, run_dir: Path, evaluation_dir: Path) -> Commands:
        def commands(job: Job, device: str | None) -> list[list[str]]:
            return [
                [
                    sys.executable,
                    "-m",
                    "tasks.video_diffusion.evaluate",
                    "--checkpoint",
                    str(run_dir / "ema.pt"),
                    "--data-root",
                    str(self.data_root),
                    "--output-dir",
                    str(evaluation_dir),
                    "--extractors",
                    *self.args.extractors,
                    *(["--camus-root", str(self.camus_root)] if self.camus_root is not None else []),
                    *(["--keep-samples"] if self.args.keep_samples else []),
                    *shlex.split(self.args.evaluate_args),
                ]
            ]

        return commands

    def _tables(self, fold: int) -> Commands:
        def commands(job: Job, device: str | None) -> list[list[str]]:
            tables_dir = self.output_root / f"fold{fold}" / "tables"
            command = [sys.executable, "-m", "tasks.video_diffusion.tables"]
            missing = []
            for resolution in self.resolutions:
                names = [f"evaluate/fold{fold}/{space}_{resolution:04d}" for space in ("beamspace", "cartesian")]
                if all(self.launcher.job(name).state == "done" for name in names):
                    base = self.output_root / f"fold{fold}"
                    paths = [
                        str(base / f"{space}_{resolution:04d}" / "evaluation") for space in ("beamspace", "cartesian")
                    ]
                    command += ["--pair", str(resolution), *paths]
                else:
                    missing.append(str(resolution))
            if len(missing) == len(self.resolutions):
                raise SkipJob("no resolution has both evaluations")
            job.reason = f"partial: resolutions {', '.join(missing)} missing" if missing else None
            command += ["--replicates", str(self.args.replicates)]
            result = [[*command, "--output-dir", str(tables_dir)]]
            if self.camus_root is not None:
                result.append([*command, "--camus", "--output-dir", f"{tables_dir}_camus"])
            return result

        return commands

    def _cv_table(self) -> Commands:
        def commands(job: Job, device: str | None) -> list[list[str]]:
            command = [sys.executable, "-m", "tasks.video_diffusion.cv_table", "--root", str(self.output_root)]
            command += ["--folds", *(str(fold) for fold in self.folds)]
            tables_dir = self.output_root / "tables"
            result = [[*command, "--output-dir", str(tables_dir)]]
            if self.camus_root is not None:
                result.append([*command, "--tables-name", "tables_camus", "--output-dir", f"{tables_dir}_camus"])
            return result

        return commands

    def run(self) -> int:
        launcher, args = self.launcher, self.args
        pending = [job for job in launcher.jobs if job.state == "pending"]
        print(f"Output root: {self.output_root}")
        print(f"Data root: {self.data_root}; splits: {self.split_csv}")
        print(f"Folds: {self.folds}; resolutions: {self.resolutions}; extractors: {args.extractors}")
        print(f"Stages: {len(launcher.jobs)} total, {len(pending)} pending")
        print(f"Status: {launcher.status_txt} (and .json); logs: {self.log_dir}", flush=True)
        final = peak = 0
        for job in pending:
            resolution = int(job.name.rsplit("_", 1)[-1]) if job.name.startswith(("train", "evaluate")) else 0
            if job.name.startswith("train"):
                final += MODEL_BYTES * (4 if args.keep_latest else 1)  # ema.pt (+ latest.pt: model, EMA, optimizer)
                peak = max(peak, 2 * 3 * MODEL_BYTES)  # latest.pt plus its temporary copy
            elif job.name.startswith("evaluate"):
                clips = 68 * 3 * 31 * resolution * resolution  # generated uint8 frames, kept until metrics.json
                final += FEATURES_BYTES_PER_EVALUATION + (clips if args.keep_samples else 0)
                peak = max(peak, clips)
        if pending and not disk_preflight(
            self.output_root,
            final_bytes=final,
            active_bytes=peak,
            margin_gb=args.disk_margin_gb,
            ignore=args.ignore_disk_check,
            what=f"{len(pending)} pending stages",
        ):
            return EXIT_DISK
        return launcher.run()


def main(argv: Sequence[str] | None = None) -> int:
    return Reproduction(_parse_args(argv)).run()


if __name__ == "__main__":
    raise SystemExit(main())
