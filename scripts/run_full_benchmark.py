"""Run the EchoXFlow benchmark matrix (Tasks 1-3) and build the table.

Every (task, method, domain, fold) cell of the table spec runs as one ``tasks.bench`` process on a GPU slot (or a CPU
slot for temporal-mean baselines). Complete cells are skipped on rerun and failing ones retried (``--max-attempts``);
a cell whose recorded config differs from the requested one is refused unless ``--force-rerun``.
``OUTPUT_DIR/status/RUN_NAME.txt`` lists every cell, a disk preflight guards the start (``--ignore-disk-check``), and
the table is rebuilt whenever a fold completes.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.utils.launcher import EXIT_DISK, Job, Launcher, child_env, disk_preflight  # noqa: E402

COMPLETE_FILE = ".complete.json"
EXIT_CONFIG_MISMATCH = 3  # tasks.bench: the cell holds another config; retrying cannot help
EXIT_LOCKED = 75  # tasks.bench: another process runs this cell
#: Disk estimates: final weights and logs of a cell, and what a running cell holds on top (training_state.pt, the staged
#: best weights and a temporary copy during a write).
FINAL_BYTES_PER_CELL = 60_000_000
ACTIVE_BYTES_PER_RUNNING_CELL = 700_000_000
#: Relative cost of one training epoch per task (clips per epoch over GPU speed relative to segmentation) and per
#: method (segmentation clips/s at 256 px on an RTX 5090), from profiling runs. Only used to start long cells first,
#: so a run does not end with one GPU busy on a slow cell while the others idle; unknown names count as slowest.
TASK_EPOCH_COST = {"segmentation": 800.0, "tissue_doppler": 270.0, "color_doppler": 1030.0}
METHOD_CLIPS_PER_SECOND = {
    "temporal_mean": 1000.0,
    "frame_unet": 15.5,
    "convnext_v2_atto": 28.5,
    "segformer_b0": 30.2,
    "dinov3_vit_s16": 20.3,
    "mambaout_femto": 27.3,
    "temporal_unet": 25.7,
    "video_swin_t": 14.7,
    "mvit_v2_s": 9.2,
    "uniformer_s_32f": 7.2,
    "u_mamba_bot": 12.8,
}


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="scripts/run_full_benchmark.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add = parser.add_argument
    add("--data-root", help="Dataset root with croissant.json (default: ECHOXFLOW_DATA_ROOT).")
    add("--output-dir", default="outputs/bench", help="Benchmark artifact root; put it on a disk with enough space.")
    add("--seed", type=int, default=0, help="Training seed.")
    add("--cv", action="store_true", help="Run 5-fold CV.")
    add("--folds", type=int, nargs="+", help="Validation folds; implies --cv.")
    add("--split-csv", help="CSV with exam_id and five_fold_split columns (default: splits.csv in the data root).")
    add("--gpus", nargs="+", help="GPUs to run cells on, e.g. 0 1 2, or 'auto' for any GPU nobody uses.")
    add("--baseline-jobs", type=int, default=10, help="CPU slots for temporal_mean cells when --gpus is set.")
    add(
        "--cpu-threads",
        type=int,
        default=os.cpu_count() or 1,
        help="CPU threads shared by the data-loader workers of the running cells; 0 disables the cap.",
    )
    add("--run-name", default=datetime.now().strftime("%Y%m%d_%H%M%S"), help="Run name (default: a timestamp).")
    add("--force-rerun", action="store_true", help="Discard and rerun every cell, also complete ones.")
    add("--smoke", action="store_true", help="Tiny model configs, 1 epoch, 10%% of the data.")
    add("--paper", action="store_true", help="Article protocol: --protocol-profile echoxflow_paper_v1, resumable.")
    add("--resolution", type=int, help="Square input and target size in pixels; appends _rN to the run name.")
    add("--table-spec", default="configs/benchmark_table.yaml", help="Rows and columns of the table.")
    add("--max-attempts", type=int, default=2, help="Attempts per cell (each retry resumes) before it is failed.")
    add("--keep-training-state", action="store_true", help="Keep each cell's training_state.pt after it completes.")
    add("--ignore-disk-check", action="store_true", help="Start even when the disk preflight finds too little space.")
    add("--disk-margin-gb", type=float, default=20.0, help="Free-space safety margin for the disk preflight.")
    add("--grace-seconds", type=float, default=60.0, help="Time running cells get to stop on SIGINT/SIGTERM.")
    args = parser.parse_args(argv)
    if args.folds:
        args.cv = True
    if args.max_attempts < 1:
        parser.error("--max-attempts must be at least 1")
    if args.resolution is not None and args.resolution <= 0:
        parser.error("--resolution must be a positive integer")
    return args


class Benchmark:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.output_root = Path(args.output_dir)
        self.run_name = str(args.run_name) + (f"_r{args.resolution}" if args.resolution else "")
        self.log_dir = self.output_root / "logs" / self.run_name
        self.folds = self._folds()
        self.configured_workers: dict[str, int] = {}
        self.cells: dict[str, tuple[str, str, str, int | None]] = {}
        self.tables_built: set[int | None] = set()
        self.table_dirty = True
        gpus = "auto" if args.gpus == ["auto"] else args.gpus
        self.launcher = Launcher(
            self._jobs(),
            title=f"Run {self.run_name}",
            status_json=self.output_root / "status" / f"{self.run_name}.json",
            max_attempts=args.max_attempts,
            grace_seconds=args.grace_seconds,
            gpus=gpus or None,
            cpu_slots=args.baseline_jobs if gpus else 0,
            no_retry_codes=(EXIT_CONFIG_MISMATCH, EXIT_LOCKED),
            skip_done=not args.force_rerun,
            on_finish=self._on_finish,
            status_extra={"run_name": self.run_name, "output_root": str(self.output_root)},
        )

    # ------------------------------------------------------------------------------------------------- setup
    def _folds(self) -> list[int | None]:
        if not self.args.cv:
            return [None]
        if self.args.folds:
            return list(self.args.folds)
        return [0, 1, 2, 3, 4]

    def _protocol_args(self) -> list[str]:
        args = ["--seed", str(self.args.seed)]
        if self.args.paper:
            args += ["--protocol-profile", "echoxflow_paper_v1", "--resume-training"]
        if self.args.resolution:
            args += ["--resolution", str(self.args.resolution)]
        return args

    def _limit_args(self) -> list[str]:
        return ["--smoke-config", "--data-fraction", "0.1", "--max-epochs", "1"] if self.args.smoke else []

    def _cell_argv(self, task: str, method: str, domain: str, fold: int | None) -> list[str]:
        """Arguments of ``python -m tasks.bench`` for one cell (without ``--force-rerun``)."""
        argv = ["--task", task, "--method", method, "--domain", domain, *self._protocol_args()]
        argv += ["--output-dir", str(self.output_root), "--run-name", self.run_name]
        for flag, value in (("--data-root", self.args.data_root), ("--split-csv", self.args.split_csv)):
            if value:
                argv += [flag, str(value)]
        if self.args.keep_training_state:
            argv.append("--keep-training-state")
        if fold is not None:
            argv += ["--validation-fold", str(fold)]
        return [*argv, "--max-val-steps", "0", *self._limit_args(), "--resume-training"]

    def _jobs(self) -> list[Job]:
        spec = yaml.safe_load(Path(self.args.table_spec).read_text(encoding="utf-8"))
        tasks = list(dict.fromkeys(str(column["task"]) for column in spec["columns"]))
        jobs = []
        for fold in self.folds:
            for task in tasks:
                for row in spec["rows"]:
                    method, domain = str(row["method"]), str(row["domain"])
                    label = f"{task}/{method}/{domain}" + ("" if fold is None else f"/fold={fold}")
                    run_dir = self.output_root / task / self.run_name / method / domain
                    run_dir = (run_dir if fold is None else run_dir / f"fold={fold}") / f"seed={self.args.seed}"
                    slug = "" if fold is None else f"_fold-{fold}"
                    self.cells[label] = (task, method, domain, fold)
                    jobs.append(
                        Job(
                            name=label,
                            commands=self._commands(task, method, domain, fold),
                            log=str(self.log_dir / f"{task}_{method}_{domain}{slug}_seed-{self.args.seed}.log"),
                            kind="cpu" if method == "temporal_mean" else "gpu",
                            done_markers=(run_dir / COMPLETE_FILE,),
                            group=fold,
                        )
                    )
        fold_order = {fold: index for index, fold in enumerate(self.folds)}
        # Longest cells first within each fold, so a run does not end with one GPU busy on a slow cell.
        return sorted(jobs, key=lambda job: (fold_order[job.group], -_relative_cost(job.name)))  # type: ignore[index]

    def _commands(
        self, task: str, method: str, domain: str, fold: int | None
    ) -> Callable[[Job, str | None], list[list[str]]]:
        def commands(job: Job, device: str | None) -> list[list[str]]:
            command = [sys.executable, "-m", "tasks.bench", *self._cell_argv(task, method, domain, fold)]
            workers = self._cell_workers(job, device)
            if workers is not None:
                command += ["--num-workers", str(workers)]
            if self.args.force_rerun and job.attempts == 1:
                command.append("--force-rerun")
            return [command]

        return commands

    def _cell_workers(self, job: Job, device: str | None) -> int | None:
        """Data-loader workers for ``job``: its configured count, capped by its share of ``--cpu-threads``."""
        budget = int(self.args.cpu_threads)
        if budget <= 0:
            return None
        launcher = self.launcher
        baselines_left = launcher.cpu_slots > 0 and any(
            other.kind == "cpu" and other.state in {"pending", "running"} for other in launcher.jobs
        )
        share = max(1, budget // (launcher.gpu_slots + (1 if baselines_left else 0)))
        if device == "cpu":
            share = max(1, share // max(1, launcher.cpu_slots))
        configured = self.configured_workers.get(job.name)
        return share if configured is None else min(configured, share)

    # --------------------------------------------------------------------------------------------------- run
    def run(self) -> int:
        launcher, args = self.launcher, self.args
        pending = [job for job in launcher.jobs if job.state == "pending"]
        print(f"Benchmark output: {self.output_root}\nRun name: {self.run_name}\nSeed: {args.seed}")
        print(f"Data root: {args.data_root or 'ECHOXFLOW_DATA_ROOT'}")
        if args.cv:
            print(f"Validation folds: {' '.join(str(fold) for fold in self.folds)}")
        print(
            f"Cells: {len(launcher.jobs)} total, {len(launcher.jobs) - len(pending)} already complete, "
            f"{len(pending)} pending; max attempts per cell: {args.max_attempts}"
        )
        print(f"Cell logs: {self.log_dir}\nStatus: {launcher.status_txt} (and .json)", flush=True)
        if not self._config_precheck():
            return EXIT_CONFIG_MISMATCH
        if (status := self._write_manifest()) != 0:
            return status
        if pending and not disk_preflight(
            self.output_root,
            final_bytes=FINAL_BYTES_PER_CELL * len(pending),
            active_bytes=ACTIVE_BYTES_PER_RUNNING_CELL * launcher.gpu_slots,
            margin_gb=args.disk_margin_gb,
            ignore=args.ignore_disk_check,
            what=f"{len(pending)} pending cells",
        ):
            return EXIT_DISK
        code = launcher.run()
        if launcher.stop_signal is None and (
            self.table_dirty or not (self.output_root / "tables" / f"{self.run_name}.tex").is_file()
        ):
            self._build_table("final")
        return code

    def _write_manifest(self) -> int:
        command = [sys.executable, str(REPO_ROOT / "scripts" / "write_run_manifest.py")]
        command += ["--output-dir", str(self.output_root), "--run-name", self.run_name]
        if self.args.data_root:
            command += ["--data-root", str(self.args.data_root)]
        if self.args.split_csv:
            command += ["--split-csv", str(self.args.split_csv)]
        command += ["--", *self._protocol_args(), "--max-val-steps", "0", *self._limit_args()]
        result = subprocess.run(command, check=False, env=child_env())
        if result.returncode != 0:
            print(f"Writing the run manifest failed with exit code {result.returncode}", file=sys.stderr)
        return result.returncode

    def _config_precheck(self) -> bool:
        """Refuse to start when a started or complete cell recorded a different config hash than it now resolves to."""
        from tasks.bench import plan_cell, recorded_config_hash

        mismatched = []
        for job in self.launcher.jobs:
            argv = self._cell_argv(*self.cells[job.name])
            try:
                planned = plan_cell(argv)
            except (Exception, SystemExit):
                continue  # an invalid cell (for example an unknown method) fails on its own when it runs
            self.configured_workers[job.name] = planned.num_workers
            recorded = recorded_config_hash(planned.run_dir)
            if recorded is not None and recorded != planned.config_hash:
                mismatched.append((job.name, recorded, planned.config_hash))
        if not mismatched:
            return True
        if self.args.force_rerun:
            print(f"--force-rerun: {len(mismatched)} cell(s) with another config will be discarded and rerun.")
            return True
        print(
            f"Refusing to start: {len(mismatched)} cell(s) of run {self.run_name!r} were run with a different "
            "configuration (resolution, protocol, step limits or config file content) than the current settings "
            "resolve to:",
            file=sys.stderr,
        )
        for name, recorded, expected in mismatched[:10]:
            print(f"  {name}: recorded {recorded}, now {expected}", file=sys.stderr)
        if len(mismatched) > 10:
            print(f"  ... and {len(mismatched) - 10} more", file=sys.stderr)
        print("Use a new --run-name, or --force-rerun to discard and rerun those cells.", file=sys.stderr)
        return False

    # ------------------------------------------------------------------------------------------------ tables
    def _on_finish(self, launcher: Launcher, job: Job) -> None:
        self.table_dirty = True
        if not self.args.cv:
            return
        busy = {other.group for other in launcher.jobs if other.state in {"pending", "running"}}
        for fold in self.folds:
            if fold in self.tables_built or fold in busy:
                continue
            self.tables_built.add(fold)
            if not all(other.previously_done for other in launcher.jobs if other.group == fold):
                self._build_table(f"after fold={fold}")

    def _build_table(self, label: str) -> None:
        print(f"Building benchmark table ({label})", flush=True)
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_benchmark_table.py"),
            "--root",
            str(self.output_root),
            "--run-name",
            self.run_name,
            "--manifest",
            str(self.output_root / f"{self.run_name}.manifest.json"),
            "--spec",
            str(self.args.table_spec),
            "--output-dir",
            str(self.output_root / "tables"),
            "--allow-missing",
        ]
        result = subprocess.run(command, check=False, env=child_env())
        if result.returncode != 0:
            print(f"WARNING: building the table failed with exit code {result.returncode}", flush=True)
        else:
            self.table_dirty = False


def _relative_cost(label: str) -> float:
    task, method = label.split("/")[:2]
    task_cost = TASK_EPOCH_COST.get(task, max(TASK_EPOCH_COST.values()))
    return task_cost / METHOD_CLIPS_PER_SECOND.get(method, min(METHOD_CLIPS_PER_SECOND.values()))


def main(argv: Sequence[str] | None = None) -> int:
    return Benchmark(_parse_args(argv)).run()


if __name__ == "__main__":
    raise SystemExit(main())
