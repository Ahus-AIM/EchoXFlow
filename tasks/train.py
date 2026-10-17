from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from tasks.registry import available_task_names, task_run_training


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one EchoXFlow example task.")
    parser.add_argument("task", choices=[name for name in available_task_names() if name != "video_diffusion"])
    parser.add_argument("--config", dest="config_path")
    parser.add_argument("--data-root", dest="data_root")
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-val-steps", type=int, default=None)
    parser.add_argument("--no-epoch-log", action="store_true")
    parser.add_argument(
        "--split-csv",
        help="CSV with exam_id and five_fold_split columns (default: splits.csv inside the data root, if present).",
    )
    parser.add_argument("--validation-fold", type=int, help="Fold held out for validation when a split file is used.")
    args = parser.parse_args(argv)

    if args.data_root is not None:
        croissant_path = Path(args.data_root) / "croissant.json"
        if not croissant_path.exists():
            print(f"Could not find croissant.json at {croissant_path}", file=sys.stderr)
            print("Set the data location by one of:", file=sys.stderr)
            print("  set ECHOXFLOW_DATA_ROOT=/path/to/EchoXFlow", file=sys.stderr)
            print("  set data.root_dir in the task YAML config", file=sys.stderr)
            return 2

    result = task_run_training(args.task)(
        config_path=args.config_path,
        data_root=args.data_root,
        max_train_steps=args.max_train_steps,
        max_val_steps=args.max_val_steps,
        write_epoch_log=not args.no_epoch_log,
        config_overrides={"data": _data_overrides(args)},
    )
    log_value = result.log_path if result.log_path is not None else "disabled"
    print(f"{args.task}: epochs={result.epochs_completed} steps={result.steps_completed} loss={result.loss:.6g}")
    print(f"log={log_value}")
    return 0


def _data_overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if args.split_csv is not None:
        overrides["fold_split_path"] = str(Path(args.split_csv).expanduser().resolve())
    if args.validation_fold is not None:
        overrides["validation_fold"] = int(args.validation_fold)
    return overrides


if __name__ == "__main__":
    raise SystemExit(main())
