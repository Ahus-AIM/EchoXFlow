"""Plot per-epoch training and selection/validation loss curves of every benchmark cell as convergence evidence.

One figure per (task, resolution), one subplot per method (methods and labels from configs/benchmark_table.yaml).
Beamspace and scan-converted get one colour each; every fold is a thin line and the fold mean a thick one. Train loss
(``train_loss`` of ``train_scores.jsonl``) is solid; the dashed curve is ``selection_loss`` (the early-stopping monitor
of the v2 protocol) or, for older runs without it, ``val_loss``. The best epoch (``best_epoch`` of
``early_stopping.json``) is marked on the dashed curve. Cells: ``<root>/<task>/<run_name>[_r<res>]/<method>/<domain>/
fold=<k>/seed=0/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

DOMAINS = {"beamspace": ("Beamspace", "#0072B2"), "cartesian": ("Scan-converted", "#D55E00")}


def read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_fold(cell: Path) -> tuple[dict[int, float], dict[int, float], int | None]:
    """Per-epoch train loss, per-epoch selection (else validation) loss, and the best epoch of one fold."""
    train: dict[int, float] = {}
    held: dict[int, float] = {}
    for row in read_rows(cell / "train_scores.jsonl"):
        epoch = int(row["epoch"])
        train[epoch] = float(row["train_loss"])
        value = row.get("selection_loss")
        value = row.get("val_loss") if value is None else value
        if value is not None:
            held[epoch] = float(value)
    best = None
    stop = cell / "early_stopping.json"
    if stop.is_file():
        best = json.loads(stop.read_text(encoding="utf-8")).get("best_epoch")
    return train, held, None if best is None else int(best)


CSV_COLUMNS = ["task", "resolution", "method", "domain", "fold", "epoch", "train_loss", "selection_loss", "val_loss"]


def write_csv(path: Path, directory: Path, task: str, resolution: int | None, methods: list[dict]) -> None:
    """The plotted data, one row per cell (method, domain, fold) and epoch; missing values are left empty."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, [*CSV_COLUMNS, "best_epoch"])
        writer.writeheader()
        for method in methods:
            for domain in DOMAINS:
                for cell in sorted((directory / method["method"] / domain).glob("fold=*/seed=0")):
                    best = load_fold(cell)[2]
                    epochs = {int(r["epoch"]): r for r in read_rows(cell / "train_scores.jsonl")}
                    for epoch, row in sorted(epochs.items()):
                        ids = {"task": task, "resolution": resolution, "method": method["method"], "domain": domain}
                        ids |= {"fold": cell.parent.name.removeprefix("fold="), "epoch": epoch}
                        writer.writerow({k: row.get(k) for k in CSV_COLUMNS} | ids | {"best_epoch": best})


def mean_curve(curves: list[dict[int, float]]) -> tuple[list[int], list[float]]:
    epochs = sorted({e for curve in curves for e in curve})
    means = [float(np.mean([c[e] for c in curves if e in c])) for e in epochs]
    return epochs, means


def run_dir(root: Path, task: str, run_name: str, resolution: int | None) -> Path | None:
    names = [run_name] if resolution is None else [f"{run_name}_r{resolution}", run_name]
    return next((root / task / n for n in names if (root / task / n).is_dir()), None)


def plot_task(root: Path, task: str, run_name: str, resolution: int | None, methods: list[dict], out_dir: Path) -> None:
    directory = run_dir(root, task, run_name, resolution)
    if directory is None:
        print(f"skip {task}: no run directory for {run_name} r{resolution}")
        return
    cols = 5
    fig, axes = plt.subplots(
        math.ceil(len(methods) / cols), cols, figsize=(3.2 * cols, 2.6 * math.ceil(len(methods) / cols))
    )
    for ax, method in zip(axes.ravel(), methods, strict=False):
        ax.set_title(method["label"], fontsize=9)
        for domain, (name, color) in DOMAINS.items():
            folds = [load_fold(c) for c in sorted((directory / method["method"] / domain).glob("fold=*/seed=0"))]
            for style, index in (("-", 0), ("--", 1)):
                curves = [f[index] for f in folds if f[index]]
                for curve in curves:
                    ax.plot(*zip(*sorted(curve.items()), strict=True), style, color=color, lw=0.6, alpha=0.4)
                if curves:
                    ax.plot(*mean_curve(curves), style, color=color, lw=1.8, label=name if index == 0 else None)
            for _, held, best in folds:
                if best in held:
                    ax.plot(best, held[best], "o", color=color, ms=3, mfc="none")
        if not ax.lines:
            ax.text(0.5, 0.5, "no training log", ha="center", va="center", transform=ax.transAxes, fontsize=8)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(methods) :]:
        ax.axis("off")
    for ax in axes.ravel():
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            break
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.suptitle(
        f"{task}{'' if resolution is None else f' ({resolution} px)'}: solid train, dashed selection/validation loss; "
        "thin = folds, thick = mean, circle = best epoch",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"losses_{task}" + ("" if resolution is None else f"_r{resolution}")
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{suffix}", dpi=200)
    plt.close(fig)
    write_csv(out_dir / f"{stem}.csv", directory, task, resolution, methods)
    print(f"wrote {out_dir / stem}.png/.pdf/.csv")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot training and validation loss curves of benchmark cells.")
    parser.add_argument("--benchmark-root", required=True, type=Path, help="Output directory of the benchmark run.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resolution", type=int, default=None, help="Appends _r<res> to the run name if present.")
    parser.add_argument("--tasks", nargs="+", default=["segmentation", "color_doppler", "tissue_doppler"])
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark_table.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures"), help="Directory for the figures.")
    args = parser.parse_args(argv)
    rows = yaml.safe_load(args.config.read_text(encoding="utf-8"))["rows"]
    methods = list({r["method"]: {"method": r["method"], "label": r["label"]} for r in rows}.values())
    for task in args.tasks:
        plot_task(args.benchmark_root, task, args.run_name, args.resolution, methods, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
