"""Cross-validated beamspace-versus-Cartesian table, pooled from the per-fold :mod:`tasks.video_diffusion.tables`.

Reads ``<root>/fold<k>/<tables-name>/comparison.json`` and ``comparison_replicates.npz`` for every fold and reports, per
resolution and feature space:

- the beamspace and Cartesian distances as the mean over folds +- SD across folds;
- the cross-validated difference ``mean_k (D_cartesian,k - D_beamspace,k)`` and its relative difference
  ``100 x difference / mean_k D_cartesian,k`` (positive favours beamspace);
- a two-sided, fold-stratified exam-cluster bootstrap test of that difference. Validation folds hold disjoint exams, so
  each fold's 2,000 paired exam-bootstrap replicates are independent of the other folds'. Replicate ``r`` of the pooled
  difference averages one replicate per fold, drawn through a fixed per-fold permutation so that equal replicate indices
  of different folds are not coupled. The p-value is the recentered two-sided bootstrap p-value; Holm's step-down
  correction runs across all reported feature-space x resolution comparisons, as in the per-fold tables;
- how many folds favour each arm, and how many of those are significant after the per-fold Holm correction.

Every fold must have finished every comparison with the same replicate count and seed; otherwise the build fails.

Example::

    uv run python -m tasks.video_diffusion.cv_table --root outputs/video_diffusion --folds 0 1 2 3 4 \\
        --output-dir outputs/video_diffusion/tables
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from tasks.utils.stats import holm_adjust, recentered_p_value, seeded_permutation
from tasks.video_diffusion.tables import replicate_key

ALPHA = 0.05


def load_fold(directory: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    comparison, replicates = directory / "comparison.json", directory / "comparison_replicates.npz"
    for path in (comparison, replicates):
        if not path.is_file():
            raise SystemExit(f"missing {path}; rebuild this fold's tables with tasks.video_diffusion.tables")
    payload = json.loads(comparison.read_text(encoding="utf-8"))
    with np.load(replicates) as data:
        deltas = {key: np.asarray(data[key], dtype=np.float64) for key in data.files}
    return payload, deltas


def pool(folds: dict[int, tuple[dict[str, Any], dict[str, np.ndarray]]]) -> tuple[list[dict[str, Any]], int, int]:
    settings = {(payload["replicates"], payload["seed"]) for payload, _ in folds.values()}
    if len(settings) != 1:
        raise SystemExit(f"folds were built with different bootstrap settings: {sorted(settings)}")
    replicates, seed = next(iter(settings))
    keyed = {
        fold: {(int(row["resolution"]), str(row["extractor"])): row for row in payload["rows"]}
        for fold, (payload, _) in folds.items()
    }
    comparisons = {key for fold_rows in keyed.values() for key in fold_rows}
    for fold, fold_rows in keyed.items():
        if missing := sorted(comparisons - fold_rows.keys()):
            raise SystemExit(f"fold {fold} lacks comparisons {missing}")
    permutations = {fold: seeded_permutation(replicates, seed=seed, namespace=f"cv-table:fold{fold}") for fold in folds}
    order = [(row["resolution"], row["extractor"]) for row in folds[min(folds)][0]["rows"]]
    rows: list[dict[str, Any]] = []
    for resolution, extractor in order:
        per_fold = {fold: keyed[fold][(resolution, extractor)] for fold in sorted(folds)}
        draws = []
        for fold, row in per_fold.items():
            deltas = folds[fold][1].get(replicate_key(resolution, extractor))
            if deltas is None or deltas.shape != (replicates,) or not np.isfinite(deltas).all():
                raise SystemExit(f"fold {fold}: bad or missing replicates for {resolution} {extractor}")
            draws.append(deltas[permutations[fold]])
        pooled = np.mean(draws, axis=0)
        beam = np.array([row["beamspace"] for row in per_fold.values()], dtype=np.float64)
        cart = np.array([row["cartesian"] for row in per_fold.values()], dtype=np.float64)
        fold_deltas = cart - beam
        delta = float(fold_deltas.mean())
        cart_mean = float(cart.mean())
        significant = [float(row["p_holm"]) < ALPHA for row in per_fold.values()]
        rows.append(
            {
                "resolution": resolution,
                "extractor": extractor,
                "label": next(iter(per_fold.values()))["label"],
                "n_folds": len(per_fold),
                "folds": ";".join(str(fold) for fold in per_fold),
                "beamspace_mean": float(beam.mean()),
                "beamspace_sd_across_folds": _sd(beam),
                "cartesian_mean": cart_mean,
                "cartesian_sd_across_folds": _sd(cart),
                "delta_cartesian_minus_beamspace": delta,
                "delta_sd_across_folds": _sd(fold_deltas),
                "relative_difference_percent": 100.0 * delta / abs(cart_mean) if cart_mean else float("nan"),
                "ci95_low": float(np.quantile(pooled, 0.025)),
                "ci95_high": float(np.quantile(pooled, 0.975)),
                "probability_beamspace_better": float(
                    (np.count_nonzero(pooled > 0) + 0.5 * np.count_nonzero(pooled == 0)) / len(pooled)
                ),
                "p_two_sided": recentered_p_value(pooled, delta),
                "folds_beamspace_better": int(np.count_nonzero(fold_deltas > 0)),
                "folds_cartesian_better": int(np.count_nonzero(fold_deltas < 0)),
                "folds_significant_beamspace": sum(
                    bool(s and d > 0) for s, d in zip(significant, fold_deltas, strict=True)
                ),
                "folds_significant_cartesian": sum(
                    bool(s and d < 0) for s, d in zip(significant, fold_deltas, strict=True)
                ),
                "fold_beamspace": ";".join(f"{value:.6g}" for value in beam),
                "fold_cartesian": ";".join(f"{value:.6g}" for value in cart),
                "fold_p_holm": ";".join(f"{float(row['p_holm']):.4g}" for row in per_fold.values()),
            }
        )
    for row, adjusted in zip(rows, holm_adjust([row["p_two_sided"] for row in rows]), strict=True):
        row["p_holm"] = adjusted
        row["direction"] = (
            "n.s."
            if adjusted >= ALPHA
            else ("beamspace" if row["delta_cartesian_minus_beamspace"] > 0 else "cartesian")
        )
    return rows, replicates, seed


def _sd(values: np.ndarray) -> float:
    return float(values.std(ddof=1)) if len(values) > 1 else float("nan")


def _number(value: float) -> str:
    if not math.isfinite(value):
        return "--"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}" if abs(value) >= 1 else f"{value:.3g}"


def _sd_text(mean: float, sd: float) -> str:
    """SD with as many decimals as the mean's display."""
    text = _number(mean)
    decimals = len(text.split(".")[1]) if "." in text else 0
    return f"{sd:.{decimals}f}" if math.isfinite(sd) else "--"


def _mean_sd(row: dict[str, Any], arm: str, plus_minus: str) -> str:
    mean, sd = row[f"{arm}_mean"], row[f"{arm}_sd_across_folds"]
    return f"{_number(mean)} {plus_minus} {_sd_text(mean, sd)}"


def _p_text(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _write_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_outputs(rows: list[dict[str, Any]], output_dir: Path, *, stem: str, replicates: int, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = (
        f"mean over {rows[0]['n_folds']} validation folds +- SD across folds; difference = D_cartesian - D_beamspace "
        f"averaged over folds (positive favours beamspace); fold-stratified paired exam-cluster bootstrap, "
        f"{replicates} replicates per fold, seed {seed}; recentered two-sided p, Holm across {len(rows)} comparisons"
    )
    _write_atomic(
        output_dir / f"{stem}.json",
        json.dumps({"protocol": protocol, "replicates": replicates, "seed": seed, "rows": rows}, indent=2) + "\n",
    )
    with (output_dir / f"{stem}.csv.tmp").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(output_dir / f"{stem}.csv.tmp", output_dir / f"{stem}.csv")

    lines = ["# ECG-conditioned generation quality, cross-validated", "", protocol + ".", ""]
    for resolution in sorted({row["resolution"] for row in rows}):
        lines += [
            f"## {resolution} x {resolution}",
            "",
            "| Feature space and metric | Beamspace | Scan-converted | Relative difference | Holm-adjusted p "
            "| Folds favouring beamspace (significant) | Folds favouring scan-converted (significant) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in (r for r in rows if r["resolution"] == resolution):
            lines.append(
                f"| {row['label']} "
                f"| {_mean_sd(row, 'beamspace', '±')} | {_mean_sd(row, 'cartesian', '±')} "
                f"| {row['relative_difference_percent']:+.1f}% | {_p_text(row['p_holm'])} "
                f"| {row['folds_beamspace_better']} ({row['folds_significant_beamspace']}) "
                f"| {row['folds_cartesian_better']} ({row['folds_significant_cartesian']}) |"
            )
        lines.append("")
    _write_atomic(output_dir / f"{stem}.md", "\n".join(lines))

    tex = [
        f"% Task 4: {protocol}.",
        "% Columns: resolution | feature space (metric) | beamspace | scan-converted | relative difference | "
        "Holm-adjusted p | folds favouring beamspace (significant per fold)",
        "% Lower distances are better; the better arm is bold where the cross-validated difference is significant.",
    ]
    for row in rows:
        beam, cart = _mean_sd(row, "beamspace", "$\\pm$"), _mean_sd(row, "cartesian", "$\\pm$")
        if row["direction"] == "beamspace":
            beam = "\\textbf{" + beam.replace(" $\\pm$", "} $\\pm$", 1)
        elif row["direction"] == "cartesian":
            cart = "\\textbf{" + cart.replace(" $\\pm$", "} $\\pm$", 1)
        p = "$<$0.001" if row["p_holm"] < 0.001 else f"{row['p_holm']:.3f}"
        tex.append(
            f"{row['resolution']} & {row['label']} & {beam} & {cart} & "
            f"{row['relative_difference_percent']:+.1f}\\% & {p} & "
            f"{row['folds_beamspace_better']}/{row['n_folds']} ({row['folds_significant_beamspace']}) \\\\"
        )
    _write_atomic(output_dir / f"{stem}.tex", "\n".join(tex) + "\n")
    print("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="Task 4 output root holding fold<k>/")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--tables-name", default="tables", help="per-fold table directory (tables_camus for the CAMUS evaluation)"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default="comparison_cv", help="output file name without extension")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if len(set(args.folds)) < 2:
        raise SystemExit("a cross-validated table needs at least two folds")
    folds = {fold: load_fold(args.root / f"fold{fold}" / args.tables_name) for fold in sorted(set(args.folds))}
    rows, replicates, seed = pool(folds)
    write_outputs(rows, args.output_dir, stem=args.stem, replicates=replicates, seed=seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
