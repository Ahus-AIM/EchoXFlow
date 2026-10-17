"""Build the EchoXFlow benchmark table with fold statistics from per-cell validation metrics.

For every row (method x domain) and column (task metric) the table reports the mean over validation folds of each
fold's examination-equal metric, "+- SD across folds", and the paired per-fold difference to a reference row (by
default the 2D U-Net of the same domain) as mean +- SD across folds. Every method of one task, domain, and fold must
have been evaluated on the same validation clips and exams; the build fails otherwise.

Only complete cells count: a cell directory with ``cell.json`` (written by ``tasks.bench``) must also have
``.complete.json``. With ``--allow-missing`` (used by the benchmark launcher while cells are still running or after
some failed) missing cells and folds are reported and shown as "--" instead of failing the build; the identical-
validation check still applies to every cell that exists, and paired differences use the folds both rows share.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tasks.utils.atomic import atomic_open, atomic_write_text  # noqa: E402
from tasks.utils.stats import bootstrap_mean, holm_adjust  # noqa: E402

SINGLE_FOLD = -1


@dataclass
class Cell:
    """All folds of one (task, method, domain) cell."""

    task: str
    method: str
    domain: str
    folds: dict[int, Mapping[str, Any]] = field(default_factory=dict)
    clip_paths: dict[int, Path] = field(default_factory=dict)
    val_exam_ids: dict[int, frozenset[str] | None] = field(default_factory=dict)


@dataclass(frozen=True)
class CellStats:
    folds: tuple[int, ...]
    values: tuple[float, ...]
    mean: float
    sd: float | None
    n_clips: tuple[int, ...]
    n_exams: tuple[int, ...]
    reference: str | None
    diff_values: tuple[float, ...] | None
    diff_mean: float | None
    diff_sd: float | None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the EchoXFlow benchmark table from validation metrics.")
    parser.add_argument("--root", default="outputs/bench")
    parser.add_argument("--output-dir", default="outputs/bench/tables", help="Directory for <run-name>.tex and .csv.")
    parser.add_argument("--precision", type=int, default=2)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--spec", default="configs/benchmark_table.yaml")
    parser.add_argument(
        "--reference",
        default=None,
        help="Reference row METHOD:DOMAIN for paired differences (default: the spec's reference_method per domain).",
    )
    parser.add_argument("--manifest", default=None, help="Run manifest JSON to cite in the table header.")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Report missing (failed or unfinished) cells and folds as missing instead of failing the build.",
    )
    parser.add_argument(
        "--domain-comparison", action="store_true", help="Write exam-paired domain tests and fold wins."
    )
    parser.add_argument("--resolution", type=int, help="Filter cells to this spatial resolution.")
    args = parser.parse_args(argv)

    spec = _load_spec(Path(args.spec))
    cells = _read_cells(Path(args.root), run_name=args.run_name, resolution=args.resolution)
    missing = _require_complete(cells, spec=spec, allow_missing=args.allow_missing)
    _require_identical_validation(cells, spec=spec)
    rows = _table_rows(cells, spec=spec, reference=args.reference, allow_missing=args.allow_missing)
    tex_path = Path(args.output_dir) / f"{args.run_name or 'table'}.tex"
    csv_path = tex_path.with_suffix(".csv")
    manifest = None if args.manifest is None or not Path(args.manifest).is_file() else Path(args.manifest)
    atomic_write_text(tex_path, _latex(rows, spec=spec, precision=args.precision, manifest=manifest, missing=missing))
    _write_csv(csv_path, rows, spec=spec)
    if args.domain_comparison:
        _domain_comparison(cells, spec=spec, output=tex_path)
    if missing:
        print(f"Missing benchmark cells ({len(missing)}): " + ", ".join(missing), file=sys.stderr)
    print(tex_path)
    print(csv_path)
    return 0


def _read_cells(
    root: Path, *, run_name: str | None = None, resolution: int | None = None
) -> dict[tuple[str, str, str], Cell]:
    cells: dict[tuple[str, str, str], Cell] = {}
    for path in sorted(root.glob("**/val_metrics.json")):
        if run_name is not None and str(run_name) not in path.parts:
            continue
        if (path.parent / "cell.json").is_file() and not (path.parent / ".complete.json").is_file():
            continue  # evaluation outputs of a cell that has not completed (for example a crashed rerun)
        if resolution is not None and _cell_resolution(path.parent) != resolution:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = (str(payload["task"]), str(payload["method"]), str(payload["domain"]))
        cell = cells.setdefault(key, Cell(*key))
        fold = int(payload["validation_fold"]) if "validation_fold" in payload else SINGLE_FOLD
        if fold in cell.folds:
            raise SystemExit(f"Duplicate results for {'/'.join(key)} fold {fold}: {path}")
        cell.folds[fold] = payload
        cell.clip_paths[fold] = path.parent / "val_clip_metrics.csv"
        cell.val_exam_ids[fold] = _val_exam_ids(path.parent / "val_clip_metrics.csv")
    return cells


def _val_exam_ids(path: Path) -> frozenset[str] | None:
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        return frozenset(str(row["exam_id"]) for row in csv.DictReader(handle))


def _load_spec(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"benchmark table spec must be a YAML mapping: {path}")
    _rows(raw)
    _columns(raw)
    return cast(dict[str, Any], raw)


def _require_complete(
    cells: Mapping[tuple[str, str, str], Cell], *, spec: Mapping[str, Any], allow_missing: bool = False
) -> list[str]:
    """Fail on missing cells or uneven fold coverage; with ``allow_missing`` return what is missing instead."""
    missing = []
    for row in _rows(spec):
        for task in _tasks(spec):
            key = (task, str(row["method"]), str(row["domain"]))
            if key not in cells or not cells[key].folds:
                missing.append("/".join(key))
    if missing and not allow_missing:
        raise SystemExit("Missing benchmark cells: " + ", ".join(missing))
    for task in _tasks(spec):
        fold_sets = {
            (str(row["method"]), str(row["domain"])): frozenset(
                cells[(task, str(row["method"]), str(row["domain"]))].folds
            )
            for row in _rows(spec)
            if (task, str(row["method"]), str(row["domain"])) in cells
        }
        all_folds = frozenset().union(*fold_sets.values()) if fold_sets else frozenset()
        if len(set(fold_sets.values())) > 1:
            details = ", ".join(f"{method}/{domain}={sorted(folds)}" for (method, domain), folds in fold_sets.items())
            if not allow_missing:
                raise SystemExit(f"Benchmark cells of {task} cover different folds: {details}")
            for (method, domain), folds in fold_sets.items():
                missing.extend(f"{task}/{method}/{domain}/fold={fold}" for fold in sorted(all_folds - folds))
    return missing


def _require_identical_validation(cells: Mapping[tuple[str, str, str], Cell], *, spec: Mapping[str, Any]) -> None:
    """Every method of one task, domain, and fold must be scored on the same number of clips and the same exams."""
    problems: list[str] = []
    for task in _tasks(spec):
        for domain in dict.fromkeys(str(row["domain"]) for row in _rows(spec)):
            methods = [
                str(row["method"])
                for row in _rows(spec)
                if str(row["domain"]) == domain and (task, str(row["method"]), domain) in cells
            ]
            folds = sorted({fold for method in methods for fold in cells[(task, method, domain)].folds})
            for fold in folds:
                present = [method for method in methods if fold in cells[(task, method, domain)].folds]
                n_clips = {method: cells[(task, method, domain)].folds[fold].get("n_val_clips") for method in present}
                if len(set(n_clips.values())) > 1:
                    problems.append(f"{task}/{domain}/fold {fold}: n_val_clips differ {n_clips}")
                exam_ids = {method: cells[(task, method, domain)].val_exam_ids.get(fold) for method in present}
                missing = sorted(method for method, ids in exam_ids.items() if ids is None)
                if missing:
                    problems.append(f"{task}/{domain}/fold {fold}: no val_clip_metrics.csv for {', '.join(missing)}")
                elif len(set(exam_ids.values())) > 1:
                    problems.append(f"{task}/{domain}/fold {fold}: validation exam ids differ between methods")
    if problems:
        raise SystemExit("Benchmark cells were not evaluated on identical validation data:\n  " + "\n  ".join(problems))


def _table_rows(
    cells: Mapping[tuple[str, str, str], Cell],
    *,
    spec: Mapping[str, Any],
    reference: str | None,
    allow_missing: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_spec in _rows(spec):
        method = str(row_spec["method"])
        domain = str(row_spec["domain"])
        ref_method, ref_domain = _reference_row(spec, domain=domain, override=reference)
        row: dict[str, Any] = {
            "label": str(row_spec["label"]),
            "input_domain": str(row_spec["input_domain"]),
            "method": method,
            "domain": domain,
        }
        for column in _columns(spec):
            task = str(column["task"])
            cell = cells.get((task, method, domain))
            row[str(column["name"])] = (
                None
                if cell is None or not cell.folds
                else _cell_stats(
                    cell,
                    metric=str(column["metric"]),
                    reference_cell=cells.get((task, ref_method, ref_domain)),
                    reference_label=f"{ref_method}:{ref_domain}",
                    common_folds_only=allow_missing,
                )
            )
        rows.append(row)
    return rows


def _reference_row(spec: Mapping[str, Any], *, domain: str, override: str | None) -> tuple[str, str]:
    if override:
        method, _, ref_domain = str(override).partition(":")
        return method, ref_domain or domain
    return str(spec.get("reference_method", "frame_unet")), domain


def _cell_stats(
    cell: Cell,
    *,
    metric: str,
    reference_cell: Cell | None,
    reference_label: str,
    common_folds_only: bool = False,
) -> CellStats:
    folds = tuple(sorted(cell.folds))
    values = tuple(_fold_metric(cell, fold, metric) for fold in folds)
    diff_values = None
    if reference_cell is not None and tuple(sorted(reference_cell.folds)) != folds and not common_folds_only:
        raise SystemExit(f"Reference {reference_label} does not cover the folds of {cell.method}/{cell.domain}")
    shared = [(fold, value) for fold, value in zip(folds, values, strict=True)]
    if reference_cell is not None:
        shared = [(fold, value) for fold, value in shared if fold in reference_cell.folds]
    if reference_cell is not None and shared:
        diff_values = tuple(value - _fold_metric(reference_cell, fold, metric) for fold, value in shared)
    return CellStats(
        folds=folds,
        values=values,
        mean=math.fsum(values) / len(values),
        sd=_sample_sd(values),
        n_clips=tuple(int(cell.folds[fold].get("n_val_clips", 0)) for fold in folds),
        n_exams=tuple(int(cell.folds[fold].get("n_val_exams", 0)) for fold in folds),
        reference=None if reference_cell is None else reference_label,
        diff_values=diff_values,
        diff_mean=None if diff_values is None else math.fsum(diff_values) / len(diff_values),
        diff_sd=None if diff_values is None else _sample_sd(diff_values),
    )


def _fold_metric(cell: Cell, fold: int, metric: str) -> float:
    metrics = cell.folds[fold].get("metrics", {})
    if not isinstance(metrics, Mapping) or metric not in metrics:
        raise SystemExit(f"Missing metric {metric!r} for {cell.task}/{cell.method}/{cell.domain} fold {fold}")
    return float(metrics[metric])


def _latex(
    rows: Sequence[Mapping[str, Any]],
    *,
    spec: Mapping[str, Any],
    precision: int,
    manifest: Path | None,
    missing: Sequence[str] = (),
) -> str:
    columns = _columns(spec)
    best: dict[str, float] = {}
    for column in columns:
        name = str(column["name"])
        values = [_display(row[name].mean, column) for row in rows if row[name] is not None]
        if values:
            best[name] = max(values) if str(column.get("best", "min")) == "max" else min(values)
    lines = [f"% {line}" for line in _header_lines(spec, manifest=manifest)]
    if missing:
        lines.append(f"% MISSING ({len(missing)}, shown as --): " + ", ".join(missing))
    lines.append("% Columns: " + " | ".join(str(column.get("title", column["name"])) for column in columns))
    for row in rows:
        cells = [str(row["label"]), str(row["input_domain"])]
        for column in columns:
            name = str(column["name"])
            stats: CellStats | None = row[name]
            if stats is None:
                cells.append("--")
                continue
            digits = int(column.get("precision", precision))
            value = _display(stats.mean, column)
            text = _fmt(value, digits)
            if value == best.get(name):
                text = f"\\textbf{{{text}}}"
            if stats.sd is not None:
                text = f"{text} $\\pm$ {_fmt(_display(stats.sd, column), digits)}"
            cells.append(text)
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("")
    lines.append("% Paired per-fold differences to the reference row (mean $\\pm$ SD across folds)")
    for row in rows:
        reference = next(
            (row[str(column["name"])].reference for column in columns if row[str(column["name"])] is not None), None
        )
        cells = [str(row["label"]), str(row["input_domain"])]
        for column in columns:
            stats = row[str(column["name"])]
            digits = int(column.get("precision", precision))
            if stats is None or stats.diff_mean is None:
                cells.append("--")
                continue
            text = f"{_display(stats.diff_mean, column):+.{digits}f}"
            if stats.diff_sd is not None:
                text = f"{text} $\\pm$ {_fmt(_display(stats.diff_sd, column), digits)}"
            cells.append(text)
        lines.append(" & ".join(cells) + f" \\\\ % vs {reference}")
    return "\n".join(lines) + "\n"


def _header_lines(spec: Mapping[str, Any], *, manifest: Path | None) -> list[str]:
    protocol = spec.get("protocol", {})
    lines = ["Values: mean over validation folds of per-fold examination-equal metrics, +- SD across folds."]
    if isinstance(protocol, Mapping) and protocol:
        lines.append("Protocol: " + ", ".join(f"{key}={value}" for key, value in protocol.items()))
    if manifest is not None:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        git = payload.get("git", {})
        dataset = payload.get("dataset", {})
        lines.append(
            f"Run manifest: {manifest.name} git={git.get('sha')} dirty={git.get('dirty')} "
            f"croissant={dataset.get('croissant_version')} schema={dataset.get('store_schema_version')} "
            f"config_hash={payload.get('config_hash')}"
        )
    return lines


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], *, spec: Mapping[str, Any]) -> None:
    fieldnames = [
        "method",
        "domain",
        "label",
        "column",
        "task",
        "metric",
        "display_scale",
        "n_folds",
        "folds",
        "mean",
        "sd_across_folds",
        "fold_values",
        "fold_n_val_clips",
        "fold_n_val_exams",
        "reference",
        "diff_mean",
        "diff_sd_across_folds",
        "fold_diffs",
    ]
    with atomic_open(path, "w", encoding="utf-8", newline="") as stream:
        _write_csv_rows(stream, rows, spec=spec, fieldnames=fieldnames)


def _write_csv_rows(
    stream: Any, rows: Sequence[Mapping[str, Any]], *, spec: Mapping[str, Any], fieldnames: list[str]
) -> None:
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        for column in _columns(spec):
            stats: CellStats | None = row[str(column["name"])]
            if stats is None:
                writer.writerow(
                    {
                        "method": row["method"],
                        "domain": row["domain"],
                        "label": row["label"],
                        "column": column["name"],
                        "task": column["task"],
                        "metric": column["metric"],
                        "n_folds": 0,
                    }
                )
                continue
            writer.writerow(
                {
                    "method": row["method"],
                    "domain": row["domain"],
                    "label": row["label"],
                    "column": column["name"],
                    "task": column["task"],
                    "metric": column["metric"],
                    "display_scale": float(column.get("display_scale", 1.0)),
                    "n_folds": len(stats.folds),
                    "folds": ";".join(str(fold) for fold in stats.folds),
                    "mean": repr(stats.mean),
                    "sd_across_folds": "" if stats.sd is None else repr(stats.sd),
                    "fold_values": ";".join(repr(value) for value in stats.values),
                    "fold_n_val_clips": ";".join(str(value) for value in stats.n_clips),
                    "fold_n_val_exams": ";".join(str(value) for value in stats.n_exams),
                    "reference": stats.reference or "",
                    "diff_mean": "" if stats.diff_mean is None else repr(stats.diff_mean),
                    "diff_sd_across_folds": "" if stats.diff_sd is None else repr(stats.diff_sd),
                    "fold_diffs": "" if stats.diff_values is None else ";".join(map(repr, stats.diff_values)),
                }
            )


def _cell_resolution(directory: Path) -> int:
    config = yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8"))
    shape = config["data"]["input_spatial_shape"]
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(f"expected square input resolution: {directory}")
    return int(shape[0])


def _exam_values(cell: Cell, metric: str) -> dict[str, float]:
    """Pool out-of-fold clips, average within each exam, and reject exams occurring in multiple folds."""
    result = {}
    for fold, path in sorted(cell.clip_paths.items()):
        grouped: dict[str, list[float]] = {}
        samples = set()
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if not {"exam_id", "sample_id", metric}.issubset(reader.fieldnames or []):
                raise ValueError(f"{path} needs exam_id, sample_id and {metric}")
            for row in reader:
                exam = row["exam_id"].strip()
                if not exam or row["sample_id"] in samples:
                    raise ValueError(f"empty exam or duplicate clip in {path}")
                samples.add(row["sample_id"])
                value = float(row[metric] or "nan")
                grouped.setdefault(exam, [])
                if math.isfinite(value):
                    grouped[exam].append(value)
        if result.keys() & grouped.keys():
            raise ValueError(f"{cell.task}/{cell.method}/{cell.domain}: an exam appears in multiple folds")
        result.update(
            {exam: math.fsum(values) / len(values) if values else math.nan for exam, values in grouped.items()}
        )
    return result


def _domain_comparison(cells: Mapping[tuple[str, str, str], Cell], *, spec: Mapping[str, Any], output: Path) -> None:
    resolutions = {_cell_resolution(path.parent) for cell in cells.values() for path in cell.clip_paths.values()}
    if len(resolutions) != 1:
        raise ValueError("domain comparison needs one resolution; pass --resolution")
    resolution = next(iter(resolutions))
    # The temporal-mean baseline is not an architecture comparison (five framewise and five time-aware methods).
    framewise = {"frame_unet", "convnext_v2_atto", "segformer_b0", "dinov3_vit_s16", "mambaout_femto"}
    methods = list(dict.fromkeys(str(row["method"]) for row in _rows(spec) if row["method"] != "temporal_mean"))
    rows = []
    for column in _columns(spec):
        task, metric = str(column["task"]), str(column["metric"])
        family = []
        for method in methods:
            beam, cart = (cells.get((task, method, domain)) for domain in ("beamspace", "cartesian"))
            if beam is None or cart is None:
                continue
            if set(beam.folds) != set(cart.folds):
                raise ValueError(f"domain fold coverage differs: {task}/{method}")
            wins = {"beamspace_wins": 0, "scan_converted_wins": 0, "ties": 0}
            for fold in beam.folds:
                if beam.val_exam_ids[fold] != cart.val_exam_ids[fold]:
                    raise ValueError(f"domain validation exams differ: {task}/{method}/fold={fold}")
                delta = _fold_metric(beam, fold, metric) - _fold_metric(cart, fold, metric)
                if not math.isfinite(delta):
                    raise ValueError(f"undefined fold metric: {task}/{method}/{metric}")
                winner = (
                    "ties"
                    if delta == 0
                    else (
                        "beamspace_wins"
                        if (delta > 0) == (column.get("best", "min") == "max")
                        else "scan_converted_wins"
                    )
                )
                wins[winner] += 1
            b, c = _exam_values(beam, metric), _exam_values(cart, metric)
            if b.keys() != c.keys():
                raise ValueError(f"domain exam cohorts differ: {task}/{method}")
            exams = sorted(exam for exam in b if math.isfinite(b[exam]) and math.isfinite(c[exam]))
            stats = bootstrap_mean(
                [b[exam] - c[exam] for exam in exams], namespace=f"domain:{resolution}:{task}:{metric}"
            )
            scale = float(column.get("display_scale", 1))
            family.append(
                {
                    "resolution": resolution,
                    "task": task,
                    "metric": metric,
                    "method": method,
                    "group": "framewise" if method in framewise else "time-aware",
                    "n_exams": len(exams),
                    "excluded_exams": len(b) - len(exams),
                    "n_folds": len(beam.folds),
                    "display_scale": scale,
                    **wins,
                    **{key: value * scale if key != "p_two_sided" else value for key, value in stats.items()},
                    "replicates": 2000,
                    "seed": 17,
                }
            )
        for row, adjusted in zip(family, holm_adjust([row["p_two_sided"] for row in family]), strict=True):
            row["p_holm"] = adjusted
            row["direction"] = (
                "n.s."
                if adjusted >= 0.05
                else ("beamspace" if (row["mean"] > 0) == (column.get("best", "min") == "max") else "scan-converted")
            )
        rows.extend(family)
        for group in ("framewise", "time-aware", "total"):
            selected = [row for row in family if group == "total" or row["group"] == group]
            rows.append(
                {
                    "resolution": resolution,
                    "task": task,
                    "metric": metric,
                    "method": "ALL",
                    "group": group,
                    **{
                        key: sum(row[key] for row in selected)
                        for key in ("beamspace_wins", "scan_converted_wins", "ties")
                    },
                }
            )
    path = output.with_name(f"{output.stem}.domain_comparison.r{resolution}.csv")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "% Difference: beamspace - scan-converted, in display units. 2000 exam bootstrap replicates; seed 17.",
        "% Holm within task metric and resolution. ALL rows count methods x available folds (ties separate).",
    ]
    for row in rows:
        stats_text = (
            "-- & -- & --"
            if row["method"] == "ALL"
            else (
                f"{row['mean']:+.3f} [{row['ci95_low']:+.3f}, {row['ci95_high']:+.3f}] & "
                f"{row['p_holm']:.4f} & {row['direction']}"
            )
        )
        label = str(row["method"]).replace("_", r"\_")
        metric_label = str(row["metric"]).replace("_", r"\_")
        lines.append(
            f"{metric_label} & {label} & {row['group']} & {row['beamspace_wins']} & "
            f"{row['scan_converted_wins']} & {row['ties']} & {stats_text} " + r"\\"
        )
    atomic_write_text(path.with_suffix(".tex"), "\n".join(lines) + "\n")
    print(path)


def _rows(spec: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    rows = spec.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("benchmark table spec requires a non-empty `rows` list")
    return tuple(cast(Mapping[str, Any], row) for row in rows)


def _columns(spec: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    columns = spec.get("columns")
    if not isinstance(columns, list) or not columns:
        raise ValueError("benchmark table spec requires a non-empty `columns` list")
    return tuple(cast(Mapping[str, Any], column) for column in columns)


def _tasks(spec: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(column["task"]) for column in _columns(spec)))


def _display(value: float, column: Mapping[str, Any]) -> float:
    return float(value) * float(column.get("display_scale", 1.0))


def _fmt(value: float, precision: int) -> str:
    return f"{float(value):.{int(precision)}f}"


def _sample_sd(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = math.fsum(values) / len(values)
    return math.sqrt(math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1))


if __name__ == "__main__":
    raise SystemExit(main())
