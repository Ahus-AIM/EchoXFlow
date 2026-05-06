from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import yaml


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the EchoXFlow benchmark table from validation metrics.")
    parser.add_argument("--root", default="outputs/bench")
    parser.add_argument("--out", default="outputs/bench/table.tex")
    parser.add_argument("--csv-out", default=None)
    parser.add_argument("--precision", type=int, default=2)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--spec", default="configs/benchmark_table.yaml")
    args = parser.parse_args(argv)

    root = Path(args.root)
    spec = _load_spec(Path(args.spec))
    entries = _read_entries(root, run_name=args.run_name)
    _require_complete(entries, spec=spec)
    tex_path = Path(args.out)
    csv_path = Path(args.csv_out) if args.csv_out else tex_path.with_suffix(".csv")
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(_latex(entries, spec=spec, precision=args.precision), encoding="utf-8")
    _write_csv(csv_path, entries, spec=spec)
    print(tex_path)
    print(csv_path)
    return 0


def _read_entries(root: Path, *, run_name: str | None = None) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    entries: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    fold_entries: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for path in sorted(root.glob("**/val_metrics.json")):
        if run_name is not None and str(run_name) not in path.parts:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = (str(payload["task"]), str(payload["method"]), str(payload["domain"]))
        if "validation_fold" in payload:
            fold_entries.setdefault(key, []).append(payload)
        else:
            entries[key] = payload
    for key, payloads in fold_entries.items():
        entries[key] = _aggregate_fold_entries(payloads)
    return entries


def _aggregate_fold_entries(payloads: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not payloads:
        raise ValueError("cannot aggregate an empty fold payload list")
    first = payloads[0]
    metric_names = sorted(
        {str(name) for payload in payloads for name in cast(Mapping[str, Any], payload.get("metrics", {})).keys()}
    )
    metrics: dict[str, float] = {}
    metric_stds: dict[str, float] = {}
    for name in metric_names:
        values = [float(cast(Mapping[str, Any], payload.get("metrics", {}))[name]) for payload in payloads]
        mean = sum(values) / len(values)
        metrics[name] = mean
        metric_stds[name] = _sample_std(values, mean=mean)
    folds = sorted(int(cast(Any, payload["validation_fold"])) for payload in payloads)
    return {
        "task": first.get("task"),
        "method": first.get("method"),
        "domain": first.get("domain"),
        "seed": first.get("seed", ""),
        "run_name": first.get("run_name", ""),
        "metrics": metrics,
        "metric_stds": metric_stds,
        "n_val_clips": sum(int(cast(Any, payload.get("n_val_clips", 0))) for payload in payloads),
        "n_folds": len(payloads),
        "validation_folds": folds,
        "git_sha": first.get("git_sha", ""),
        "wallclock_s": round(sum(float(cast(Any, payload.get("wallclock_s", 0.0))) for payload in payloads), 2),
    }


def _load_spec(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"benchmark table spec must be a YAML mapping: {path}")
    _rows(raw)
    _columns(raw)
    return cast(dict[str, Any], raw)


def _require_complete(entries: Mapping[tuple[str, str, str], Mapping[str, Any]], *, spec: Mapping[str, Any]) -> None:
    missing = []
    for row in _rows(spec):
        for column in _columns(spec):
            task = str(column["task"])
            method = str(row["method"])
            domain = str(row["domain"])
            if (task, method, domain) not in entries:
                missing.append(f"{task}/{method}/{domain}")
    if missing:
        raise SystemExit("Missing benchmark cells: " + ", ".join(missing))


def _latex(
    entries: Mapping[tuple[str, str, str], Mapping[str, Any]], *, spec: Mapping[str, Any], precision: int
) -> str:
    rows = _table_rows(entries, spec=spec)
    columns = _columns(spec)
    best = {}
    for column in columns:
        values = [_display_value(float(row[str(column["name"])]), column) for row in rows]
        best[str(column["name"])] = max(values) if str(column.get("best", "min")) == "max" else min(values)
    lines = []
    for row in rows:
        cells = [str(row["label"]), str(row["input_domain"])]
        for column in columns:
            name = str(column["name"])
            cells.append(
                _fmt_best_with_std(
                    _display_value(float(row[name]), column),
                    _display_value(float(row.get(f"{name}_std", 0.0)), column) if f"{name}_std" in row else None,
                    float(best[name]),
                    _column_precision(column, default=precision),
                )
            )
        lines.append(" & ".join(cells) + " \\\\")
    return "\n".join(lines) + "\n"


def _table_rows(
    entries: Mapping[tuple[str, str, str], Mapping[str, Any]], *, spec: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_spec in _rows(spec):
        method = str(row_spec["method"])
        domain = str(row_spec["domain"])
        row = {"label": str(row_spec["label"]), "input_domain": str(row_spec["input_domain"])}
        for column in _columns(spec):
            name = str(column["name"])
            row[name] = _metric(
                entries,
                task=str(column["task"]),
                method=method,
                domain=domain,
                key=str(column["metric"]),
            )
            metric_std = _metric_std(
                entries,
                task=str(column["task"]),
                method=method,
                domain=domain,
                key=str(column["metric"]),
            )
            if metric_std is not None:
                row[f"{name}_std"] = metric_std
        rows.append(row)
    return rows


def _write_csv(
    path: Path, entries: Mapping[tuple[str, str, str], Mapping[str, Any]], *, spec: Mapping[str, Any]
) -> None:
    metric_names = tuple(dict.fromkeys(str(column["metric"]) for column in _columns(spec)))
    metric_std_names = tuple(f"{name}_std" for name in metric_names)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "method",
                "domain",
                "task",
                *metric_names,
                *metric_std_names,
                "seed",
                "n_val_clips",
                "n_folds",
                "validation_folds",
            ],
        )
        writer.writeheader()
        for row in _rows(spec):
            method = str(row["method"])
            domain = str(row["domain"])
            for task in tuple(dict.fromkeys(str(column["task"]) for column in _columns(spec))):
                payload = entries[(task, method, domain)]
                metrics = payload.get("metrics", {})
                metric_stds = payload.get("metric_stds", {})
                values = {name: metrics.get(name, "") for name in metric_names}
                std_values = {
                    f"{name}_std": metric_stds.get(name, "") if isinstance(metric_stds, Mapping) else ""
                    for name in metric_names
                }
                writer.writerow(
                    {
                        "method": method,
                        "domain": domain,
                        "task": task,
                        **values,
                        **std_values,
                        "seed": payload.get("seed", ""),
                        "n_val_clips": payload.get("n_val_clips", ""),
                        "n_folds": payload.get("n_folds", ""),
                        "validation_folds": ",".join(str(fold) for fold in payload.get("validation_folds", ())),
                    }
                )


def _metric(
    entries: Mapping[tuple[str, str, str], Mapping[str, Any]],
    *,
    task: str,
    method: str,
    domain: str,
    key: str,
) -> float:
    metrics = entries[(task, method, domain)].get("metrics", {})
    if not isinstance(metrics, Mapping) or key not in metrics:
        raise SystemExit(f"Missing metric {key!r} for {task}/{method}/{domain}")
    return float(metrics[key])


def _metric_std(
    entries: Mapping[tuple[str, str, str], Mapping[str, Any]],
    *,
    task: str,
    method: str,
    domain: str,
    key: str,
) -> float | None:
    metric_stds = entries[(task, method, domain)].get("metric_stds", {})
    if not isinstance(metric_stds, Mapping) or key not in metric_stds:
        return None
    return float(metric_stds[key])


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


def _fmt(value: float, precision: int) -> str:
    return f"{float(value):.{int(precision)}f}"


def _display_value(value: float, column: Mapping[str, Any]) -> float:
    return float(value) * float(column.get("display_scale", 1.0))


def _column_precision(column: Mapping[str, Any], *, default: int) -> int:
    return int(column.get("precision", default))


def _fmt_best(value: float, best: float, precision: int) -> str:
    formatted = _fmt(value, precision)
    if value == best:
        return f"\\textbf{{{formatted}}}"
    return formatted


def _fmt_best_with_std(value: float, std: float | None, best: float, precision: int) -> str:
    formatted = _fmt_best(value, best, precision)
    if std is None:
        return formatted
    return f"{formatted} $\\pm$ {_fmt(std, precision)}"


def _sample_std(values: Sequence[float], *, mean: float) -> float:
    if len(values) < 2:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return variance**0.5


if __name__ == "__main__":
    raise SystemExit(main())
