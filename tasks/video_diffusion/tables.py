"""Beamspace-versus-Cartesian comparison tables with cluster-bootstrap tests and Holm correction.

Each ``--pair RESOLUTION BEAMSPACE_EVAL_DIR CARTESIAN_EVAL_DIR`` names the ``evaluation`` directories written by
:mod:`tasks.video_diffusion.evaluate` for the two coordinate spaces at one resolution.

EchoXFlow protocol (default): a two-sided, paired exam-cluster bootstrap. Each replicate resamples exams with
replacement, applies the same exam multiplicities to the real, beamspace and Cartesian rows, recomputes FD (frame
spaces) or paired KID (clip spaces, same subsets as the point estimate) per view, and averages the views equally.
The p-value is the recentered two-sided bootstrap p-value of ``D_cartesian - D_beamspace``; Holm's step-down
correction is applied across all reported feature-space x resolution comparisons. The replicate differences are
saved to ``comparison_replicates.npz`` so :mod:`tasks.video_diffusion.cv_table` can pool the folds.

CAMUS protocol (``--camus``, reads ``<eval dir>/camus``): distances are unpaired; generated and real rows belong to
the same patients, so one patient-cluster bootstrap resamples both sides.

Example::

    uv run python -m tasks.video_diffusion.tables \\
        --pair 128 runs/beam128/evaluation runs/cart128/evaluation \\
        --pair 256 runs/beam256/evaluation runs/cart256/evaluation \\
        --output-dir outputs/video_diffusion/tables
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tasks.utils.stats import bootstrap_multiplicities, holm_adjust, recentered_p_value
from tasks.video_diffusion.features import EXTRACTOR_LABELS
from tasks.video_diffusion.metrics import (
    _moment_factor,
    _trace_sqrt_product,
    kid_subsets,
    paired_kid,
    unpaired_kid,
)

ARMS = ("beamspace", "cartesian")


def _fd_weighted(real: torch.Tensor, real_w: torch.Tensor, generated: torch.Tensor, generated_w: torch.Tensor) -> float:
    real_mean, real_factor = _moment_factor(real, real_w)
    generated_mean, generated_factor = _moment_factor(generated, generated_w)
    value = (
        (real_mean - generated_mean).square().sum()
        + real_factor.square().sum()
        + generated_factor.square().sum()
        - 2.0 * _trace_sqrt_product(real_factor, generated_factor)
    )
    return max(float(value), 0.0)


def load_arm(directory: Path, name: str) -> dict[str, np.ndarray] | None:
    path = directory / "features" / f"{name}.npz"
    if not path.is_file():
        return None
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def compare_echoxflow(
    arms: dict[str, dict[str, np.ndarray]],
    *,
    kind: str,
    replicates: int,
    seed: int,
    kid: dict[str, int],
    device: torch.device,
) -> dict[str, Any]:
    beam, cart = arms["beamspace"], arms["cartesian"]
    if not np.array_equal(beam["generated_ids"], cart["generated_ids"]):
        raise ValueError("beamspace and Cartesian evaluations used different clip selections")
    exams = sorted(set(beam["real_exam_ids"].tolist()))
    exam_index = {exam: index for index, exam in enumerate(exams)}
    plan = bootstrap_multiplicities(
        len(exams), replicates=replicates, seed=seed, namespace=f"internal-distance:{kind}:exam"
    )
    views = sorted(set(beam["generated_views"].tolist()))
    observed: dict[str, list[float]] = {arm: [] for arm in ARMS}
    deltas = np.zeros(replicates)
    for view in views:
        mask = beam["generated_views"] == view
        rows = np.asarray([exam_index[exam] for exam in beam["generated_exam_ids"][mask]])
        real = torch.as_tensor(beam["real"][mask], device=device, dtype=torch.float64)
        generated = {
            arm: torch.as_tensor(arms[arm]["generated"][mask], device=device, dtype=torch.float64) for arm in ARMS
        }
        if kind == "frame":
            ones = torch.ones(len(rows), device=device, dtype=torch.float64)
            for arm in ARMS:
                observed[arm].append(_fd_weighted(real, ones, generated[arm], ones))
            for replicate in range(replicates):
                weights = torch.as_tensor(plan[replicate, rows], device=device)
                values = {arm: _fd_weighted(real, weights, generated[arm], weights) for arm in ARMS}
                deltas[replicate] += (values["cartesian"] - values["beamspace"]) / len(views)
        else:
            subsets = kid_subsets(
                len(rows),
                subsets=kid["subsets"],
                subset_size=kid["subset_size"],
                seed=kid["seed"],
                namespace=f"{view}:paired:kid",
            )
            for arm in ARMS:
                observed[arm].append(paired_kid(real, generated[arm], subsets, device=device))
            for replicate in range(replicates):
                weights = plan[replicate, rows]
                values = {
                    arm: paired_kid(real, generated[arm], subsets, weights=weights, device=device) for arm in ARMS
                }
                deltas[replicate] += (values["cartesian"] - values["beamspace"]) / len(views)
    return _summary(observed, deltas, clusters=len(exams), views=views)


def _unpaired_distance(
    real: torch.Tensor,
    generated: torch.Tensor,
    real_weights: np.ndarray,
    generated_weights: np.ndarray,
    *,
    kind: str,
    kid: dict[str, int],
    namespace: str,
    device: torch.device,
) -> float:
    if kind == "frame":
        return _fd_weighted(
            real,
            torch.as_tensor(real_weights, device=device),
            generated,
            torch.as_tensor(generated_weights, device=device),
        )
    return unpaired_kid(
        real,
        generated,
        subsets=kid["subsets"],
        subset_size=kid["subset_size"],
        seed=kid["seed"],
        namespace=namespace,
        real_weights=real_weights,
        generated_weights=generated_weights,
        device=device,
    )


def compare_camus(
    arms: dict[str, dict[str, np.ndarray]],
    *,
    kind: str,
    replicates: int,
    seed: int,
    kid: dict[str, int],
    device: torch.device,
) -> dict[str, Any]:
    beam, cart = arms["beamspace"], arms["cartesian"]
    if not np.array_equal(beam["generated_ids"], cart["generated_ids"]):
        raise ValueError("beamspace and Cartesian CAMUS evaluations used different conditioning patients")
    # Generated and real rows belong to the same patients, so one patient-cluster bootstrap resamples both sides.
    patients = sorted(set(beam["generated_exam_ids"].tolist()) | set(beam["real_exam_ids"].tolist()))
    plan = bootstrap_multiplicities(len(patients), replicates=replicates, seed=seed, namespace="camus/patient")
    views = sorted(set(beam["generated_views"].tolist()))
    observed: dict[str, list[float]] = {arm: [] for arm in ARMS}
    deltas = np.zeros(replicates)
    for view in views:
        generated_mask, real_mask = beam["generated_views"] == view, beam["real_views"] == view
        generated_rows = np.asarray([patients.index(p) for p in beam["generated_exam_ids"][generated_mask]])
        real_rows = np.asarray([patients.index(p) for p in beam["real_exam_ids"][real_mask]])
        real = torch.as_tensor(beam["real"][real_mask], device=device, dtype=torch.float64)
        generated = {
            arm: torch.as_tensor(arms[arm]["generated"][generated_mask], device=device, dtype=torch.float64)
            for arm in ARMS
        }

        namespace = f"{view}:kid"
        ones_real, ones_generated = np.ones(len(real_rows)), np.ones(len(generated_rows))
        for arm in ARMS:
            observed[arm].append(
                _unpaired_distance(
                    real,
                    generated[arm],
                    ones_real,
                    ones_generated,
                    kind=kind,
                    kid=kid,
                    namespace=namespace,
                    device=device,
                )
            )
        for replicate in range(replicates):
            real_w, generated_w = plan[replicate, real_rows], plan[replicate, generated_rows]
            values = {
                arm: _unpaired_distance(
                    real,
                    generated[arm],
                    real_w,
                    generated_w,
                    kind=kind,
                    kid=kid,
                    namespace=namespace,
                    device=device,
                )
                for arm in ARMS
            }
            deltas[replicate] += (values["cartesian"] - values["beamspace"]) / len(views)
    result = _summary(observed, deltas, clusters=len(patients), views=views)
    result["bootstrap_patient_ids"] = patients
    return result


def _summary(
    observed: dict[str, list[float]], deltas: np.ndarray, *, clusters: int, views: list[str]
) -> dict[str, Any]:
    beam, cart = float(np.mean(observed["beamspace"])), float(np.mean(observed["cartesian"]))
    delta = cart - beam
    return {
        "beamspace": beam,
        "cartesian": cart,
        "relative_difference_percent": 100.0 * delta / abs(cart) if cart else float("nan"),
        "delta_cartesian_minus_beamspace": delta,
        "probability_beamspace_better": float(
            (np.count_nonzero(deltas > 0) + 0.5 * np.count_nonzero(deltas == 0)) / len(deltas)
        ),
        "p_two_sided": recentered_p_value(deltas, delta),
        "ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
        "clusters": clusters,
        "views": views,
        "per_view": {arm: dict(zip(views, values, strict=True)) for arm, values in observed.items()},
        "replicate_deltas": deltas,
    }


def _format(value: float) -> str:
    return f"{value:.1f}" if abs(value) >= 1 else f"{value:.3g}"


def replicate_key(resolution: int, extractor: str) -> str:
    return f"r{int(resolution):04d}_{extractor}"


def write_tables(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    title: str,
    replicates: int,
    seed: int,
    replicate_deltas: dict[str, np.ndarray] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if replicate_deltas is not None:
        temporary = output_dir / "comparison_replicates.npz.tmp"
        with temporary.open("wb") as stream:
            np.savez(stream, **replicate_deltas)  # type: ignore[arg-type]  # numpy stubs type **kwds as bool
        os.replace(temporary, output_dir / "comparison_replicates.npz")
    (output_dir / "comparison.json").write_text(
        json.dumps({"replicates": replicates, "seed": seed, "rows": rows}, indent=2), encoding="utf-8"
    )
    columns = [
        "resolution",
        "extractor",
        "label",
        "beamspace",
        "cartesian",
        "relative_difference_percent",
        "probability_beamspace_better",
        "p_two_sided",
        "p_holm",
        "delta_cartesian_minus_beamspace",
        "clusters",
    ]
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        f"# {title}",
        "",
        f"Positive relative difference favours beamspace; {replicates} bootstrap replicates, "
        f"Holm across {len(rows)} comparisons.",
        "",
    ]
    for resolution in sorted({row["resolution"] for row in rows}):
        lines += [
            f"## {resolution} x {resolution}",
            "",
            "| Feature space and metric | Beamspace | Scan-converted | Relative difference "
            "| Holm-adjusted two-sided p |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in (r for r in rows if r["resolution"] == resolution):
            lines.append(
                f"| {row['label']} | {_format(row['beamspace'])} | {_format(row['cartesian'])} | "
                f"{row['relative_difference_percent']:+.1f}% | {row['p_holm']:.2f} |"
            )
        lines.append("")
    (output_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--pair", nargs=3, action="append", metavar=("RESOLUTION", "BEAMSPACE_DIR", "CARTESIAN_DIR"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camus", action="store_true", help="compare the CAMUS evaluations (<dir>/camus)")
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=None, help="default 17 (EchoXFlow) or 31 (CAMUS)")
    parser.add_argument("--kid-subsets", type=int, default=100)
    parser.add_argument("--kid-subset-size", type=int, default=32)
    parser.add_argument("--kid-seed", type=int, default=None, help="default 17 (EchoXFlow) or 20260724 (CAMUS)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    seed = args.seed if args.seed is not None else (31 if args.camus else 17)
    kid = {
        "subsets": args.kid_subsets,
        "subset_size": args.kid_subset_size,
        "seed": args.kid_seed if args.kid_seed is not None else (20260724 if args.camus else 17),
    }
    device = torch.device(args.device)
    compare = compare_camus if args.camus else compare_echoxflow
    rows = []
    replicate_deltas: dict[str, np.ndarray] = {}
    for resolution, beam_dir, cart_dir in args.pair:
        directories = {"beamspace": Path(beam_dir), "cartesian": Path(cart_dir)}
        if args.camus:
            directories = {arm: path / "camus" for arm, path in directories.items()}
        for name, label in EXTRACTOR_LABELS.items():
            loaded = {arm: load_arm(path, name) for arm, path in directories.items()}
            if any(value is None for value in loaded.values()):
                print(f"{resolution}: no {name} features in both arms; skipped")
                continue
            arms = {arm: value for arm, value in loaded.items() if value is not None}
            kind = str(arms["beamspace"]["kind"])
            result = compare(arms, kind=kind, replicates=args.replicates, seed=seed, kid=kid, device=device)
            replicate_deltas[replicate_key(int(resolution), name)] = result.pop("replicate_deltas")
            rows.append({"resolution": int(resolution), "extractor": name, "label": label, **result})
            print(
                f"{resolution} {label}: beamspace {result['beamspace']:.4g}, Cartesian {result['cartesian']:.4g}",
                flush=True,
            )
    for row, adjusted in zip(rows, holm_adjust([row["p_two_sided"] for row in rows]), strict=True):
        row["p_holm"] = adjusted
    title = "CAMUS generation quality" if args.camus else "ECG-conditioned generation quality"
    write_tables(
        rows, args.output_dir, title=title, replicates=args.replicates, seed=seed, replicate_deltas=replicate_deltas
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
