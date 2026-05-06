from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tasks._config import TrainingConfig  # noqa: E402
from tasks.registry import task_benchmark_loss_fn, task_benchmark_val_metrics_fn, task_build_dataloaders  # noqa: E402
from tasks.tissue_doppler.roi import RoiMask, tissue_doppler_roi_mask_for_sample  # noqa: E402
from tasks.utils.seed import seed_everything  # noqa: E402
from tasks.utils.training.device import (  # noqa: E402
    _autocast_context,
    _resolve_amp_config,
    _resolve_device,
    move_to_device,
)
from tasks.utils.training.module import TrainingModule  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate tissue Doppler full-frame and sampling-gate ROI loss.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory containing config.yaml and weights.pt.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Dataset root containing croissant.json. Overrides config data.root_dir.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to RUN_DIR/roi_loss_metrics.json.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional per-video CSV output path.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional debug limit on validation videos.",
    )
    args = parser.parse_args(argv)

    run_dir = args.run_dir.expanduser().resolve()
    output_json = args.output_json.expanduser().resolve() if args.output_json else run_dir / "roi_loss_metrics.json"
    output_csv = args.output_csv.expanduser().resolve() if args.output_csv else None
    payload = evaluate_run(
        run_dir=run_dir,
        data_root=None if args.data_root is None else args.data_root.expanduser(),
        max_samples=args.max_samples,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if output_csv is not None:
        _write_csv(output_csv, payload["per_video"])
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


def evaluate_run(
    *,
    run_dir: Path,
    data_root: Path | None,
    max_samples: int | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    config_path = run_dir / "config.yaml"
    weights_path = run_dir / "weights.pt"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights: {weights_path}")

    config = _load_config(config_path)
    data_section = config.data
    if str(data_section.get("coordinate_space", "")).strip().lower() != "beamspace":
        raise ValueError("ROI loss evaluation requires a beamspace tissue Doppler config")
    seed_everything(int(data_section.get("seed", 0)))
    _train_loader, val_loader = task_build_dataloaders("tissue_doppler")(
        config=config,
        data_root=data_root,
    )
    del _train_loader
    dataset = val_loader.dataset

    module = TrainingModule(
        config=config,
        loss_fn=task_benchmark_loss_fn("tissue_doppler"),
        val_metrics_fn=task_benchmark_val_metrics_fn("tissue_doppler"),
    )
    state = torch.load(weights_path, map_location="cpu")
    module.model.load_state_dict(state)
    trainer_config = config.trainer
    device = _resolve_device(trainer_config)
    amp = _resolve_amp_config(trainer_config, device)
    module.to(device)
    module.eval()

    rows: list[dict[str, Any]] = []
    full_sum_total = 0.0
    full_count_total = 0.0
    roi_sum_total = 0.0
    roi_count_total = 0.0
    missing_roi = 0
    limit = len(dataset) if max_samples is None else min(len(dataset), int(max_samples))

    with torch.no_grad():
        for index in range(limit):
            sample = dataset[index]
            roi = tissue_doppler_roi_mask_for_sample(sample)
            sample_on_device = move_to_device(sample, device)
            with _autocast_context(amp, device):
                prediction = module.model(
                    sample_on_device.frames,
                    velocity_scale_mps_per_px_frame=sample_on_device.velocity_scale_mps_per_px_frame,
                ).float()
            target = sample_on_device.doppler_target.to(device=prediction.device, dtype=prediction.dtype)
            limit_mps = sample_on_device.sector_velocity_limit_mps
            full_sum, full_count = _velocity_band_l1_sum_count(prediction, target, limit_mps)
            full_loss = full_sum / max(full_count, 1.0)
            full_sum_total += full_sum
            full_count_total += full_count

            row: dict[str, Any] = {
                "index": int(index),
                "sample_id": str(getattr(sample, "sample_id", index)),
                "full_loss": full_loss,
                "full_voxel_count": int(round(full_count)),
            }
            if roi is None:
                missing_roi += 1
                row.update(
                    {
                        "roi_loss": None,
                        "roi_minus_full": None,
                        "roi_vs_full_percent": None,
                        "roi_spatial_pixel_count": 0,
                        "roi_voxel_count": 0,
                    }
                )
            else:
                roi_sum, roi_count = _velocity_band_l1_sum_count(
                    prediction,
                    target,
                    limit_mps,
                    valid_mask=roi.mask.to(device=prediction.device, dtype=prediction.dtype),
                )
                roi_loss = roi_sum / max(roi_count, 1.0)
                roi_sum_total += roi_sum
                roi_count_total += roi_count
                delta = roi_loss - full_loss
                row.update(
                    {
                        "roi_loss": roi_loss,
                        "roi_minus_full": delta,
                        "roi_vs_full_percent": _percent_delta(delta, full_loss),
                        "roi_spatial_pixel_count": int(roi.spatial_pixel_count),
                        "roi_voxel_count": int(round(roi_count)),
                        "roi_metadata": _jsonable_metadata(roi),
                    }
                )
            rows.append(row)

    full_losses = [float(row["full_loss"]) for row in rows]
    roi_losses = [float(row["roi_loss"]) for row in rows if row["roi_loss"] is not None]
    full_pixel_weighted = full_sum_total / max(full_count_total, 1.0)
    roi_pixel_weighted = None if roi_count_total <= 0.0 else roi_sum_total / roi_count_total
    summary: dict[str, Any] = {
        "n_val_videos": int(len(rows)),
        "n_roi_videos": int(len(roi_losses)),
        "n_missing_roi": int(missing_roi),
        "mean_full_frame_loss": _mean_or_none(full_losses),
        "mean_roi_loss": _mean_or_none(roi_losses),
        "roi_minus_full": _nullable_delta(_mean_or_none(roi_losses), _mean_or_none(full_losses)),
        "roi_vs_full_percent": _nullable_percent_delta(_mean_or_none(roi_losses), _mean_or_none(full_losses)),
        "pixel_weighted_full_frame_loss": float(full_pixel_weighted),
        "pixel_weighted_roi_loss": None if roi_pixel_weighted is None else float(roi_pixel_weighted),
        "pixel_weighted_roi_minus_full": (
            None if roi_pixel_weighted is None else float(roi_pixel_weighted - full_pixel_weighted)
        ),
        "pixel_weighted_roi_vs_full_percent": (
            None
            if roi_pixel_weighted is None
            else _percent_delta(float(roi_pixel_weighted - full_pixel_weighted), full_pixel_weighted)
        ),
        "full_frame_loss_distribution": _distribution(full_losses),
        "roi_loss_distribution": _distribution(roi_losses),
        "wallclock_s": round(time.perf_counter() - started, 2),
    }
    return {
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "weights_path": str(weights_path),
        "data_root": None if data_root is None else str(data_root),
        "summary": summary,
        "per_video": rows,
    }


def _velocity_band_l1_sum_count(
    prediction: Tensor,
    target: Tensor,
    sector_velocity_limit_mps: Tensor,
    valid_mask: Tensor | None = None,
) -> tuple[float, float]:
    limit = sector_velocity_limit_mps.reshape(-1, 1, 1, 1, 1).to(device=prediction.device, dtype=prediction.dtype)
    period = torch.clamp(2.0 * limit, min=torch.finfo(prediction.dtype).eps)
    raw_error = prediction - target.to(device=prediction.device, dtype=prediction.dtype)
    wrapped = torch.remainder(raw_error + limit, period) - limit
    normalized = torch.abs(wrapped) / torch.clamp(limit, min=torch.finfo(prediction.dtype).eps)
    finite = torch.isfinite(target).to(device=prediction.device, dtype=prediction.dtype)
    mask = finite if valid_mask is None else finite * valid_mask.to(device=prediction.device, dtype=prediction.dtype)
    return float(torch.sum(normalized * mask).detach().cpu()), float(torch.sum(mask).detach().cpu())


def _load_config(path: Path) -> TrainingConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"Expected YAML mapping in {path}")
    return TrainingConfig(raw=raw)


def _distribution(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "std": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _mean_or_none(values: Sequence[float]) -> float | None:
    return None if not values else float(np.mean(np.asarray(values, dtype=np.float64)))


def _nullable_delta(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else float(left - right)


def _nullable_percent_delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return _percent_delta(float(left - right), right)


def _percent_delta(delta: float, baseline: float) -> float | None:
    return None if abs(float(baseline)) < 1e-12 else float(delta / float(baseline) * 100.0)


def _jsonable_metadata(roi: RoiMask) -> dict[str, object]:
    return {str(key): value for key, value in roi.metadata.items() if isinstance(value, (str, int, float, bool))}


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    import csv

    fieldnames = (
        "index",
        "sample_id",
        "full_loss",
        "roi_loss",
        "roi_minus_full",
        "roi_vs_full_percent",
        "full_voxel_count",
        "roi_spatial_pixel_count",
        "roi_voxel_count",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


if __name__ == "__main__":
    raise SystemExit(main())
