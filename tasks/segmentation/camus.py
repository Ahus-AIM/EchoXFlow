"""External CAMUS evaluation of the Task 3 benchmark checkpoints on the CAMUS EchoXFlow package.

The package (https://huggingface.co/datasets/Ahus-AIM/CAMUS-beamspace) stores every CAMUS half-sequence as an EchoXFlow
recording: B-mode on the fitted fan in beamspace and per-frame LV contours (``annotations/camus_lv_contour``) in the
layout of EchoXFlow strain contours. Each recording is prepared exactly like a Task 3 validation clip (input domain,
resolution, per-frame min-max normalization), the contours are rasterized to LV cavity and myocardium masks on the
Cartesian metric grid, and the benchmark's own validation metric scores the prediction. Every checkpoint of the five
folds is evaluated on all 1,000 recordings.

Scores per recording: thresholded Dice over all frames (the benchmark metric), and the Dice at the ED and ES frames.

    python -m tasks.segmentation.camus --package-root /path/to/CAMUS-beamspace \
        --benchmark-root /path/to/benchmark --resolutions 128 256 320 --output-dir outputs/segmentation_camus
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from echoxflow import RecordingRecord, find_recordings, load_croissant, open_recording
from echoxflow.scan import CartesianGrid, SectorGeometry, sector_to_cartesian
from tasks.registry import task_load_config
from tasks.segmentation.dataset import _minmax_normalize_frames_per_frame, _target_masks_from_contour_points
from tasks.segmentation.types import Sample
from tasks.utils.dataset import (
    cartesian_grid_sample_grid,
    cartesian_image_sample_grid,
    cartesian_sector_mask,
    collate_samples,
    resize_stack,
)
from tasks.utils.models.registry import METHODS
from tasks.utils.models.temporal_mean import TemporalMeanModel
from tasks.utils.training import TrainingModule
from tasks.utils.training.loss import shared_eval_predictions
from tasks.utils.training.metrics import segmentation_val_metrics

DOMAINS = ("beamspace", "cartesian")
CONTENT_TYPE = "2d_left_ventricular_strain"
FIELDS = (
    "resolution",
    "method",
    "domain",
    "fold",
    "exam_id",
    "recording_id",
    "view",
    "frames",
    "dice_all",
    "dice_cavity_all",
    "dice_myocardium_all",
    "dice_ed",
    "dice_es",
)


def cell_dir(root: Path, resolution: int, method: str, domain: str, fold: int) -> Path:
    path = root / "segmentation" / f"paper_r{resolution}" / method / domain / f"fold={fold}" / "seed=0"
    if not (path / "config.yaml").is_file():
        raise FileNotFoundError(f"missing benchmark cell {path}")
    return path


def load_cell_model(path: Path, method: str) -> tuple[torch.nn.Module, Any]:
    config = task_load_config("segmentation")(path / "config.yaml")
    if method == "temporal_mean":
        state = torch.load(path / "mean.pt", map_location="cpu", weights_only=True)
        height, width = state["mean"].shape[-2:]
        model: torch.nn.Module = TemporalMeanModel(task_kind="segmentation", mean=torch.full((2, height, width), 0.25))
        model.load_state_dict(state, strict=True)
    else:
        module = TrainingModule(config=config, loss_fn=cast(Any, None), val_metrics_fn=None)
        state = torch.load(path / "weights.pt", map_location="cpu", weights_only=True)
        module.model.load_state_dict(state, strict=True)
        model = module.model
    return model.eval(), config


class CamusRecordings(Dataset[Any]):
    """One full CAMUS recording per item, prepared like a Task 3 validation clip."""

    def __init__(self, root: Path, records: Sequence[RecordingRecord], *, domain: str, resolution: int) -> None:
        self.root, self.records, self.domain, self.size = root, list(records), domain, int(resolution)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        store = open_recording(record, root=self.root)
        stream = store.load_stream("2d_brightness_mode")
        frames = np.asarray(stream.data, dtype=np.float32)
        geometry = stream.metadata.geometry
        if not isinstance(geometry, SectorGeometry):
            raise ValueError(f"{record.recording_id}: missing sector geometry")
        contours = np.asarray(store.load_array("annotations/camus_lv_contour"), dtype=np.float32)
        if contours.shape[0] != frames.shape[0]:
            raise ValueError(f"{record.recording_id}: {contours.shape[0]} contours for {frames.shape[0]} frames")
        attrs = json.loads((self.root / record.zarr_path / ".zattrs").read_text(encoding="utf-8"))
        camus = attrs["recording_manifest"]["metadata"]["camus"]
        shape = (self.size, self.size)
        # Input frames: Task 3 preprocessing (beamspace resize, or full scan conversion then resize).
        grid = CartesianGrid.from_sector_height(geometry, self.size)
        if self.domain == "cartesian":
            converted = [sector_to_cartesian(frame, geometry, grid=grid, interpolation="linear") for frame in frames]
            inputs = np.stack([image.data for image in converted]).astype(np.float32, copy=False)
            inputs = resize_stack(inputs, shape) if inputs.shape[-2:] != shape else inputs
            sample_grid = cartesian_image_sample_grid(grid)
        else:
            inputs = resize_stack(frames, shape)
            sample_grid = cartesian_grid_sample_grid(geometry, grid, source_shape=shape)
        inputs = _minmax_normalize_frames_per_frame(inputs)
        masks, valid = _target_masks_from_contour_points(
            contour_points=contours,
            geometry=geometry,
            content_type=CONTENT_TYPE,
            role_id="lv",
            image_shape=grid.shape,
            source_image_shape=frames.shape[-2:],
            output_grid=grid,
            target_mask_channels=2,
            base_valid_mask=np.ones(len(frames), dtype=np.float32),
        )
        sector = cartesian_sector_mask(geometry, grid, shape=grid.shape).astype(np.float32, copy=False)
        if not bool(valid.all()):
            raise ValueError(f"{record.recording_id}: contours did not rasterize on every frame")
        return {
            "record": record,
            "inputs": inputs.astype(np.float32, copy=False),
            "masks": masks.astype(bool),
            "valid": (valid * sector[None, None]).astype(bool),
            "sample_grid": sample_grid,
            "ed": int(camus["ed_frame_index"]),
            "es": int(camus["es_frame_index"]),
            "view": str(camus["view"]),
        }


def make_sample(item: dict[str, Any], valid: np.ndarray, frames: torch.Tensor) -> Sample:
    device = frames.device
    count = int(item["inputs"].shape[0])
    sample = Sample(
        frames=frames.cpu(),
        target_masks=torch.zeros((1, 2, count, 1, 1)),
        sample_id=f"{item['record'].recording_id}:camus",
        record=item["record"],
        coordinate_space="camus",
        cartesian_metric_target_masks=torch.from_numpy(item["masks"][None].astype(np.float32)),
        cartesian_metric_valid_mask=torch.from_numpy(valid[None].astype(np.float32)),
        cartesian_metric_sample_grid=torch.from_numpy(item["sample_grid"][None]),
    )
    batch = collate_samples([sample])
    for name in ("cartesian_metric_target_masks", "cartesian_metric_valid_mask", "cartesian_metric_sample_grid"):
        setattr(batch, name, getattr(batch, name).to(device))
    # The same frames tensor for every call, so the shared prediction cache runs each model once per recording.
    batch.frames = frames
    return batch


def score(model: torch.nn.Module, config: Any, item: dict[str, Any], frames: torch.Tensor) -> dict[str, float]:
    valid_all = item["valid"]
    subsets = {"all": valid_all}
    for phase in ("ed", "es"):
        keep = np.zeros_like(valid_all)
        keep[:, item[phase]] = valid_all[:, item[phase]]
        subsets[phase] = keep
    out: dict[str, float] = {}
    for name, valid in subsets.items():
        terms = segmentation_val_metrics(cast(Any, model), make_sample(item, valid, frames), config)
        out[f"dice_{name}"] = float(terms["cartesian_dice_mean"].float().mean())
        if name == "all":
            out["dice_cavity_all"] = float(terms["cartesian_dice_ch0"].float().mean())
            out["dice_myocardium_all"] = float(terms["cartesian_dice_ch1"].float().mean())
    return out


def _keep(item: dict[str, Any]) -> dict[str, Any]:
    return item


def _single_threaded(worker_id: int) -> None:
    del worker_id
    torch.set_num_threads(1)  # each loader worker preprocesses one recording; the cores are shared between them


def evaluate_group(
    args: argparse.Namespace, records: Sequence[RecordingRecord], resolution: int, domain: str, writer: Any
) -> None:
    device = torch.device(args.device)
    models = []
    for method in args.methods:
        for fold in args.folds:
            model, config = load_cell_model(cell_dir(args.benchmark_root, resolution, method, domain, fold), method)
            models.append((method, fold, model.to(device), config))
    dataset = CamusRecordings(args.package_root, records, domain=domain, resolution=resolution)
    loader = DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        prefetch_factor=2 if args.workers else None,
        worker_init_fn=_single_threaded if args.workers else None,
        collate_fn=cast(Any, _keep),
    )
    for index, item in enumerate(loader):
        frames = torch.from_numpy(item["inputs"])[None, None].to(device)
        for method, fold, model, config in models:
            with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16), shared_eval_predictions():
                scores = score(model, config, item, frames)
            writer.writerow(
                {
                    "resolution": resolution,
                    "method": method,
                    "domain": domain,
                    "fold": fold,
                    "exam_id": item["record"].exam_id,
                    "recording_id": item["record"].recording_id,
                    "view": item["view"],
                    "frames": int(item["inputs"].shape[0]),
                    **scores,
                }
            )
        if (index + 1) % 50 == 0 or index + 1 == len(records):
            print(f"r{resolution}/{domain}: {index + 1}/{len(records)} recordings", flush=True)
    del models
    torch.cuda.empty_cache()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package-root", type=Path, required=True, help="CAMUS-beamspace package (has croissant.json)")
    parser.add_argument("--benchmark-root", type=Path, required=True, help="folder holding segmentation/paper_r*/")
    parser.add_argument("--resolutions", type=int, nargs="+", default=[128, 256, 320])
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--methods", nargs="+", choices=("temporal_mean", *METHODS), default=["temporal_mean", *METHODS]
    )
    parser.add_argument("--max-recordings", type=int, help="smoke runs only")
    parser.add_argument("--workers", type=int, default=8, help="recording preprocessing processes")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/segmentation_camus"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    torch.set_num_threads(4)  # the metrics run on the device; the main process only collates
    found = find_recordings(
        croissant=load_croissant(args.package_root / "croissant.json"), array_paths=("2d_brightness_mode",)
    )
    records = sorted(found, key=lambda record: record.recording_id)
    if args.max_recordings is not None:
        records = records[: args.max_recordings]
    elif len(records) != 1000:
        parser.error(f"expected 1000 CAMUS recordings, found {len(records)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for resolution in args.resolutions:
        for domain in args.domains:
            path = args.output_dir / f"per_recording_r{resolution}_{domain}.csv"
            if path.is_file():
                print(f"{path} exists; skipping", flush=True)
                continue
            partial = path.with_suffix(".partial")
            with partial.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                evaluate_group(args, records, resolution, domain, writer)
            partial.rename(path)
            print(path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
