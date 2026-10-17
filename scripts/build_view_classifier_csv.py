"""Per-recording view-classifier probabilities for single-plane 2-D B-mode recordings.

One CSV row per recording with ``data/2d_brightness_mode`` (single-plane; biplane/triplane planes live under other
array paths and are not included). Three classifier sources can be selected with ``--models``:

``native``
    The scanner's own view detector stored by EchoXFlow in
    ``recording_manifest.metadata.scanner_view_detector.scores`` (ten logits, class order undocumented). The CSV keeps
    the raw logits (``native_score_0..9``) and their softmax (``native_prob_0..9``); inferred class names live in a
    separate sidecar JSON (``native_class_map.json``), not in the column names. Recordings without scores get empty
    native columns.

``echoprime``
    EchoPrime view classifier (ConvNeXt-B, 11 coarse views; ``ECHOPRIME_VIEW_CLASSIFIER``).

``echofine``
    EchoFine / EchoView47 Xception (47 fine views; ``ECHOFINE_WEIGHTS``).

Both image classifiers were trained on scan-converted (DICOM-like) frames, so every selected frame is scan converted
with :func:`echoxflow.scan.sector_to_cartesian`. Preprocessing matches the script that produced the original paper
selection (``bmode_view_classifier_predictions.csv``):

* intensity: per-frame 5th/99th percentile clip and min-max scaling to ``[0, 1]`` of the native beamspace frame;
* EchoPrime: nearest-neighbour scan conversion at 224 px depth, zero outside the sector, centre square crop, bicubic
  resize to 224x224, grey -> RGB in 0..255 and EchoPrime mean/std normalisation;
* EchoFine: bilinear scan conversion at 224 px depth, zero outside the sector, aspect-preserving letterbox onto a black
  224x224 canvas (bilinear), grey -> RGB in 0..255 (the model divides by 255 internally).

The original selection classified a single frame (the last frame before an ECG QRS trigger). EchoXFlow does not
store QRS trigger times, so this script instead averages the softmax probabilities over ``--frames-per-recording``
evenly spaced frames (``np.linspace(0, n_frames - 1, N)`` rounded, de-duplicated; all frames when shorter). The frame
indices used are written to ``frames_used``.

The output is appended to and resumable: recordings already present in the output file are skipped. When every eligible
recording has a row, the file is rewritten sorted by ``(exam_id, recording_id)``.

Example::

    source outputs/video_diffusion_checks/extractor_env.sh
    uv run python scripts/build_view_classifier_csv.py \\
        --data-root /path/to/echoxflow --split-csv /path/to/splits.csv \\
        --output-dir outputs/view_classifiers --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from echoxflow import data_root, load_croissant, open_recording  # noqa: E402
from echoxflow.scan import SectorGeometry, sector_to_cartesian  # noqa: E402

BMODE_STREAM = "2d_brightness_mode"
BMODE_PATH = f"data/{BMODE_STREAM}"
DEFAULT_OUTPUT_DIR = Path("outputs/view_classifiers")
MODEL_CHOICES = ("native", "echoprime", "echofine")
NATIVE_CLASSES = 10
INPUT_SIZE = 224
PERCENTILES = (5.0, 99.0)

ECHOPRIME_CLASSES = (
    "A2C",
    "A3C",
    "A4C",
    "A5C",
    "Apical_Doppler",
    "Doppler_Parasternal_Long",
    "Doppler_Parasternal_Short",
    "Parasternal_Long",
    "Parasternal_Short",
    "SSN",
    "Subcostal",
)
ECHOFINE_CLASSES = (
    "a2ch-full",
    "a2ch-la",
    "a2ch-lv",
    "a3ch-full",
    "a3ch-la",
    "a3ch-lv",
    "a3ch-outflow",
    "a4ch-full",
    "a4ch-ias",
    "a4ch-la",
    "a4ch-lv",
    "a4ch-ra",
    "a4ch-rv",
    "a5ch-full",
    "a5ch-outflow",
    "apex",
    "doppler-ao-descending",
    "doppler-av",
    "doppler-mv",
    "doppler-pv",
    "doppler-tissue-lateral",
    "doppler-tissue-rv",
    "doppler-tissue-septal",
    "doppler-tv",
    "mmode-a4ch-rv",
    "mmode-ivc",
    "mmode-plax-av",
    "mmode-plax-lv",
    "mmode-plax-mitral",
    "plax-full-la",
    "plax-full-lv",
    "plax-full-mv",
    "plax-full-out",
    "plax-full-rv-ao",
    "plax-tv",
    "plax-valves-av",
    "plax-valves-mv",
    "psax-all",
    "psax-av",
    "psax-lv-apex",
    "psax-lv-base",
    "psax-lv-mid",
    "psax-pv",
    "psax-tv",
    "subcostal-heart",
    "subcostal-ivc",
    "suprasternal",
)
MODEL_CLASSES = {"echoprime": ECHOPRIME_CLASSES, "echofine": ECHOFINE_CLASSES}
KEY_COLUMNS = (
    "exam_id",
    "recording_id",
    "zarr_path",
    "five_fold_split",
    "n_frames",
    "frames_used",
    "skip_reason",
)


def sanitize(label: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in str(label).strip()) or "class"


# ---------------------------------------------------------------------------------------------------------------------
# Columns


def csv_columns(models: Sequence[str]) -> list[str]:
    columns = list(KEY_COLUMNS)
    if "native" in models:
        columns += [f"native_score_{index}" for index in range(NATIVE_CLASSES)]
        columns += [f"native_prob_{index}" for index in range(NATIVE_CLASSES)]
        columns += ["native_top1", "native_top1_prob"]
    for model in ("echoprime", "echofine"):
        if model in models:
            columns += [f"{model}_prob_{sanitize(label)}" for label in MODEL_CLASSES[model]]
            columns += [f"{model}_top1", f"{model}_top1_prob"]
    return columns


def _fmt(value: float, decimals: int = 6) -> str:
    return f"{float(value):.{decimals}g}" if math.isfinite(float(value)) else ""


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = np.asarray(values, dtype=np.float64) - float(np.max(values))
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum()


def native_columns(scores: Sequence[float] | None) -> dict[str, str]:
    if scores is None or len(scores) != NATIVE_CLASSES:
        return {}
    logits = np.asarray(scores, dtype=np.float64)
    probabilities = softmax(logits)
    top = int(np.argmax(probabilities))
    row = {f"native_score_{index}": _fmt(value, 7) for index, value in enumerate(logits)}
    row.update({f"native_prob_{index}": _fmt(value) for index, value in enumerate(probabilities)})
    row.update({"native_top1": str(top), "native_top1_prob": _fmt(probabilities[top])})
    return row


def model_columns(model: str, probabilities: np.ndarray) -> dict[str, str]:
    labels = MODEL_CLASSES[model]
    top = int(np.argmax(probabilities))
    row = {f"{model}_prob_{sanitize(label)}": _fmt(value) for label, value in zip(labels, probabilities, strict=True)}
    row.update({f"{model}_top1": labels[top], f"{model}_top1_prob": _fmt(probabilities[top])})
    return row


# ---------------------------------------------------------------------------------------------------------------------
# Frame loading and preprocessing (CPU workers)


@dataclass(frozen=True)
class Job:
    exam_id: str
    recording_id: str
    zarr_path: str


@dataclass
class Loaded:
    job: Job
    n_frames: int = 0
    frames_used: tuple[int, ...] = ()
    native_scores: list[float] | None = None
    echoprime: np.ndarray | None = None  # [F, 224, 224] uint8
    echofine: np.ndarray | None = None  # [F, 224, 224] uint8
    skip_reason: str = ""


_WORKER: dict[str, Any] = {}


def _init_worker(root: str, models: tuple[str, ...], frames_per_recording: int) -> None:
    _WORKER.update(root=Path(root), models=models, frames=int(frames_per_recording))


def frame_indices(n_frames: int, count: int) -> np.ndarray:
    if n_frames <= 0:
        return np.zeros((0,), dtype=np.int64)
    if n_frames <= count:
        return np.arange(n_frames, dtype=np.int64)
    return np.unique(np.rint(np.linspace(0, n_frames - 1, count)).astype(np.int64))


def percentile_clip(frames: np.ndarray) -> np.ndarray:
    """Per-frame 5/99 percentile clip and min-max to ``[0, 1]`` (matches the original selection script)."""
    output = np.zeros(frames.shape, dtype=np.float32)
    for index, frame in enumerate(np.asarray(frames, dtype=np.float32)):
        low, high = np.percentile(frame, PERCENTILES)
        if high > low:
            output[index] = (np.clip(frame, low, high) - low) / (high - low)
    return output


def _scan_convert(frames: np.ndarray, geometry: SectorGeometry, interpolation: str) -> np.ndarray:
    """``[F, R, A]`` in 0..1 -> uint8 ``[F, H, W]`` Cartesian at 224 px depth, zero outside the sector."""
    converted = sector_to_cartesian(
        np.moveaxis(frames, 0, -1),
        geometry,
        output_height=INPUT_SIZE,
        interpolation=interpolation,
    )
    image = np.clip(np.asarray(converted.data, dtype=np.float32), 0.0, 1.0)
    image[~np.asarray(converted.mask, dtype=bool)] = 0.0
    return np.moveaxis(np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8), -1, 0)


def echoprime_images(frames: np.ndarray, geometry: SectorGeometry) -> np.ndarray:
    from PIL import Image

    cartesian = _scan_convert(frames, geometry, "nearest")
    height, width = cartesian.shape[1:]
    side = min(height, width)
    y0, x0 = (height - side) // 2, (width - side) // 2
    square = cartesian[:, y0 : y0 + side, x0 : x0 + side]
    resample = Image.Resampling.BICUBIC
    return np.stack([np.asarray(Image.fromarray(image).resize((INPUT_SIZE, INPUT_SIZE), resample)) for image in square])


def echofine_images(frames: np.ndarray, geometry: SectorGeometry) -> np.ndarray:
    from PIL import Image

    cartesian = _scan_convert(frames, geometry, "linear")
    height, width = cartesian.shape[1:]
    scale = min(INPUT_SIZE / max(1, width), INPUT_SIZE / max(1, height))
    new_width = max(1, min(INPUT_SIZE, round(width * scale)))
    new_height = max(1, min(INPUT_SIZE, round(height * scale)))
    output = []
    for image in cartesian:
        resized = Image.fromarray(image).resize((new_width, new_height), Image.Resampling.BILINEAR)
        canvas = Image.new("L", (INPUT_SIZE, INPUT_SIZE), 0)
        canvas.paste(resized, ((INPUT_SIZE - new_width) // 2, (INPUT_SIZE - new_height) // 2))
        output.append(np.asarray(canvas))
    return np.stack(output)


def _native_scores(manifest: Mapping[str, Any]) -> list[float] | None:
    metadata = manifest.get("metadata")
    detector = metadata.get("scanner_view_detector") if isinstance(metadata, Mapping) else None
    scores = detector.get("scores") if isinstance(detector, Mapping) else None
    if isinstance(scores, list) and len(scores) == NATIVE_CLASSES:
        values = [float(value) for value in scores]
        if all(math.isfinite(value) for value in values):
            return values
    return None


def load_job(job: Job) -> Loaded:
    root: Path = _WORKER["root"]
    models: tuple[str, ...] = _WORKER["models"]
    loaded = Loaded(job=job)
    try:
        store = open_recording(root / job.zarr_path, root=root)
        manifest = store.group.attrs.get("recording_manifest") or {}
        loaded.native_scores = _native_scores(manifest)
        array = store.group[BMODE_PATH]
        loaded.n_frames = int(array.shape[0])
        if array.ndim != 3:
            raise ValueError(f"{BMODE_PATH} is not single-plane [T, R, A]: shape {tuple(array.shape)}")
        if not {"echoprime", "echofine"} & set(models):
            return loaded
        geometry = store.load_stream_slice(BMODE_STREAM, 0, 1).metadata.geometry
        if geometry is None:
            raise ValueError("no B-mode sector geometry")
        indices = frame_indices(loaded.n_frames, _WORKER["frames"])
        if not len(indices):
            raise ValueError("no B-mode frames")
        frames = np.asarray(array.get_orthogonal_selection((indices, slice(None), slice(None))))
        normalized = percentile_clip(frames)
        loaded.frames_used = tuple(int(index) for index in indices)
        if "echoprime" in models:
            loaded.echoprime = echoprime_images(normalized, geometry)
        if "echofine" in models:
            loaded.echofine = echofine_images(normalized, geometry)
    except Exception as error:  # noqa: BLE001 - recorded per row; the run continues
        loaded.skip_reason = f"{type(error).__name__}: {error}"[:300]
    return loaded


# ---------------------------------------------------------------------------------------------------------------------
# Classifiers (GPU)


class ImageClassifiers:
    """EchoPrime and EchoFine heads built on the loaders in :mod:`tasks.video_diffusion.features`."""

    def __init__(self, models: Sequence[str], device: str) -> None:
        import torch
        from torch import nn

        from tasks.video_diffusion import features

        self.torch = torch
        self.features = features
        self.device = torch.device(device)
        self.models: dict[str, Any] = {}
        if "echoprime" in models:
            extractor = features.EchoPrimeViewExtractor(self.device)
            state = features._state_dict(
                features._weights_path(
                    "ECHOPRIME_VIEW_CLASSIFIER",
                    "EchoPrime model_data/weights/view_classifier.pt",
                )
            )
            head = nn.Linear(int(state["classifier.2.weight"].shape[1]), len(ECHOPRIME_CLASSES))
            head.load_state_dict(
                {
                    "weight": state["classifier.2.weight"],
                    "bias": state["classifier.2.bias"],
                }
            )
            self.models["echoprime"] = (
                extractor.model,
                head.eval().requires_grad_(False).to(self.device),
            )
        if "echofine" in models:
            self.models["echofine"] = features.EchoFineExtractor(self.device).model

    def probabilities(self, model: str, images: np.ndarray, batch_size: int) -> np.ndarray:
        torch, features = self.torch, self.features
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(images), batch_size):
                rgb = features._rgb(images[start : start + batch_size], self.device)
                if model == "echoprime":
                    backbone, head = self.models[model]
                    logits = head(backbone(features._normalized(rgb, features.ECHOPRIME_MEAN, features.ECHOPRIME_STD)))
                else:
                    network = self.models[model]
                    logits = network.dense(network.forward_features(rgb))
                outputs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        return np.concatenate(outputs).astype(np.float64)


# ---------------------------------------------------------------------------------------------------------------------
# Driver


def load_splits(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as stream:
        return {row["exam_id"]: str(row.get("five_fold_split", "")) for row in csv.DictReader(stream)}


def discover_jobs(root: Path) -> list[Job]:
    catalog = load_croissant(root / "croissant.json", root=root)
    jobs = {
        (record.exam_id, record.recording_id): Job(record.exam_id, record.recording_id, record.zarr_path)
        for record in catalog.recordings
        if record.has_array_path(BMODE_PATH)
    }
    return [jobs[key] for key in sorted(jobs)]


def existing_rows(path: Path, columns: Sequence[str]) -> set[tuple[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return set()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if list(reader.fieldnames or ()) != list(columns):
            raise SystemExit(
                f"{path} has different columns (other --models?); choose another --output-dir or delete it"
            )
        return {(row["exam_id"], row["recording_id"]) for row in reader}


def build_rows(
    batch: Sequence[Loaded],
    classifiers: ImageClassifiers | None,
    models: Sequence[str],
    **kw: Any,
):
    """Classify one batch of loaded recordings and yield CSV rows."""
    splits: Mapping[str, str] = kw["splits"]
    probabilities: dict[str, list[np.ndarray | None]] = {}
    if classifiers is not None:
        for model in classifiers.models:
            stacks = [getattr(item, model) for item in batch]
            valid = [stack for stack in stacks if stack is not None]
            per_frame = (
                classifiers.probabilities(model, np.concatenate(valid), kw["batch_size"]) if valid else np.empty(0)
            )
            offset, results = 0, []
            for stack in stacks:
                if stack is None:
                    results.append(None)
                    continue
                results.append(per_frame[offset : offset + len(stack)].mean(axis=0))
                offset += len(stack)
            probabilities[model] = results
    for index, item in enumerate(batch):
        row = {
            "exam_id": item.job.exam_id,
            "recording_id": item.job.recording_id,
            "zarr_path": item.job.zarr_path,
            "five_fold_split": splits.get(item.job.exam_id, ""),
            "n_frames": str(item.n_frames) if item.n_frames else "",
            "frames_used": ";".join(str(frame) for frame in item.frames_used),
            "skip_reason": item.skip_reason,
        }
        if "native" in models:
            row.update(native_columns(item.native_scores))
        for model, results in probabilities.items():
            if results[index] is not None:
                row.update(model_columns(model, results[index]))
        yield row


def _batches(iterator: Iterator[Loaded], frames_per_batch: int) -> Iterator[list[Loaded]]:
    batch: list[Loaded] = []
    frames = 0
    for item in iterator:
        batch.append(item)
        frames += max(len(item.frames_used), 1)
        if frames >= frames_per_batch:
            yield batch
            batch, frames = [], 0
    if batch:
        yield batch


def sort_output(path: Path, columns: Sequence[str]) -> None:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows.sort(key=lambda row: (row["exam_id"], row["recording_id"]))
    temporary = path.with_suffix(".sorting.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="EchoXFlow root (croissant.json + exams/).",
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=None,
        help="exam_id,five_fold_split CSV (joined on exam_id).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for view_classifier_probabilities.csv."
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames-per-recording", type=int, default=8)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only the first N eligible recordings (sorted ids).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_CHOICES,
        default=["native", "echoprime", "echofine"],
        help="Classifiers to run.",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="GPU batch size in frames.")
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="CPU worker processes for loading/scan conversion.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    models = tuple(model for model in MODEL_CHOICES if model in args.models)
    root = data_root(args.data_root)
    split_csv = args.split_csv or (root / "splits.csv")
    splits = load_splits(split_csv)
    columns = csv_columns(models)
    out: Path = args.output_dir.expanduser() / "view_classifier_probabilities.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    jobs = discover_jobs(root)
    eligible = jobs[: args.limit] if args.limit is not None else jobs
    done = existing_rows(out, columns)
    todo = [job for job in eligible if (job.exam_id, job.recording_id) not in done]
    print(
        f"{len(jobs)} single-plane B-mode recordings; {len(eligible)} selected; {len(eligible) - len(todo)} already in"
        f" {out}; {len(todo)} to do. splits: {len(splits)} exams from {split_csv if splits else 'none'}",
        flush=True,
    )
    if todo:
        context = mp.get_context("fork")
        with context.Pool(
            max(1, args.workers),
            initializer=_init_worker,
            initargs=(str(root), models, args.frames_per_recording),
        ) as pool:
            classifiers = ImageClassifiers(models, args.device) if {"echoprime", "echofine"} & set(models) else None
            new_file = not out.is_file() or out.stat().st_size == 0
            with out.open("a", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                if new_file:
                    writer.writeheader()
                processed, skipped, loop_start = 0, 0, time.perf_counter()
                loaded = pool.imap(load_job, todo, chunksize=2)
                for batch in _batches(loaded, max(1, args.batch_size)):
                    for row in build_rows(
                        batch,
                        classifiers,
                        models,
                        splits=splits,
                        batch_size=args.batch_size,
                    ):
                        writer.writerow(row)
                        skipped += bool(row["skip_reason"])
                    stream.flush()
                    processed += len(batch)
                    elapsed = time.perf_counter() - loop_start
                    rate = processed / max(elapsed, 1e-9)
                    eta_h = (len(todo) - processed) / max(rate, 1e-9) / 3600
                    print(
                        f"{processed}/{len(todo)} recordings, {rate:.2f} rec/s, skipped {skipped}, ETA {eta_h:.2f} h",
                        flush=True,
                    )
    if args.limit is None and len(existing_rows(out, columns)) >= len(jobs):
        sort_output(out, columns)
        print(f"complete; sorted {out}", flush=True)
    print(f"wrote {out} in {time.perf_counter() - started:.0f} s", flush=True)
    summary = {
        "frames_per_recording": args.frames_per_recording,
        "models": list(models),
        "data_root": str(root),
    }
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
