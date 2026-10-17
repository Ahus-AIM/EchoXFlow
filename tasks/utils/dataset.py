"""Shared dataset helpers.

Recording cache: ``data.recording_cache_dir`` (or ``data.recording_cache: true``
for ``outputs/cache/recordings``) stores
each complete pre-resize training array once under ``arrays/``; windows are sliced from the cached arrays. By default
only the task's training arrays are cached (B-mode plus the task's targets, including their stored locations).
``data.recording_cache_read_only`` reads without writing misses; ``data.recording_cache_include`` and
``data.recording_cache_exclude`` take shell-style path patterns.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import random
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence, Sized
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler, SequentialSampler
from tqdm.auto import tqdm

from echoxflow import RecordingRecord, load_exam_folds, open_recording
from echoxflow import data_root as resolve_data_root
from echoxflow.scan import CartesianGrid
from echoxflow.scan.geometry import SectorGeometry
from tasks.utils.batching import attach_batch_items, collated_items
from tasks.utils.deferred import DeferredBatch, DeferredSample
from tasks.utils.seed import seed_everything, seeded_dataloader_kwargs

DEFAULT_FOLD_SPLIT_FILENAME = "splits.csv"
DEFAULT_VALIDATION_FOLD = 0
DEFAULT_SELECTION_SEED = 0
EXAM_IDS_ATTR = "exam_ids"
DEFAULT_RECORDING_CACHE_DIR = Path("outputs/cache/recordings")


@dataclass(frozen=True)
class SkippedCase:
    case_id: str
    reason: str


class HasRecord(Protocol):
    @property
    def record(self) -> RecordingRecord:
        raise NotImplementedError


class CaseIndexedDataset(Protocol):
    sample_indices_by_case: dict[str, list[int]]

    def __len__(self) -> int:
        raise NotImplementedError


SampleT = TypeVar("SampleT")
ItemT = TypeVar("ItemT")


@dataclass(frozen=True)
class SharedTrainingTransform:
    pass


def shared_training_transform_from_config(data_config: Mapping[str, object]) -> SharedTrainingTransform:
    del data_config
    return SharedTrainingTransform()


class ShuffledCaseSampler(Sampler[int]):
    def __init__(self, dataset: CaseIndexedDataset, *, seed: int | None = None) -> None:
        self.dataset = dataset
        self.seed = None if seed is None else int(seed)
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        rng = random if self.seed is None else random.Random(self.seed + self.epoch)
        self.epoch += 1
        remaining = {case_id: list(indices) for case_id, indices in self.dataset.sample_indices_by_case.items()}
        for indices in remaining.values():
            rng.shuffle(indices)
        while remaining:
            case_ids = list(remaining)
            rng.shuffle(case_ids)
            for case_id in case_ids:
                case_indices = remaining.get(case_id)
                if not case_indices:
                    continue
                yield case_indices.pop()
                if not case_indices:
                    del remaining[case_id]

    def __len__(self) -> int:
        return len(self.dataset)


class OneSamplePerCaseSampler(Sampler[int]):
    def __init__(
        self,
        dataset: CaseIndexedDataset,
        *,
        seed: int | None = None,
        shuffle: bool = True,
        randomize: bool = True,
    ) -> None:
        self.dataset = dataset
        self.seed = None if seed is None else int(seed)
        self.shuffle = bool(shuffle)
        self.randomize = bool(randomize)
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        rng = random if self.seed is None else random.Random(self.seed + self.epoch)
        self.epoch += 1
        grouped = [
            (case_id, list(indices)) for case_id, indices in self.dataset.sample_indices_by_case.items() if indices
        ]
        if self.shuffle:
            rng.shuffle(grouped)
        for _case_id, indices in grouped:
            if self.randomize:
                yield indices[rng.randrange(len(indices))]
            else:
                yield indices[len(indices) // 2]

    def __len__(self) -> int:
        return sum(1 for indices in self.dataset.sample_indices_by_case.values() if indices)


def _sample_selection(value: object, *, name: str) -> str:
    selection = "one_per_case" if value is None else str(value).strip().lower()
    if selection not in {"one_per_case", "all"}:
        raise ValueError(f"{name} must be 'one_per_case' or 'all', got {value!r}")
    return selection


def sample_indices_by_case(refs: Sequence[HasRecord]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for index, ref in enumerate(refs):
        grouped.setdefault(f"{ref.record.exam_id}/{ref.record.recording_id}", []).append(index)
    return grouped


def progress_iter(
    items: Iterable[ItemT],
    *,
    description: str,
    total: int | None = None,
    unit: str = "item",
) -> Iterator[ItemT]:
    from tasks.utils.training.progress import interactive_output

    with tqdm(
        items,
        desc=description,
        total=total,
        unit=unit,
        leave=False,
        dynamic_ncols=True,
        disable=not interactive_output(),
    ) as progress:
        for item in progress:
            yield item


def discover_records_from_candidates(
    candidates: Sequence[RecordingRecord],
    *,
    description: str,
    exam_ids: Iterable[str] | None = None,
    max_cases: int | None = None,
    sample_fraction: object | None = None,
    seed: int | None = None,
    predicate: Callable[[RecordingRecord], bool] | None = None,
) -> list[RecordingRecord]:
    found = records_matching_exam_ids(candidates, exam_ids)
    found = tuple(limit_records_by_fraction(found, sample_fraction, seed=seed))
    limit = None if max_cases is None else int(max_cases)
    records: list[RecordingRecord] = []
    for record in progress_iter(
        found,
        description=description,
        total=len(found),
        unit="recording",
    ):
        if predicate is not None and not predicate(record):
            continue
        records.append(record)
        if limit is not None and len(records) >= limit:
            break
    return records


def records_matching_exam_ids(
    records: Sequence[RecordingRecord],
    exam_ids: Iterable[str] | None,
) -> tuple[RecordingRecord, ...]:
    if exam_ids is None:
        return tuple(records)
    wanted_exam_ids = {str(exam_id) for exam_id in exam_ids}
    return tuple(record for record in records if record.exam_id in wanted_exam_ids)


def resize_stack(frames: np.ndarray, shape: tuple[int, int], *, interpolation: str = "linear") -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(frames, dtype=np.float32))[:, None]
    if interpolation == "linear":
        resized = F.interpolate(tensor, size=shape, mode="bilinear", align_corners=False)
    elif interpolation == "nearest":
        resized = F.interpolate(tensor, size=shape, mode="nearest")
    else:
        raise ValueError(f"Unsupported stack interpolation: {interpolation}")
    return np.asarray(resized[:, 0].numpy(), dtype=np.float32)


def training_transform_from_args(
    *, training_transform: SharedTrainingTransform | None
) -> SharedTrainingTransform | None:
    return training_transform


class _VelocityStream(Protocol):
    @property
    def metadata(self) -> object: ...

    @property
    def data(self) -> np.ndarray: ...


def cartesian_grid_sample_grid(
    geometry: SectorGeometry,
    grid: CartesianGrid,
    *,
    source_shape: tuple[int, int],
    output_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    out_h, out_w = grid.shape if output_shape is None else (int(output_shape[0]), int(output_shape[1]))
    src_h, src_w = int(source_shape[0]), int(source_shape[1])
    xs = np.linspace(grid.x_range_m[0], grid.x_range_m[1], out_w, dtype=np.float64)
    ys = np.linspace(grid.y_range_m[0], grid.y_range_m[1], out_h, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    radii = np.sqrt(xx**2 + yy**2)
    angles = np.arctan2(xx, yy)
    rows = (radii - geometry.depth_start_m) / geometry.depth_span_m * max(0, src_h - 1)
    cols = (angles - geometry.angle_start_rad) / geometry.angle_span_rad * max(0, src_w - 1)
    x = cols / max(1, src_w - 1) * 2.0 - 1.0
    y = rows / max(1, src_h - 1) * 2.0 - 1.0
    return np.stack([x, y], axis=-1).astype(np.float32)


def cartesian_image_sample_grid(grid: CartesianGrid) -> np.ndarray:
    out_h, out_w = grid.shape
    x = np.linspace(-1.0, 1.0, out_w, dtype=np.float64)
    y = np.linspace(-1.0, 1.0, out_h, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)
    return np.stack([xx, yy], axis=-1).astype(np.float32)


def timestamps_or_index(timestamps: np.ndarray | None, count: int) -> np.ndarray:
    if timestamps is None:
        return np.arange(int(count), dtype=np.float64)
    return np.asarray(timestamps, dtype=np.float64)


def velocity_limit_from_stream(stream: _VelocityStream) -> float:
    metadata = stream.metadata
    velocity_limit_mps = getattr(metadata, "velocity_limit_mps", None)
    if velocity_limit_mps is not None:
        return float(velocity_limit_mps)
    value_range = getattr(metadata, "value_range", None)
    if value_range is not None:
        return max(abs(float(value_range[0])), abs(float(value_range[1])))
    return max(float(np.nanmax(np.abs(stream.data))), 1e-6)


def cartesian_sector_mask(
    geometry: SectorGeometry,
    grid: CartesianGrid,
    *,
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Boolean ``[H,W]`` mask of the Cartesian grid pixels that lie inside the ultrasound sector."""
    out_h, out_w = grid.shape if shape is None else (int(shape[0]), int(shape[1]))
    xs = np.linspace(grid.x_range_m[0], grid.x_range_m[1], out_w, dtype=np.float64)
    ys = np.linspace(grid.y_range_m[0], grid.y_range_m[1], out_h, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    radii = np.sqrt(xx**2 + yy**2)
    angles = np.arctan2(xx, yy)
    return cast(
        np.ndarray,
        (
            (radii >= geometry.depth_start_m)
            & (radii <= geometry.depth_end_m)
            & (angles >= geometry.angle_start_rad)
            & (angles <= geometry.angle_end_rad)
        ),
    )


def minmax_normalize(frames: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(frames, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    low = float(np.min(finite))
    high = float(np.max(finite))
    value_range = high - low
    if not np.isfinite(value_range) or value_range < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return np.asarray(np.clip((arr - low) / value_range, 0.0, 1.0), dtype=np.float32)


def target_timestamps(timestamps: np.ndarray | None, start: int, stop: int, target_count: int) -> np.ndarray:
    sliced = slice_timestamps(timestamps, start, stop)
    if sliced.size >= 2:
        return np.linspace(float(sliced[0]), float(sliced[-1]), target_count + 2, dtype=np.float32)[1:-1]
    return np.arange(target_count, dtype=np.float32)


def slice_timestamps(timestamps: np.ndarray | None, start: int, stop: int) -> np.ndarray:
    if timestamps is None:
        return np.arange(stop - start, dtype=np.float32)
    return np.asarray(timestamps[start:stop], dtype=np.float32)


def selected_timestamps(timestamps: np.ndarray | None, indices: np.ndarray) -> np.ndarray:
    if timestamps is None:
        return np.arange(indices.size, dtype=np.float32)
    return np.asarray(timestamps[indices], dtype=np.float32)


def fps_from_timestamps(timestamps: np.ndarray | None) -> float | None:
    if timestamps is None or len(timestamps) < 2:
        return None
    diffs = np.diff(np.asarray(timestamps, dtype=np.float64))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    return None if diffs.size == 0 else float(1.0 / np.median(diffs))


def ordered_float_pair(value: object, *, name: str = "value") -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-value float range, got {value!r}")
    try:
        left = float(value[0])
        right = float(value[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite numeric values, got {value!r}") from exc
    if not np.isfinite(left) or not np.isfinite(right):
        raise ValueError(f"{name} must contain finite numeric values, got {value!r}")
    return (left, right) if left <= right else (right, left)


def optional_ordered_float_pair(value: object, *, name: str = "value") -> tuple[float, float] | None:
    if value is None or value == "":
        return None
    return ordered_float_pair(value, name=name)


def two_dimensional_frame_count(
    record: RecordingRecord, data_root: str | Path | None, data_path: str = "data/2d_brightness_mode"
) -> int:
    count = record.frame_counts_by_content_type.get(data_path.removeprefix("data/"))
    if count is not None:
        return int(count)
    # The stream header has the stored shape without loading the samples.
    header = open_recording(record, root=data_root).load_stream_header(data_path)
    return int(header.shape[0])


def recording_cache_kwargs(data_config: Mapping[str, object]) -> dict[str, object]:
    cache_dir = _recording_cache_dir(data_config)
    if cache_dir is None:
        return {}
    return {
        "recording_cache_dir": cache_dir,
        "recording_cache_read_only": _config_bool(data_config.get("recording_cache_read_only", False)),
        "recording_cache_include": _config_string_tuple(data_config.get("recording_cache_include")),
        "recording_cache_exclude": _config_string_tuple(data_config.get("recording_cache_exclude")),
    }


def split_records(
    records: Sequence[RecordingRecord],
    *,
    val_fraction: object,
    seed: int | None = None,
    fold_split_path: str | Path | None = None,
    validation_fold: object = DEFAULT_VALIDATION_FOLD,
) -> tuple[list[RecordingRecord], list[RecordingRecord]]:
    if fold_split_path is not None:
        return split_records_by_fold(records, fold_split_path=fold_split_path, validation_fold=validation_fold)
    ordered = list(records)
    if seed is not None:
        random.Random(int(seed)).shuffle(ordered)
    split = max(1, int(round(len(ordered) * (1.0 - float(cast(Any, val_fraction))))))
    train_records = list(ordered[:split])
    val_records = list(ordered[split:]) or train_records[:1]
    return train_records, val_records


def split_records_by_fold(
    records: Sequence[RecordingRecord],
    *,
    fold_split_path: str | Path,
    validation_fold: object = DEFAULT_VALIDATION_FOLD,
) -> tuple[list[RecordingRecord], list[RecordingRecord]]:
    fold_by_exam_id = load_exam_folds(fold_split_path)
    val_fold = int(cast(Any, validation_fold))
    train_records: list[RecordingRecord] = []
    val_records: list[RecordingRecord] = []
    missing_exam_ids = sorted({record.exam_id for record in records if record.exam_id not in fold_by_exam_id})
    if missing_exam_ids:
        preview = ", ".join(missing_exam_ids[:5])
        suffix = "" if len(missing_exam_ids) <= 5 else f", ... ({len(missing_exam_ids)} total)"
        raise ValueError(f"Fold split is missing exam_id values: {preview}{suffix}")
    for record in records:
        if fold_by_exam_id[record.exam_id] == val_fold:
            val_records.append(record)
        else:
            train_records.append(record)
    if not train_records:
        raise ValueError(f"Fold split {fold_split_path} leaves no training records for validation fold {val_fold}")
    if not val_records:
        raise ValueError(f"Fold split {fold_split_path} leaves no validation records for validation fold {val_fold}")
    return train_records, val_records


def selected_fold_split_exam_ids(
    fold_by_exam_id: Mapping[str, int],
    *,
    validation_fold: object = DEFAULT_VALIDATION_FOLD,
    fraction: object | None = None,
    seed: int | None = None,
    selection_fraction: object = 0.0,
    selection_seed: int = DEFAULT_SELECTION_SEED,
) -> set[str]:
    """Return the exam ids used for one fold, optionally limited by ``fraction``.

    With ``selection_fraction > 0`` the model-selection exams (see :func:`is_selection_exam`) are drawn from the
    training folds first, and ``fraction`` then limits training, selection, and validation exams separately, so a
    small smoke fraction still keeps a non-empty selection split.
    """
    val_fold = int(cast(Any, validation_fold))
    train_exam_ids = [exam_id for exam_id, fold in fold_by_exam_id.items() if int(fold) != val_fold]
    val_exam_ids = [exam_id for exam_id, fold in fold_by_exam_id.items() if int(fold) == val_fold]
    if not train_exam_ids:
        raise ValueError(f"Fold split leaves no training exams for validation fold {val_fold}")
    if not val_exam_ids:
        raise ValueError(f"Fold split leaves no validation exams for validation fold {val_fold}")
    selection_value = validated_selection_fraction(selection_fraction)
    selection_exam_ids = [
        exam_id
        for exam_id in train_exam_ids
        if is_selection_exam(exam_id, fraction=selection_value, seed=selection_seed)
    ]
    selected = set(selection_exam_ids)
    train_exam_ids = [exam_id for exam_id in train_exam_ids if exam_id not in selected]
    train_exam_ids = limit_items_by_fraction(train_exam_ids, fraction, seed=seed)
    selection_exam_ids = limit_items_by_fraction(selection_exam_ids, fraction, seed=seed)
    val_exam_ids = limit_items_by_fraction(val_exam_ids, fraction, seed=seed)
    return set(train_exam_ids) | set(selection_exam_ids) | set(val_exam_ids)


def validated_selection_fraction(value: object) -> float:
    """Return ``data.selection_fraction`` as a float in ``[0, 1)``; ``None`` means disabled (0)."""
    fraction = 0.0 if value is None else float(cast(Any, value))
    if not np.isfinite(fraction) or not 0.0 <= fraction < 1.0:
        raise ValueError(f"data.selection_fraction must be in [0, 1), got {value!r}")
    return fraction


def is_selection_exam(exam_id: str, *, fraction: float, seed: int = DEFAULT_SELECTION_SEED) -> bool:
    """Whether a training exam belongs to the inner model-selection split.

    The rule hashes ``f"{seed}:{exam_id}"`` with SHA-1 and keeps the exam when the hash, read as a fraction of
    ``2**160``, is below ``fraction``. It depends only on the exam id and ``seed`` (default 0, independent of the
    training seed), so every method and every run of one fold shares the same selection exams.
    """
    if fraction <= 0.0:
        return False
    digest = hashlib.sha1(f"{int(seed)}:{exam_id}".encode()).hexdigest()
    return int(digest, 16) / 2**160 < float(fraction)


def split_selection_records(
    train_records: Sequence[RecordingRecord],
    *,
    selection_fraction: object,
    selection_seed: int = DEFAULT_SELECTION_SEED,
) -> tuple[list[RecordingRecord], list[RecordingRecord]]:
    """Move the records of selection exams out of ``train_records``; return ``(train, selection)``."""
    fraction = validated_selection_fraction(selection_fraction)
    train: list[RecordingRecord] = []
    selection: list[RecordingRecord] = []
    for record in train_records:
        if is_selection_exam(record.exam_id, fraction=fraction, seed=selection_seed):
            selection.append(record)
        else:
            train.append(record)
    return train, selection


def require_disjoint_exam_splits(splits: Mapping[str, Iterable[str]]) -> None:
    """Raise when any two named exam-id sets overlap."""
    named = [(name, set(exam_ids)) for name, exam_ids in splits.items()]
    for index, (left_name, left) in enumerate(named):
        for right_name, right in named[index + 1 :]:
            overlap = sorted(left & right)
            if overlap:
                preview = ", ".join(overlap[:5])
                raise ValueError(f"{left_name} and {right_name} exams overlap ({len(overlap)}): {preview}")


def dataloader_exam_ids(dataloader: object) -> frozenset[str] | None:
    """Exam ids of the records behind a loader built by :func:`build_task_dataloaders`."""
    exam_ids = getattr(dataloader, EXAM_IDS_ATTR, None)
    return None if exam_ids is None else frozenset(str(exam_id) for exam_id in exam_ids)


def exam_split_summary(*, train: object, val: object | None, selection: object | None) -> dict[str, object]:
    """Exam counts of the train, selection, and validation loaders, after checking they are pairwise disjoint."""
    named = {"train": train, "selection": selection, "val": val}
    exam_ids = {name: dataloader_exam_ids(loader) for name, loader in named.items() if loader is not None}
    require_disjoint_exam_splits({name: ids for name, ids in exam_ids.items() if ids is not None})
    summary: dict[str, object] = {}
    for name in named:
        ids = exam_ids.get(name)
        summary[f"n_{name}_exams"] = None if ids is None else len(ids)
    selection_ids = exam_ids.get("selection")
    summary["selection_exam_ids"] = [] if selection_ids is None else sorted(selection_ids)
    return summary


def resolve_fold_split_path(root: str | Path, data_config: Mapping[str, object]) -> Path | None:
    configured = data_config.get("fold_split_path")
    if configured is not None and (
        configured is False or str(configured).strip().lower() in {"false", "none", "off", "disabled"}
    ):
        return None
    if configured is not None and str(configured).strip():
        path = Path(str(configured)).expanduser()
        if not path.is_absolute():
            path = Path(root) / path
        if not path.exists():
            raise FileNotFoundError(f"Fold split CSV not found: {path}")
        return path
    path = Path(root) / DEFAULT_FOLD_SPLIT_FILENAME
    return path if path.exists() else None


def limit_records_by_fraction(
    records: Sequence[RecordingRecord],
    fraction: object | None,
    *,
    seed: int | None = None,
) -> list[RecordingRecord]:
    return limit_items_by_fraction(records, fraction, seed=seed)


def limit_items_by_fraction(
    items: Sequence[ItemT],
    fraction: object | None,
    *,
    seed: int | None = None,
) -> list[ItemT]:
    ordered = list(items)
    if fraction is None:
        return ordered
    value = float(cast(Any, fraction))
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"data.data_fraction must be a positive finite value, got {fraction!r}")
    if value >= 1.0:
        return ordered
    if seed is not None:
        random.Random(int(seed)).shuffle(ordered)
    return ordered[: _fraction_count(len(ordered), value)]


def _fraction_count(total: int, fraction: float) -> int:
    if total <= 0 or fraction <= 0.0:
        return 0
    return min(total, max(1, int(round(total * fraction))))


def dataset_root(root: object) -> str | Path:
    if root is None:
        return resolve_data_root()
    return root if isinstance(root, (str, Path)) else str(root)


def dataloader_kwargs(data_config: Mapping[str, object]) -> dict[str, Any]:
    _configure_multiprocessing_sharing_strategy(data_config)
    num_workers = int(cast(Any, data_config.get("num_workers", 0)))
    kwargs: dict[str, Any] = {
        "batch_size": None,
        "num_workers": num_workers,
    }
    if "pin_memory" in data_config:
        kwargs["pin_memory"] = bool(data_config.get("pin_memory"))
    if num_workers > 0 and "persistent_workers" in data_config:
        kwargs["persistent_workers"] = bool(data_config.get("persistent_workers"))
    kwargs.update(seeded_dataloader_kwargs(int(cast(Any, data_config.get("seed", 0)))))
    return kwargs


def shutdown_dataloader_workers(*loaders: object) -> None:
    """Stop the worker processes that persistent-worker data loaders keep between epochs.

    A ``DataLoader`` with ``persistent_workers=True`` keeps its iterator (and its worker processes) alive after an
    epoch so the next epoch starts without respawning workers. Call this when a loader is no longer iterated, so the
    workers exit now instead of whenever the loader is garbage collected. A later ``iter(loader)`` starts new
    workers. Other loaders and ``None`` are ignored.
    """
    for loader in loaders:
        if not isinstance(loader, DataLoader):
            continue
        iterator = getattr(loader, "_iterator", None)
        if iterator is None:
            continue
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown):
            shutdown()
        loader._iterator = None


def _configure_multiprocessing_sharing_strategy(data_config: Mapping[str, object]) -> None:
    value = data_config.get("multiprocessing_sharing_strategy")
    if value is None or value == "":
        return
    strategy = str(value)
    available = torch.multiprocessing.get_all_sharing_strategies()
    if strategy not in available:
        choices = ", ".join(sorted(available))
        raise ValueError(f"Unsupported multiprocessing_sharing_strategy {strategy!r}; expected one of: {choices}")
    if torch.multiprocessing.get_sharing_strategy() != strategy:
        torch.multiprocessing.set_sharing_strategy(strategy)


def batched_dataloader_kwargs(
    data_config: Mapping[str, object],
    *,
    batch_size: int | None,
) -> dict[str, Any]:
    kwargs = dataloader_kwargs(data_config)
    if batch_size is None:
        return kwargs
    kwargs["batch_size"] = int(batch_size)
    kwargs["collate_fn"] = collate_samples
    if kwargs["num_workers"] > 0 and data_config.get("prefetch_factor") is not None:
        kwargs["prefetch_factor"] = int(cast(Any, data_config["prefetch_factor"]))
    return kwargs


def collate_samples(samples: Sequence[SampleT]) -> SampleT:
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    first = samples[0]
    if isinstance(first, DeferredSample):
        return cast(SampleT, DeferredBatch(tuple(cast(Sequence[DeferredSample], samples))))
    if not is_dataclass(first):
        raise TypeError(f"Expected dataclass samples, got {type(first).__name__}")
    values: dict[str, object] = {}
    for field in fields(first):
        field_values = [getattr(sample, field.name) for sample in samples]
        first_value = field_values[0]
        if isinstance(first_value, torch.Tensor):
            values[field.name] = _cat_tensors_with_padding(cast(Sequence[torch.Tensor], field_values))
        elif field.name == "sample_id":
            values[field.name] = "|".join(str(value) for value in field_values)
        else:
            values[field.name] = first_value
    batch = cast(Any, type(first))(**values)
    attach_batch_items(batch, collated_items(samples))
    return cast(SampleT, batch)


def _cat_tensors_with_padding(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    if not tensors:
        raise ValueError("Cannot collate an empty tensor sequence")
    shapes = [tuple(int(dim) for dim in tensor.shape) for tensor in tensors]
    if len(set(shapes)) == 1:
        return torch.cat(tuple(tensors), dim=0)
    ndim = tensors[0].ndim
    if any(tensor.ndim != ndim for tensor in tensors):
        raise ValueError(f"Cannot collate tensors with different ranks: {shapes}")
    max_shape = tuple(max(shape[dim] for shape in shapes) for dim in range(ndim))
    padded: list[torch.Tensor] = []
    for tensor in tensors:
        out = tensor.new_zeros(max_shape)
        slices = tuple(slice(0, int(size)) for size in tensor.shape)
        out[slices] = tensor
        padded.append(out)
    return torch.cat(tuple(padded), dim=0)


def build_task_dataloaders(
    *,
    config: Any,
    data_root: str | Path | None,
    task_name: str,
    dataset_cls: Any,
    discover_records: Callable[..., list[RecordingRecord]],
    empty_message: str,
    discover_kwargs_fn: Callable[[Mapping[str, object]], Mapping[str, object]] | None = None,
    dataset_kwargs_fn: Callable[[Any, Mapping[str, object], str | Path], Mapping[str, object]] | None = None,
    train_dataset_kwargs_fn: Callable[[Mapping[str, object]], Mapping[str, object]] | None = None,
    val_dataset_kwargs_fn: Callable[[Mapping[str, object]], Mapping[str, object]] | None = None,
    train_batch_size_fn: Callable[[Mapping[str, object]], int | None] | None = None,
    val_batch_size_fn: Callable[[Mapping[str, object]], int | None] | None = None,
    extra_fields_fn: (
        Callable[[Mapping[str, object], Mapping[str, object], object, object], Mapping[str, object]] | None
    ) = None,
) -> tuple[DataLoader[SampleT], ...]:
    """Build ``(train, val)`` loaders, or ``(train, val, selection)`` when ``data.selection_fraction > 0``.

    The selection loader covers the inner model-selection exams drawn from the training folds (see
    :func:`is_selection_exam`); those exams are removed from training. It uses the validation dataset options,
    sampler, and batch size, so its per-clip loss is computed exactly like the reported validation loss. Selection
    only uses the loss, so its samples skip the Cartesian metric targets (``include_cartesian_metrics``), which only
    the validation metrics read.
    """
    data_config = config.section("data")
    seed = int(cast(Any, data_config.get("seed", 0)))
    selection_fraction = validated_selection_fraction(data_config.get("selection_fraction"))
    selection_seed = int(cast(Any, data_config.get("selection_seed", DEFAULT_SELECTION_SEED)))
    seed_everything(seed)
    root = data_root or data_config.get("root_dir")
    root_path = dataset_root(root)
    fold_split_path = resolve_fold_split_path(root_path, data_config)
    if fold_split_path is None:
        logging.getLogger(__name__).warning(
            "%s: no fold split file (data.fold_split_path or splits.csv in %s); using a random %s validation split",
            task_name,
            root_path,
            data_config.get("val_fraction", 0.2),
        )
    extra_discover_kwargs = dict(discover_kwargs_fn(data_config) if discover_kwargs_fn is not None else {})
    fraction_limited_discovery = (
        fold_split_path is None
        and data_config.get("data_fraction") is not None
        and _supports_keyword(discover_records, "sample_fraction")
    )
    if fold_split_path is not None:
        selected_exam_ids = selected_fold_split_exam_ids(
            load_exam_folds(fold_split_path),
            validation_fold=data_config.get("validation_fold", DEFAULT_VALIDATION_FOLD),
            fraction=data_config.get("data_fraction"),
            seed=seed,
            selection_fraction=selection_fraction,
            selection_seed=selection_seed,
        )
        extra_discover_kwargs.setdefault("exam_ids", selected_exam_ids)
    elif fraction_limited_discovery:
        extra_discover_kwargs.setdefault("sample_fraction", data_config.get("data_fraction"))
        extra_discover_kwargs.setdefault("seed", seed)
    records = discover_records(
        root_dir=root_path,
        max_cases=optional_int(data_config.get("max_files")),
        **extra_discover_kwargs,
    )
    if not records:
        raise ValueError(empty_message)
    if fold_split_path is None:
        if not fraction_limited_discovery:
            records = limit_records_by_fraction(records, data_config.get("data_fraction"), seed=seed)
        train_records, val_records = split_records(
            records,
            val_fraction=data_config.get("val_fraction", 0.2),
            seed=seed,
        )
    else:
        train_records, val_records = split_records(
            records,
            val_fraction=data_config.get("val_fraction", 0.2),
            seed=seed,
            fold_split_path=fold_split_path,
            validation_fold=data_config.get("validation_fold", DEFAULT_VALIDATION_FOLD),
        )
        records = train_records + val_records
    train_records, selection_records = split_selection_records(
        train_records,
        selection_fraction=selection_fraction,
        selection_seed=selection_seed,
    )
    if selection_fraction > 0.0 and not selection_records:
        raise ValueError(
            f"{task_name}: data.selection_fraction={selection_fraction} selected no training exams for model selection"
        )
    if not train_records:
        raise ValueError(f"{task_name}: no training records remain after removing the selection exams")
    require_disjoint_exam_splits(
        {
            "train": {record.exam_id for record in train_records},
            "selection": {record.exam_id for record in selection_records},
            "val": {record.exam_id for record in val_records},
        }
    )
    dataset_kwargs = dict(dataset_kwargs_fn(config, data_config, root_path) if dataset_kwargs_fn is not None else {})
    dataset_kwargs.update(recording_cache_kwargs(data_config))
    train_kwargs = {
        **dataset_kwargs,
        **dict(train_dataset_kwargs_fn(data_config) if train_dataset_kwargs_fn is not None else {}),
    }
    val_kwargs = {
        **dataset_kwargs,
        **dict(val_dataset_kwargs_fn(data_config) if val_dataset_kwargs_fn is not None else {}),
    }
    train_dataset = cast(
        Dataset[SampleT],
        dataset_cls(
            records=train_records,
            max_samples=optional_int(data_config.get("max_train_samples")),
            **train_kwargs,
        ),
    )
    val_dataset = cast(
        Dataset[SampleT],
        dataset_cls(
            records=val_records,
            max_samples=optional_int(data_config.get("max_val_samples")),
            **val_kwargs,
        ),
    )
    selection_kwargs = dict(val_kwargs)
    if "include_cartesian_metrics" in selection_kwargs:
        selection_kwargs["include_cartesian_metrics"] = False
    selection_dataset = (
        None
        if not selection_records
        else cast(
            Dataset[SampleT],
            dataset_cls(
                records=selection_records,
                max_samples=optional_int(data_config.get("max_val_samples")),
                **selection_kwargs,
            ),
        )
    )
    return build_train_val_dataloaders(
        task_name=task_name,
        root=root_path,
        records=records,
        train_records=train_records,
        val_records=val_records,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        selection_records=selection_records,
        selection_dataset=selection_dataset,
        data_config=data_config,
        seed=seed,
        train_batch_size=(
            train_batch_size_fn(data_config)
            if train_batch_size_fn is not None
            else optional_int(data_config.get("train_batch_size", data_config.get("batch_size")))
        ),
        val_batch_size=(
            val_batch_size_fn(data_config)
            if val_batch_size_fn is not None
            else optional_int(data_config.get("val_batch_size", data_config.get("batch_size")))
        ),
        extra_fields=(
            None
            if extra_fields_fn is None
            else extra_fields_fn(data_config, dataset_kwargs, train_dataset, val_dataset)
        ),
    )


def build_train_val_dataloaders(
    *,
    task_name: str,
    root: str | Path,
    records: Sequence[RecordingRecord],
    train_records: Sequence[RecordingRecord],
    val_records: Sequence[RecordingRecord],
    train_dataset: Dataset[SampleT],
    val_dataset: Dataset[SampleT],
    data_config: Mapping[str, object],
    seed: int,
    train_batch_size: int | None,
    val_batch_size: int | None,
    extra_fields: Mapping[str, object] | None = None,
    selection_records: Sequence[RecordingRecord] = (),
    selection_dataset: Dataset[SampleT] | None = None,
) -> tuple[DataLoader[SampleT], ...]:
    """Return ``(train, val)`` loaders, plus a selection loader when ``selection_dataset`` is given."""
    train_sample_selection = _sample_selection(
        data_config.get("train_sample_selection"),
        name="data.train_sample_selection",
    )
    val_sample_selection = _sample_selection(
        data_config.get("val_sample_selection"),
        name="data.val_sample_selection",
    )
    train_sampler: Sampler[int]
    if train_sample_selection == "all":
        train_sampler = ShuffledCaseSampler(cast(CaseIndexedDataset, train_dataset), seed=seed)
    else:
        train_sampler = OneSamplePerCaseSampler(cast(CaseIndexedDataset, train_dataset), seed=seed)
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        **batched_dataloader_kwargs(data_config, batch_size=train_batch_size),
    )
    val_loader = _evaluation_dataloader(
        val_dataset,
        sample_selection=val_sample_selection,
        data_config=data_config,
        seed=seed,
        batch_size=val_batch_size,
    )
    setattr(train_loader, EXAM_IDS_ATTR, frozenset(record.exam_id for record in train_records))
    setattr(val_loader, EXAM_IDS_ATTR, frozenset(record.exam_id for record in val_records))
    selection_loader = None
    if selection_dataset is not None:
        # The selection split is scored exactly like the reported validation split: same dataset options,
        # deterministic sampler, and batch size.
        selection_loader = _evaluation_dataloader(
            selection_dataset,
            sample_selection=val_sample_selection,
            data_config=data_config,
            seed=seed,
            batch_size=val_batch_size,
        )
        setattr(selection_loader, EXAM_IDS_ATTR, frozenset(record.exam_id for record in selection_records))
    log_dataset_summary(
        task_name=task_name,
        root=root,
        records=records,
        train_records=train_records,
        val_records=val_records,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        data_config=data_config,
        extra_fields={
            **dict(extra_fields or {}),
            **(
                {}
                if selection_loader is None
                else {
                    "selection_records": len(selection_records),
                    "selection_exams": len({record.exam_id for record in selection_records}),
                    "selection_samples": _safe_len(selection_dataset),
                    "selection_batches": _safe_len_or_na(selection_loader),
                }
            ),
        },
    )
    if selection_loader is None:
        return train_loader, val_loader
    return train_loader, val_loader, selection_loader


def _evaluation_dataloader(
    dataset: Dataset[SampleT],
    *,
    sample_selection: str,
    data_config: Mapping[str, object],
    seed: int,
    batch_size: int | None,
) -> DataLoader[SampleT]:
    sampler: Sampler[int]
    if sample_selection == "all":
        sampler = SequentialSampler(cast(Sized, dataset))
    else:
        sampler = OneSamplePerCaseSampler(
            cast(CaseIndexedDataset, dataset),
            seed=seed,
            shuffle=False,
            randomize=False,
        )
    return DataLoader(dataset, sampler=sampler, **batched_dataloader_kwargs(data_config, batch_size=batch_size))


def log_dataset_summary(
    *,
    task_name: str,
    root: str | Path,
    records: Sequence[RecordingRecord],
    train_records: Sequence[RecordingRecord],
    val_records: Sequence[RecordingRecord],
    train_dataset: object,
    val_dataset: object,
    train_loader: object,
    val_loader: object,
    data_config: Mapping[str, object],
    extra_fields: Mapping[str, object] | None = None,
) -> None:
    extra = "" if not extra_fields else " " + " ".join(f"{key}={value}" for key, value in extra_fields.items())
    logging.getLogger(f"tasks.{task_name}.dataset").info(
        (
            "%s data root=%s records=%d train_records=%d val_records=%d "
            "samples=%d train_samples=%d val_samples=%d "
            "train_batches=%s val_batches=%s "
            "max_files=%s data_fraction=%s max_train_samples=%s max_val_samples=%s%s"
        ),
        task_name,
        root,
        len(records),
        len(train_records),
        len(val_records),
        _safe_len(train_dataset) + _safe_len(val_dataset),
        _safe_len(train_dataset),
        _safe_len(val_dataset),
        _safe_len_or_na(train_loader),
        _safe_len_or_na(val_loader),
        _limit_value(data_config.get("max_files")),
        _limit_value(data_config.get("data_fraction")),
        _limit_value(data_config.get("max_train_samples")),
        _limit_value(data_config.get("max_val_samples")),
        extra,
    )


def _supports_keyword(func: Callable[..., object], keyword: str) -> bool:
    try:
        parameters = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(parameter.kind == inspect.Parameter.VAR_KEYWORD or parameter.name == keyword for parameter in parameters)


def _safe_len(value: object) -> int:
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return 0


def _safe_len_or_na(value: object) -> str:
    try:
        return str(len(value))  # type: ignore[arg-type]
    except TypeError:
        return "n/a"


def _limit_value(value: object) -> str:
    return "none" if value is None else str(value)


def _recording_cache_dir(data_config: Mapping[str, object]) -> Path | None:
    enabled = data_config.get("recording_cache")
    configured_dir = data_config.get("recording_cache_dir")
    if enabled is not None and not _config_bool(enabled):
        return None
    if configured_dir is None or str(configured_dir).strip() == "":
        return DEFAULT_RECORDING_CACHE_DIR if _config_bool(enabled) else None
    return Path(str(configured_dir)).expanduser()


def _config_string_tuple(value: object) -> tuple[str, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return (str(value).strip(),)


def _config_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off", "none"}
    return bool(value)


def as_spatial_shape(value: tuple[int, int]) -> tuple[int, int]:
    result = tuple(int(item) for item in value)
    if len(result) != 2 or min(result) <= 0:
        raise ValueError(f"Expected positive 2D shape, got {value}")
    return result[0], result[1]


def spatial_shape_from_config(value: object) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"Expected 2D shape sequence, got {type(value).__name__}")
    result = tuple(int(cast(Any, item)) for item in value)
    if len(result) != 2:
        raise ValueError(f"Expected 2D shape, got {value}")
    return as_spatial_shape((result[0], result[1]))


def optional_int(value: object) -> int | None:
    return None if value is None else int(cast(Any, value))
