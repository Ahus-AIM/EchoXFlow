from __future__ import annotations

import csv
import hashlib
import inspect
import json
import logging
import os
import random
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from threading import get_ident
from typing import Any, Protocol, TypeVar, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm

from echoxflow import RecordingRecord
from echoxflow import data_root as resolve_data_root
from echoxflow import open_recording, resample_sector_stack, sector_lookup
from echoxflow.scan.geometry import SectorGeometry
from tasks.utils.seed import seed_everything, seeded_dataloader_kwargs

DEFAULT_FOLD_SPLIT_FILENAMES = ("splits.csv", "five_fold_split.csv")
DEFAULT_VALIDATION_FOLD = 0
DEFAULT_SAMPLE_CACHE_DIR = Path("outputs/cache/samples")
DEFAULT_RECORDING_CACHE_DIR = Path("outputs/cache/recordings")
_SAMPLE_CACHE_VERSION = 1
_SAMPLE_CACHE_CONTROL_KEYS = {
    "sample_cache",
    "sample_cache_dir",
    "sample_cache_read_only",
    "sample_cache_version",
    "recording_cache",
    "recording_cache_dir",
    "recording_cache_read_only",
    "recording_cache_include",
    "recording_cache_exclude",
}
_DATALOADER_ONLY_CONFIG_KEYS = {
    "batch_size",
    "train_batch_size",
    "val_batch_size",
    "num_workers",
    "prefetch_factor",
    "pin_memory",
    "persistent_workers",
    "multiprocessing_sharing_strategy",
}


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


RefT = TypeVar("RefT", bound=HasRecord)
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


class CachedDataset(Dataset[SampleT]):
    """On-disk sample cache wrapper for task datasets.

    Task datasets still own discovery and sample construction. This wrapper only
    memoizes completed ``__getitem__`` results, which keeps it independent of the
    concrete task sample schema.
    """

    def __init__(
        self,
        dataset: Dataset[SampleT],
        *,
        cache_dir: str | Path,
        namespace: str,
        item_keys: Sequence[str] | None = None,
        read_only: bool = False,
    ) -> None:
        self.dataset = dataset
        self.cache_dir = Path(cache_dir)
        self.namespace = str(namespace)
        self.read_only = bool(read_only)
        self.sample_indices_by_case = getattr(dataset, "sample_indices_by_case", {})
        self._root = self.cache_dir / "objects"
        self._item_keys = list(item_keys) if item_keys is not None else _dataset_item_cache_keys(dataset)

    def __len__(self) -> int:
        return _safe_len(self.dataset)

    def __getitem__(self, index: int) -> SampleT:
        cache_path = self._cache_path(index)
        if cache_path.exists():
            try:
                return cast(SampleT, _torch_load_sample(cache_path))
            except Exception:
                if self.read_only:
                    raise
        sample = self.dataset[index]
        if not self.read_only:
            try:
                self._write_sample(cache_path, sample)
            except (OSError, RuntimeError) as exc:
                logging.getLogger("tasks.dataset.cache").warning(
                    "could not write sample cache %s: %s",
                    cache_path,
                    exc,
                )
        return sample

    def __getattr__(self, name: str) -> Any:
        dataset = self.__dict__.get("dataset")
        if dataset is None:
            raise AttributeError(name)
        return getattr(dataset, name)

    def _cache_path(self, index: int) -> Path:
        key = self._item_keys[index] if 0 <= int(index) < len(self._item_keys) else str(index)
        return self._root / key[:2] / f"{key}.pt"

    def _write_sample(self, path: Path, sample: SampleT) -> None:
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{get_ident()}.tmp")
        try:
            torch.save(sample, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


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
    with tqdm(items, desc=description, total=total, unit=unit, leave=False, dynamic_ncols=True) as progress:
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


def build_window_sample_refs(
    *,
    records: Sequence[RecordingRecord],
    window_length: int,
    stride: int,
    min_frames_per_case: int = 1,
    data_root: str | Path | None,
    max_samples: int | None,
    frame_count_fn: Callable[[RecordingRecord, str | Path | None], int],
    make_ref: Callable[[RecordingRecord, int, int], RefT],
    description: str = "Scanning samples",
) -> list[RefT]:
    refs: list[RefT] = []
    min_count = max(int(window_length), int(min_frames_per_case))
    for record in progress_iter(records, description=description, total=len(records), unit="case"):
        frame_count = frame_count_fn(record, data_root)
        if frame_count < min_count:
            continue
        for start in range(0, frame_count - int(window_length) + 1, int(stride)):
            refs.append(make_ref(record, start, start + int(window_length)))
            if max_samples is not None and len(refs) >= int(max_samples):
                return refs
    return refs


def resize_stack(frames: np.ndarray, shape: tuple[int, int], *, interpolation: str = "linear") -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(frames, dtype=np.float32))[:, None]
    if interpolation == "linear":
        resized = F.interpolate(tensor, size=shape, mode="bilinear", align_corners=False)
    elif interpolation == "nearest":
        resized = F.interpolate(tensor, size=shape, mode="nearest")
    else:
        raise ValueError(f"Unsupported stack interpolation: {interpolation}")
    return np.asarray(resized[:, 0].numpy(), dtype=np.float32)


def preprocess_bmode_frames(
    bmode_data_or_stream: object,
    *,
    transform: SharedTrainingTransform | None,
    input_spatial_shape: tuple[int, int],
) -> np.ndarray:
    del transform
    if hasattr(bmode_data_or_stream, "to_float"):
        frames = cast(Any, bmode_data_or_stream).to_float().data
    else:
        frames = bmode_data_or_stream
    values = np.asarray(frames, dtype=np.float32)
    return minmax_normalize(resize_stack(values, input_spatial_shape))


def resample_to_reference_geometry(
    frames: np.ndarray,
    *,
    source_geometry: SectorGeometry | None,
    reference_geometry: SectorGeometry | None,
    target_shape: tuple[int, int],
    return_region_mask: bool = False,
    interpolation: str = "linear",
) -> np.ndarray | tuple[np.ndarray, np.ndarray | None]:
    if source_geometry is None or reference_geometry is None:
        values = resize_stack(frames, target_shape, interpolation=interpolation)
        return (values, None) if return_region_mask else values
    target_geometry = replace(reference_geometry, grid_shape=target_shape)
    values = np.asarray(
        resample_sector_stack(
            np.asarray(frames, dtype=np.float32),
            source_geometry,
            target_geometry,
            output_shape=target_shape,
            interpolation=cast(Any, interpolation),
        ),
        dtype=np.float32,
    )
    if not return_region_mask:
        return values
    lookup = sector_lookup(
        source_geometry,
        target_geometry,
        source_shape=np.asarray(frames).shape[1:3],
        output_shape=target_shape,
    )
    return values, np.asarray(lookup.mask, dtype=bool)


def resize_volume_window(window: np.ndarray, *, spatial_shape: tuple[int, int, int]) -> np.ndarray:
    e_idx = nearest_indices(window.shape[1], spatial_shape[0])
    a_idx = nearest_indices(window.shape[2], spatial_shape[1])
    r_idx = nearest_indices(window.shape[3], spatial_shape[2])
    return window[:, e_idx][:, :, a_idx][:, :, :, r_idx].astype(np.float32, copy=False)


def training_transform_from_args(
    *, training_transform: SharedTrainingTransform | None
) -> SharedTrainingTransform | None:
    return training_transform


def velocity_limit_from_stream(stream: object) -> float:
    metadata = getattr(stream, "metadata")
    velocity_limit_mps = getattr(metadata, "velocity_limit_mps", None)
    if velocity_limit_mps is not None:
        return float(velocity_limit_mps)
    value_range = getattr(metadata, "value_range", None)
    if value_range is not None:
        return max(abs(float(value_range[0])), abs(float(value_range[1])))
    return max(float(np.nanmax(np.abs(getattr(stream, "data")))), 1e-6)


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


def midpoint_timestamps(
    timestamps: np.ndarray | None,
    start: int,
    stop: int,
    target_count: int,
) -> np.ndarray:
    sliced = slice_timestamps(timestamps, start, stop)
    if sliced.size >= 2:
        return np.asarray((sliced[:-1] + sliced[1:]) * 0.5, dtype=np.float32)
    return np.arange(target_count, dtype=np.float32)


def target_timestamps(timestamps: np.ndarray | None, start: int, stop: int, target_count: int) -> np.ndarray:
    sliced = slice_timestamps(timestamps, start, stop)
    if sliced.size >= 2:
        return np.linspace(float(sliced[0]), float(sliced[-1]), target_count + 2, dtype=np.float32)[1:-1]
    return np.arange(target_count, dtype=np.float32)


def nearest_time_indices(source: np.ndarray | None, target: np.ndarray | None, fallback_count: int) -> np.ndarray:
    if target is None or source is None:
        return np.arange(min(len(target) if target is not None else fallback_count, fallback_count), dtype=np.int64)
    source_values = np.asarray(source, dtype=np.float32).reshape(-1)
    target_values = np.asarray(target, dtype=np.float32).reshape(-1)
    return np.asarray([int(np.argmin(np.abs(source_values - value))) for value in target_values], dtype=np.int64)


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


def local_std_3x3(values: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(values[:, None])
    mean = F.avg_pool2d(tensor, kernel_size=3, stride=1, padding=1)
    mean_sq = F.avg_pool2d(tensor * tensor, kernel_size=3, stride=1, padding=1)
    return np.asarray(torch.sqrt(torch.clamp(mean_sq - mean * mean, min=0.0))[:, 0].numpy(), dtype=np.float32)


def nearest_indices(source_size: int, target_size: int) -> np.ndarray:
    if target_size == source_size:
        return np.arange(source_size, dtype=np.int64)
    return np.clip(np.rint(np.linspace(0, source_size - 1, target_size)), 0, source_size - 1).astype(np.int64)


def two_dimensional_frame_count(
    record: RecordingRecord, data_root: str | Path | None, data_path: str = "data/2d_brightness_mode"
) -> int:
    count = record.frame_counts_by_content_type.get(data_path.removeprefix("data/"))
    if count is not None:
        return int(count)
    stream = open_recording(record, root=data_root).load_stream(data_path)
    return int(stream.data.shape[0])


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
) -> set[str]:
    val_fold = int(cast(Any, validation_fold))
    train_exam_ids = [exam_id for exam_id, fold in fold_by_exam_id.items() if int(fold) != val_fold]
    val_exam_ids = [exam_id for exam_id, fold in fold_by_exam_id.items() if int(fold) == val_fold]
    if not train_exam_ids:
        raise ValueError(f"Fold split leaves no training exams for validation fold {val_fold}")
    if not val_exam_ids:
        raise ValueError(f"Fold split leaves no validation exams for validation fold {val_fold}")
    train_exam_ids = limit_items_by_fraction(train_exam_ids, fraction, seed=seed)
    val_exam_ids = limit_items_by_fraction(val_exam_ids, fraction, seed=seed)
    return set(train_exam_ids) | set(val_exam_ids)


def load_exam_folds(path: str | Path) -> dict[str, int]:
    split_path = Path(path).expanduser()
    with split_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        if "exam_id" not in fields:
            raise ValueError(f"Fold split CSV must contain exam_id: {split_path}")
        fold_field = "five_fold_split" if "five_fold_split" in fields else "fold" if "fold" in fields else None
        if fold_field is None:
            raise ValueError(f"Fold split CSV must contain five_fold_split: {split_path}")
        fold_by_exam_id: dict[str, int] = {}
        for row_number, row in enumerate(reader, start=2):
            exam_id = str(row.get("exam_id") or "").strip()
            if not exam_id:
                raise ValueError(f"Fold split CSV has an empty exam_id at row {row_number}: {split_path}")
            try:
                fold = int(str(row.get(fold_field) or "").strip())
            except ValueError as exc:
                raise ValueError(f"Fold split CSV has a non-integer fold at row {row_number}: {split_path}") from exc
            existing = fold_by_exam_id.get(exam_id)
            if existing is not None and existing != fold:
                raise ValueError(f"Fold split CSV has conflicting folds for {exam_id}: {split_path}")
            fold_by_exam_id[exam_id] = fold
    if not fold_by_exam_id:
        raise ValueError(f"Fold split CSV has no rows: {split_path}")
    return fold_by_exam_id


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
    for filename in DEFAULT_FOLD_SPLIT_FILENAMES:
        path = Path(root) / filename
        if path.exists():
            return path
    return None


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
    return cast(SampleT, cast(Any, type(first))(**values))


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
    validate_data_config: Callable[[Mapping[str, object]], None] | None = None,
    discover_kwargs_fn: Callable[[Mapping[str, object]], Mapping[str, object]] | None = None,
    dataset_kwargs_fn: Callable[[Any, Mapping[str, object], str | Path], Mapping[str, object]] | None = None,
    train_dataset_kwargs_fn: Callable[[Mapping[str, object]], Mapping[str, object]] | None = None,
    val_dataset_kwargs_fn: Callable[[Mapping[str, object]], Mapping[str, object]] | None = None,
    train_batch_size_fn: Callable[[Mapping[str, object]], int | None] | None = None,
    val_batch_size_fn: Callable[[Mapping[str, object]], int | None] | None = None,
    extra_fields_fn: (
        Callable[[Mapping[str, object], Mapping[str, object], object, object], Mapping[str, object]] | None
    ) = None,
) -> tuple[DataLoader[SampleT], DataLoader[SampleT]]:
    data_config = config.section("data")
    if validate_data_config is not None:
        validate_data_config(data_config)
    seed = int(cast(Any, data_config.get("seed", 0)))
    seed_everything(seed)
    root = data_root or data_config.get("root_dir")
    root_path = dataset_root(root)
    fold_split_path = resolve_fold_split_path(root_path, data_config)
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
    train_dataset = _maybe_cached_dataset(
        task_name=task_name,
        split="train",
        dataset=train_dataset,
        data_config=data_config,
        root=root_path,
        records=train_records,
        dataset_kwargs=dataset_kwargs,
        split_kwargs=train_kwargs,
    )
    val_dataset = _maybe_cached_dataset(
        task_name=task_name,
        split="val",
        dataset=val_dataset,
        data_config=data_config,
        root=root_path,
        records=val_records,
        dataset_kwargs=dataset_kwargs,
        split_kwargs=val_kwargs,
    )
    return build_train_val_dataloaders(
        task_name=task_name,
        root=root_path,
        records=records,
        train_records=train_records,
        val_records=val_records,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
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
) -> tuple[DataLoader[SampleT], DataLoader[SampleT]]:
    train_loader = DataLoader(
        train_dataset,
        sampler=OneSamplePerCaseSampler(cast(CaseIndexedDataset, train_dataset), seed=seed),
        **batched_dataloader_kwargs(data_config, batch_size=train_batch_size),
    )
    val_loader = DataLoader(
        val_dataset,
        sampler=OneSamplePerCaseSampler(
            cast(CaseIndexedDataset, val_dataset),
            seed=seed,
            shuffle=False,
            randomize=False,
        ),
        **batched_dataloader_kwargs(data_config, batch_size=val_batch_size),
    )
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
        extra_fields=extra_fields,
    )
    return train_loader, val_loader


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
        return int(len(value))  # type: ignore[arg-type]
    except TypeError:
        return 0


def _safe_len_or_na(value: object) -> str:
    try:
        return str(int(len(value)))  # type: ignore[arg-type]
    except TypeError:
        return "n/a"


def _limit_value(value: object) -> str:
    return "none" if value is None else str(value)


def _maybe_cached_dataset(
    *,
    task_name: str,
    split: str,
    dataset: Dataset[SampleT],
    data_config: Mapping[str, object],
    root: str | Path,
    records: Sequence[RecordingRecord],
    dataset_kwargs: Mapping[str, object],
    split_kwargs: Mapping[str, object],
) -> Dataset[SampleT]:
    del records
    cache_dir = _sample_cache_dir(data_config)
    if cache_dir is None:
        return dataset
    cache_context = _sample_cache_context(
        task_name=task_name,
        dataset=dataset,
        data_config=data_config,
        root=root,
        dataset_kwargs=dataset_kwargs,
        split_kwargs=split_kwargs,
    )
    return CachedDataset(
        dataset,
        cache_dir=cache_dir,
        namespace=split,
        item_keys=_dataset_item_cache_keys(dataset, context=cache_context),
        read_only=_config_bool(data_config.get("sample_cache_read_only", False)),
    )


def _sample_cache_dir(data_config: Mapping[str, object]) -> Path | None:
    enabled = data_config.get("sample_cache")
    configured_dir = data_config.get("sample_cache_dir")
    if enabled is not None and not _config_bool(enabled):
        return None
    if configured_dir is None or str(configured_dir).strip() == "":
        return DEFAULT_SAMPLE_CACHE_DIR if _config_bool(enabled) else None
    return Path(str(configured_dir)).expanduser()


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


def _sample_cache_context(
    *,
    task_name: str,
    dataset: object,
    data_config: Mapping[str, object],
    root: str | Path,
    dataset_kwargs: Mapping[str, object],
    split_kwargs: Mapping[str, object],
) -> Mapping[str, object]:
    version = int(cast(Any, data_config.get("sample_cache_version", _SAMPLE_CACHE_VERSION)))
    return {
        "version": version,
        "task": task_name,
        "dataset_class": f"{type(dataset).__module__}.{type(dataset).__qualname__}",
        "root": str(root),
        "data_config": {
            str(key): _cache_json_value(value)
            for key, value in data_config.items()
            if key not in _SAMPLE_CACHE_CONTROL_KEYS and key not in _DATALOADER_ONLY_CONFIG_KEYS
        },
        "dataset_kwargs": _cache_json_value(_cache_relevant_kwargs(dataset_kwargs)),
        "split_kwargs": _cache_json_value(_cache_relevant_kwargs(split_kwargs)),
    }


def _dataset_item_cache_keys(dataset: object, *, context: object | None = None) -> list[str]:
    refs = getattr(dataset, "sample_refs", None)
    try:
        count = len(dataset)  # type: ignore[arg-type]
    except TypeError:
        count = 0
    keys: list[str] = []
    for index in range(int(count)):
        ref = refs[index] if isinstance(refs, Sequence) and index < len(refs) else None
        value = {
            "context": _cache_json_value(context),
            "index": index,
            "sample_id": getattr(ref, "sample_id", None),
            "ref": _cache_json_value(ref),
        }
        keys.append(_hash_cache_payload(value))
    return keys


def _record_cache_value(record: RecordingRecord) -> Mapping[str, object]:
    return {
        "exam_id": record.exam_id,
        "recording_id": record.recording_id,
        "zarr_path": record.zarr_path,
        "array_paths": tuple(record.array_paths),
        "content_types": tuple(record.content_types),
        "frame_counts_by_content_type": dict(record.frame_counts_by_content_type),
        "median_delta_time_by_content_type": dict(record.median_delta_time_by_content_type),
    }


def _cache_json_value(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": True,
            "shape": tuple(int(dim) for dim in value.shape),
            "dtype": str(value.dtype),
            "values": value.tolist(),
        }
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        return {
            "__tensor__": True,
            "shape": tuple(int(dim) for dim in tensor.shape),
            "dtype": str(tensor.dtype),
            "values": tensor.tolist(),
        }
    if isinstance(value, Mapping):
        return {str(key): _cache_json_value(value[key]) for key in sorted(value, key=lambda item: str(item))}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _cache_json_value(getattr(value, field.name))
            for field in fields(value)
            if hasattr(value, field.name)
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_cache_json_value(item) for item in value]
    return repr(value)


def _cache_relevant_kwargs(values: Mapping[str, object]) -> Mapping[str, object]:
    return {str(key): value for key, value in values.items() if str(key) not in _SAMPLE_CACHE_CONTROL_KEYS}


def _hash_cache_payload(payload: object) -> str:
    encoded = json.dumps(_cache_json_value(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _config_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off", "none"}
    return bool(value)


def _torch_load_sample(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


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


def as_3d_shape(value: tuple[int, int, int]) -> tuple[int, int, int]:
    result = tuple(int(item) for item in value)
    if len(result) != 3 or min(result) <= 0:
        raise ValueError(f"spatial_shape must contain three positive ints, got {value}")
    return result


def optional_3d_shape(value: object) -> tuple[int, int, int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"spatial_shape must be a sequence of three ints, got {type(value).__name__}")
    items = tuple(int(cast(Any, item)) for item in value)
    if len(items) != 3:
        raise ValueError(f"spatial_shape must contain three ints, got {value}")
    return as_3d_shape((items[0], items[1], items[2]))


def optional_int(value: object) -> int | None:
    return None if value is None else int(cast(Any, value))
