from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import Dataset

from echoxflow import RecordingRecord, find_recordings, open_recording
from echoxflow.scan import (
    BeamspacePixelGrid,
    CartesianGrid,
    CartesianPixelGrid,
    SectorGeometry,
    build_contour_masks,
    contour_group_layout_for_metadata,
    mesh_frame_indices_for_volume_timestamps,
    rasterize_polygon_pixels,
    resize_pixel_xy,
    sector_geometry_from_mapping,
    sector_to_cartesian,
)
from tasks.segmentation.types import Sample
from tasks.utils.dataset import (
    SharedTrainingTransform,
    SkippedCase,
)
from tasks.utils.dataset import as_spatial_shape as _as_spatial_shape
from tasks.utils.dataset import (
    build_task_dataloaders,
    discover_records_from_candidates,
    optional_int,
    progress_iter,
    resize_stack,
    shared_training_transform_from_config,
    training_transform_from_args,
)


@dataclass(frozen=True)
class SampleRef:
    record: RecordingRecord
    role_id: str
    start: int
    stop: int
    sample_id: str
    content_type: str = ""
    view_code: str = ""


class RawDataset(Dataset[Sample]):
    """2D LV contour segmentation samples."""

    def __init__(
        self,
        *,
        records: list[RecordingRecord],
        clip_length: int,
        clip_stride: int = 1,
        input_spatial_shape: tuple[int, int] = (128, 128),
        data_root: str | Path | None = None,
        max_samples: int | None = None,
        target_mask_channels: int = 2,
        coordinate_space: str = "beamspace",
        cartesian_height: int | None = None,
        include_cartesian_metrics: bool = False,
        content_types: tuple[str, ...] | None = None,
        sampling_mode: str = "sliding",
        target_frame_policy: str = "all",
        annotation_context_frames: int = 0,
        training_transform: SharedTrainingTransform | None = None,
        recording_cache_dir: str | Path | None = None,
        recording_cache_read_only: bool = False,
        recording_cache_include: tuple[str, ...] = (),
        recording_cache_exclude: tuple[str, ...] = (),
        **_: object,
    ) -> None:
        self.records = sorted(records, key=lambda record: (record.exam_id, record.recording_id))
        self.clip_length = int(clip_length)
        self.clip_stride = int(clip_stride)
        self.input_spatial_shape = _as_spatial_shape(input_spatial_shape)
        self.data_root = data_root
        self.target_mask_channels = int(target_mask_channels)
        self.coordinate_space = str(coordinate_space).strip().lower()
        self.cartesian_height = (
            int(cartesian_height) if cartesian_height is not None else int(self.input_spatial_shape[0])
        )
        self.include_cartesian_metrics = bool(include_cartesian_metrics)
        self.content_types = _string_tuple(content_types)
        self.sampling_mode = str(sampling_mode).strip().lower()
        self.target_frame_policy = str(target_frame_policy).strip().lower()
        self.annotation_context_frames = max(0, int(annotation_context_frames))
        self.recording_cache_dir = None if recording_cache_dir is None else Path(recording_cache_dir)
        self.recording_cache_read_only = bool(recording_cache_read_only)
        self.recording_cache_include = tuple(str(path) for path in recording_cache_include) or (
            ("data/2d_brightness_mode*", "data/*_contour") if self.recording_cache_dir is not None else ()
        )
        self.recording_cache_exclude = tuple(str(path) for path in recording_cache_exclude)
        if self.coordinate_space not in {"beamspace", "cartesian"}:
            raise ValueError(
                f"segmentation coordinate_space must be 'beamspace' or 'cartesian', got {coordinate_space!r}"
            )
        if self.sampling_mode not in {"sliding", "full_recording", "annotation"}:
            raise ValueError(
                "segmentation sampling_mode must be 'sliding', 'full_recording', or "
                f"'annotation', got {sampling_mode!r}"
            )
        if self.target_frame_policy not in {"all", "annotated"}:
            raise ValueError(
                f"segmentation target_frame_policy must be 'all' or 'annotated', got {target_frame_policy!r}"
            )
        self.training_transform = training_transform_from_args(training_transform=training_transform)
        self.skipped_cases: list[SkippedCase] = []
        self.empty_mask_count = 0
        self.sample_refs = self._build_sample_refs(max_samples=max_samples)
        self.sample_indices_by_case = _sample_indices_by_bmode_panel(self.sample_refs)

    def __len__(self) -> int:
        return len(self.sample_refs)

    def __getitem__(self, index: int) -> Sample:
        return self.sample_from_ref(self.sample_refs[index])

    def sample_from_ref(self, ref: SampleRef) -> Sample:
        store = open_recording(
            ref.record,
            root=self.data_root,
            cache_dir=self.recording_cache_dir,
            cache_read_only=self.recording_cache_read_only,
            cache_include=self.recording_cache_include,
            cache_exclude=self.recording_cache_exclude,
        )
        obj = store.load_object()
        panel = next(panel for panel in obj.panels if panel.role_id == ref.role_id)
        source_store = store.open_reference(panel.bmode.recording)
        frames, bmode_timestamps, bmode_metadata = _load_bmode_clip(
            source_store,
            data_path=panel.bmode.data_path,
            timestamps_path=panel.bmode.timestamps_path,
            start=ref.start,
            stop=ref.stop,
        )
        if frames.ndim != 3:
            raise ValueError(f"Segmentation B-mode frames must be [time,height,width], got {frames.shape}")
        source_image_shape = (int(frames.shape[-2]), int(frames.shape[-1]))
        frames, geometry, cartesian_grid, valid_region_mask = self._prepare_frames(frames, panel=panel)
        masks, mask_valid = _contour_masks(
            store,
            panel=panel,
            start=ref.start,
            stop=ref.stop,
            target_timestamps=bmode_timestamps,
            bmode_metadata=bmode_metadata,
            source_store=source_store,
            geometry=geometry,
            content_type=ref.content_type,
            output_shape=(int(frames.shape[-2]), int(frames.shape[-1])),
            source_image_shape=source_image_shape,
            output_grid=cartesian_grid,
            target_mask_channels=self.target_mask_channels,
            target_frame_policy=self.target_frame_policy,
        )
        metric_masks: np.ndarray | None = None
        metric_valid: np.ndarray | None = None
        metric_sample_grid: np.ndarray | None = None
        if self.include_cartesian_metrics:
            metric_masks, metric_valid, _metric_grid, metric_sample_grid = self._cartesian_metric_targets(
                store=store,
                panel=panel,
                start=ref.start,
                stop=ref.stop,
                target_timestamps=bmode_timestamps,
                bmode_metadata=bmode_metadata,
                source_store=source_store,
                geometry=geometry,
                content_type=ref.content_type,
                source_image_shape=source_image_shape,
                target_mask_channels=self.target_mask_channels,
                target_frame_policy=self.target_frame_policy,
            )
        if not bool(mask_valid.any()):
            self.empty_mask_count += 1
        return self._sample(
            ref=ref,
            frames=frames,
            masks=masks,
            mask_valid=mask_valid,
            valid_region_mask=valid_region_mask,
            cartesian_metric_target_masks=metric_masks,
            cartesian_metric_valid_mask=metric_valid,
            cartesian_metric_sample_grid=metric_sample_grid,
            sample_id=ref.sample_id,
        )

    def _prepare_frames(
        self,
        frames: np.ndarray,
        *,
        panel: Any,
    ) -> tuple[np.ndarray, SectorGeometry | None, CartesianGrid | None, np.ndarray | None]:
        source_shape = (int(frames.shape[-2]), int(frames.shape[-1]))
        geometry = _panel_geometry(panel, source_shape=source_shape)
        if self.coordinate_space == "cartesian":
            if geometry is None:
                raise ValueError(
                    f"Segmentation cartesian coordinate_space requires sector geometry for {panel.role_id!r}"
                )
            grid = CartesianGrid.from_sector_height(geometry, self.cartesian_height)
            converted = tuple(
                sector_to_cartesian(frame, geometry, grid=grid, interpolation="linear") for frame in frames
            )
            frames = np.stack([image.data for image in converted], axis=0).astype(np.float32, copy=False)
            valid = np.asarray(converted[0].mask, dtype=np.float32)
            if tuple(frames.shape[-2:]) != self.input_spatial_shape:
                frames = resize_stack(frames, self.input_spatial_shape)
                valid = resize_stack(valid[None], self.input_spatial_shape)[0]
            frames = _minmax_normalize_frames_per_frame(frames)
            return frames, geometry, grid, valid
        frames = resize_stack(frames, self.input_spatial_shape)
        frames = _minmax_normalize_frames_per_frame(frames)
        return frames, geometry, None, None

    def _cartesian_metric_targets(
        self,
        *,
        store: Any,
        panel: Any,
        start: int,
        stop: int,
        target_timestamps: np.ndarray,
        bmode_metadata: Any,
        source_store: Any,
        geometry: SectorGeometry | None,
        content_type: str,
        source_image_shape: tuple[int, int],
        target_mask_channels: int,
        target_frame_policy: str,
    ) -> tuple[np.ndarray | None, np.ndarray | None, CartesianGrid | None, np.ndarray | None]:
        if geometry is None:
            return None, None, None, None
        grid = CartesianGrid.from_sector_height(geometry, self.cartesian_height)
        metric_shape = grid.shape
        metric_masks, metric_valid = _contour_masks(
            store,
            panel=panel,
            start=start,
            stop=stop,
            target_timestamps=target_timestamps,
            bmode_metadata=bmode_metadata,
            source_store=source_store,
            geometry=geometry,
            content_type=content_type,
            output_shape=metric_shape,
            source_image_shape=source_image_shape,
            output_grid=grid,
            target_mask_channels=target_mask_channels,
            target_frame_policy=target_frame_policy,
        )
        sector_mask = _cartesian_sector_mask(geometry, grid, shape=metric_shape).astype(np.float32, copy=False)
        metric_valid_mask = metric_valid * sector_mask[None, None, :, :]
        if self.coordinate_space == "beamspace":
            sample_grid = _cartesian_grid_sample_grid(
                geometry,
                grid,
                source_shape=self.input_spatial_shape,
            )
        else:
            sample_grid = _cartesian_image_sample_grid(grid)
        return metric_masks, metric_valid_mask, grid, sample_grid

    def _sample(
        self,
        *,
        ref: SampleRef,
        frames: np.ndarray,
        masks: np.ndarray,
        mask_valid: np.ndarray,
        valid_region_mask: np.ndarray | None,
        cartesian_metric_target_masks: np.ndarray | None,
        cartesian_metric_valid_mask: np.ndarray | None,
        cartesian_metric_sample_grid: np.ndarray | None,
        sample_id: str,
        clip_start: int | None = None,
        clip_stop: int | None = None,
    ) -> Sample:
        if valid_region_mask is None:
            valid_mask = np.ones((1, int(frames.shape[0]), 1, 1), dtype=np.float32)
            target_valid = mask_valid
        else:
            region = np.asarray(valid_region_mask, dtype=np.float32)
            valid_mask = np.broadcast_to(region[None, None, :, :], (1, int(frames.shape[0]), *region.shape)).copy()
            target_valid = mask_valid * region[None, None, :, :]
        return Sample(
            frames=torch.from_numpy(frames[None, None, ...].astype(np.float32, copy=False)),
            target_masks=torch.from_numpy(masks[None, ...].astype(np.float32, copy=False)),
            valid_mask=torch.from_numpy(valid_mask[None, ...]),
            target_mask_valid=torch.from_numpy(target_valid[None, ...].astype(np.float32, copy=False)),
            conditioning=None,
            sample_id=sample_id,
            record=ref.record,
            data_root=self.data_root,
            clip_start=ref.start if clip_start is None else int(clip_start),
            clip_stop=ref.stop if clip_stop is None else int(clip_stop),
            role_id=ref.role_id,
            coordinate_space=self.coordinate_space,
            cartesian_metric_target_masks=(
                None
                if cartesian_metric_target_masks is None
                else torch.from_numpy(cartesian_metric_target_masks[None, ...].astype(np.float32, copy=False))
            ),
            cartesian_metric_valid_mask=(
                None
                if cartesian_metric_valid_mask is None
                else torch.from_numpy(cartesian_metric_valid_mask[None, ...].astype(np.float32, copy=False))
            ),
            cartesian_metric_sample_grid=(
                None
                if cartesian_metric_sample_grid is None
                else torch.from_numpy(cartesian_metric_sample_grid[None, ...].astype(np.float32, copy=False))
            ),
        )

    def _build_sample_refs(self, *, max_samples: int | None) -> list[SampleRef]:
        refs: list[SampleRef] = []
        for record in progress_iter(
            self.records,
            description="Scanning segmentation samples",
            total=len(self.records),
            unit="case",
        ):
            store = open_recording(
                record,
                root=self.data_root,
                cache_dir=self.recording_cache_dir,
                cache_read_only=self.recording_cache_read_only,
                cache_include=self.recording_cache_include,
                cache_exclude=self.recording_cache_exclude,
            )
            obj = store.load_object()
            content_type = _primary_segmentation_content_type(record, wanted=self.content_types)
            if not content_type:
                continue
            for panel in obj.panels:
                if not _role_matches_content_type(str(panel.role_id), content_type):
                    continue
                if not _has_contour(store, panel):
                    continue
                contour_path, _contour_timestamps_path = _contour_paths(panel)
                contour_shape = _contour_shape(store, contour_path)
                if contour_shape is not None and int(contour_shape[0]) == 0:
                    self.skipped_cases.append(
                        SkippedCase(
                            case_id=f"{record.exam_id}/{record.recording_id}/{panel.role_id}",
                            reason=f"empty contour array {contour_path!r} with shape {contour_shape}",
                        )
                    )
                    continue
                frame_count = _panel_frame_count(store, panel)
                if frame_count is None:
                    self.skipped_cases.append(
                        SkippedCase(
                            case_id=f"{record.exam_id}/{record.recording_id}/{panel.role_id}",
                            reason=f"missing linked B-mode array {panel.bmode.data_path!r}",
                        )
                    )
                    continue
                if frame_count <= 0:
                    self.skipped_cases.append(
                        SkippedCase(
                            case_id=f"{record.exam_id}/{record.recording_id}/{panel.role_id}",
                            reason=f"empty linked B-mode array {panel.bmode.data_path!r}",
                        )
                    )
                    continue
                view_code = str(panel.view_code or panel.role_id)
                if self.sampling_mode == "annotation":
                    bounds = _annotation_window_bounds(
                        store,
                        panel=panel,
                        frame_count=frame_count,
                        context_frames=self.annotation_context_frames,
                    )
                    if bounds is None:
                        self.skipped_cases.append(
                            SkippedCase(
                                case_id=f"{record.exam_id}/{record.recording_id}/{panel.role_id}",
                                reason="no contour annotations align to linked B-mode frames",
                            )
                        )
                        continue
                    start, stop = bounds
                    refs.append(
                        SampleRef(
                            record=record,
                            role_id=panel.role_id,
                            start=start,
                            stop=stop,
                            content_type=content_type,
                            view_code=view_code,
                            sample_id=(
                                f"{record.exam_id}/{record.recording_id}/{panel.role_id}:"
                                f"frames_annotation_{start}_{stop - 1}"
                            ),
                        )
                    )
                    if max_samples is not None and len(refs) >= int(max_samples):
                        return refs
                    continue
                if frame_count < self.clip_length:
                    if self.sampling_mode == "full_recording":
                        refs.append(
                            SampleRef(
                                record=record,
                                role_id=panel.role_id,
                                start=0,
                                stop=frame_count,
                                content_type=content_type,
                                view_code=view_code,
                                sample_id=f"{record.exam_id}/{record.recording_id}/{panel.role_id}:frames_full",
                            )
                        )
                        if max_samples is not None and len(refs) >= int(max_samples):
                            return refs
                        continue
                    self.skipped_cases.append(
                        SkippedCase(
                            case_id=f"{record.exam_id}/{record.recording_id}/{panel.role_id}",
                            reason=f"only {frame_count} linked B-mode frames, need at least {self.clip_length}",
                        )
                    )
                    continue
                if self.sampling_mode == "full_recording":
                    refs.append(
                        SampleRef(
                            record=record,
                            role_id=panel.role_id,
                            start=0,
                            stop=frame_count,
                            content_type=content_type,
                            view_code=view_code,
                            sample_id=f"{record.exam_id}/{record.recording_id}/{panel.role_id}:frames_full",
                        )
                    )
                    if max_samples is not None and len(refs) >= int(max_samples):
                        return refs
                    continue
                for start in range(0, frame_count - self.clip_length + 1, self.clip_stride):
                    refs.append(
                        SampleRef(
                            record=record,
                            role_id=panel.role_id,
                            start=start,
                            stop=start + self.clip_length,
                            content_type=content_type,
                            view_code=view_code,
                            sample_id=(
                                f"{record.exam_id}/{record.recording_id}/{panel.role_id}:"
                                f"frames_{start}_{start + self.clip_length - 1}"
                            ),
                        )
                    )
                    if max_samples is not None and len(refs) >= int(max_samples):
                        return refs
        return refs


def discover_records(
    *,
    root_dir: str | Path,
    max_cases: int | None = None,
    exam_ids: Iterable[str] | None = None,
    sample_fraction: object | None = None,
    seed: int | None = None,
    content_types: tuple[str, ...] | None = None,
) -> list[RecordingRecord]:
    wanted = {str(value).strip() for value in (content_types or ()) if str(value).strip()}
    found = find_recordings(
        root=root_dir,
        predicate=lambda record: _segmentation_discovery_predicate(record, wanted=tuple(wanted)),
    )
    return discover_records_from_candidates(
        found,
        exam_ids=exam_ids,
        max_cases=max_cases,
        sample_fraction=sample_fraction,
        seed=seed,
        description="Scanning segmentation recordings",
    )


def _segmentation_discovery_predicate(record: RecordingRecord, *, wanted: tuple[str, ...]) -> bool:
    if not any("strain" in content_type for content_type in record.content_types):
        return False
    if wanted and not bool(set(wanted) & set(record.content_types)):
        return False
    return any(path.startswith("data/") and path.endswith("_contour") for path in record.array_paths)


def discover_case_dirs(*, root_dir: str | Path, max_cases: int | None = None) -> list[Path]:
    return [record.path(root_dir) for record in discover_records(root_dir=root_dir, max_cases=max_cases)]


def _discover_kwargs(data_config: Mapping[str, object]) -> dict[str, object]:
    return {"content_types": _string_tuple(data_config.get("content_types"))}


def _dataset_kwargs(config: Any, data_config: Mapping[str, object], root: str | Path) -> dict[str, object]:
    model_out_channels = int(config.model.get("out_channels", 1))
    return {
        "clip_length": int(cast(Any, data_config.get("clip_length", data_config.get("output_frames", 32)))),
        "clip_stride": int(cast(Any, data_config.get("clip_stride", 1))),
        "input_spatial_shape": _as_spatial_shape(
            cast(Any, data_config.get("input_spatial_shape", data_config.get("spatial_shape", (128, 128))))
        ),
        "data_root": root,
        "target_mask_channels": int(cast(Any, data_config.get("target_mask_channels", max(1, model_out_channels - 1)))),
        "coordinate_space": str(data_config.get("coordinate_space", "beamspace")),
        "cartesian_height": optional_int(data_config.get("cartesian_height")),
        "content_types": _string_tuple(data_config.get("content_types")),
    }


def _train_dataset_kwargs(data_config: Mapping[str, object]) -> dict[str, object]:
    return {"training_transform": shared_training_transform_from_config(data_config)}


def _val_dataset_kwargs(data_config: Mapping[str, object]) -> dict[str, object]:
    return {
        "include_cartesian_metrics": True,
        "sampling_mode": str(data_config.get("val_sampling_mode", data_config.get("sampling_mode", "full_recording"))),
        "target_frame_policy": "annotated",
        "annotation_context_frames": int(cast(Any, data_config.get("val_annotation_context_frames", 8))),
    }


def _extra_fields(
    data_config: Mapping[str, object],
    dataset_kwargs: Mapping[str, object],
    train_dataset: object,
    val_dataset: object,
) -> dict[str, object]:
    del data_config
    return {
        "coordinate_space": dataset_kwargs["coordinate_space"],
        "cartesian_height": dataset_kwargs["cartesian_height"],
        "clip_length": dataset_kwargs["clip_length"],
        "train_target_frame_policy": "full_clip_qrs_aware_contour",
        "val_sampling_mode": cast(Any, val_dataset).sampling_mode,
        "val_target_frame_policy": cast(Any, val_dataset).target_frame_policy,
        "val_annotation_context_frames": cast(Any, val_dataset).annotation_context_frames,
        "input_spatial_shape": dataset_kwargs["input_spatial_shape"],
        "train_skipped_panels": len(cast(Any, train_dataset).skipped_cases),
        "val_skipped_panels": len(cast(Any, val_dataset).skipped_cases),
        "train_empty_masks": cast(Any, train_dataset).empty_mask_count,
        "val_empty_masks": cast(Any, val_dataset).empty_mask_count,
    }


build_dataloaders = partial(
    build_task_dataloaders,
    task_name="segmentation",
    dataset_cls=RawDataset,
    discover_records=discover_records,
    empty_message="Segmentation dataset is empty",
    validate_data_config=lambda data_config: _reject_obsolete_segmentation_data_config(
        cast(dict[str, Any], data_config)
    ),
    discover_kwargs_fn=_discover_kwargs,
    dataset_kwargs_fn=_dataset_kwargs,
    train_dataset_kwargs_fn=_train_dataset_kwargs,
    val_dataset_kwargs_fn=_val_dataset_kwargs,
    extra_fields_fn=_extra_fields,
)


def _sample_indices_by_bmode_panel(refs: list[SampleRef]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for index, ref in enumerate(refs):
        grouped.setdefault(f"{ref.record.exam_id}/{ref.record.recording_id}/{ref.role_id}", []).append(index)
    return grouped


def _annotation_window_bounds(
    store: Any,
    *,
    panel: Any,
    frame_count: int,
    context_frames: int,
) -> tuple[int, int] | None:
    contour_path, contour_timestamps_path = _contour_paths(panel)
    if contour_path is None or contour_path not in store.group:
        return None
    contour_count = int(getattr(store.group[contour_path], "shape", (0,))[0])
    if contour_count <= 0 or int(frame_count) <= 0:
        return None
    contour_timestamps = _load_timestamps(store, contour_timestamps_path, start=0, stop=contour_count)
    if contour_timestamps_path and contour_timestamps_path in store.group:
        try:
            source_store = store.open_reference(panel.bmode.recording)
        except (FileNotFoundError, KeyError, ValueError):
            return None
        target_timestamps = _load_timestamps(
            source_store,
            panel.bmode.timestamps_path,
            start=0,
            stop=int(frame_count),
        )
        matches = _annotated_target_frame_matches(
            contour_timestamps=contour_timestamps,
            target_timestamps=target_timestamps,
            contour_count=contour_count,
            target_count=int(frame_count),
        )
        annotated_frames = [int(target_index) for _contour_index, target_index in matches]
    else:
        annotated_frames = list(range(min(int(frame_count), contour_count)))
    if not annotated_frames:
        return None
    first = min(annotated_frames)
    last = max(annotated_frames)
    context = max(0, int(context_frames))
    start = max(0, first - context)
    stop = min(int(frame_count), last + context + 1)
    return (start, stop) if stop > start else None


def _has_contour(store: Any, panel: Any) -> bool:
    return any(
        _is_contour_annotation(annotation) and annotation.value.path in store.group for annotation in panel.annotations
    )


def _panel_frame_count(store: Any, panel: Any) -> int | None:
    try:
        source_store = store.open_reference(panel.bmode.recording)
    except (FileNotFoundError, KeyError, ValueError):
        return None
    data_path = str(panel.bmode.data_path).strip("/")
    if data_path not in source_store.group:
        return None
    shape = tuple(getattr(source_store.group[data_path], "shape", ()))
    if not shape:
        return None
    return int(shape[0])


def _load_bmode_clip(
    store: Any,
    *,
    data_path: str,
    timestamps_path: str | None,
    start: int,
    stop: int,
) -> tuple[np.ndarray, np.ndarray, Any]:
    stream = store.load_stream_slice(data_path, start, stop)
    frames = np.asarray(stream.data, dtype=np.float32)
    frame_count = int(frames.shape[0]) if frames.ndim >= 1 else int(stop) - int(start)
    metadata = getattr(getattr(stream, "metadata", None), "raw", None)
    explicit = _load_optional_timestamp_slice(
        store,
        timestamps_path,
        start=start,
        stop=stop,
        expected_count=frame_count,
    )
    if explicit is not None:
        return frames, explicit, metadata
    stream_timestamps = _valid_timestamp_array(stream.timestamps, expected_count=frame_count)
    if stream_timestamps is not None:
        return frames, stream_timestamps, metadata
    return frames, np.arange(int(start), int(start) + frame_count, dtype=np.float32), metadata


def _load_optional_timestamp_slice(
    store: Any,
    path: str | None,
    *,
    start: int,
    stop: int,
    expected_count: int,
) -> np.ndarray | None:
    if not path or path not in store.group:
        return None
    return _valid_timestamp_array(store.load_array_slice(path, start, stop), expected_count=expected_count)


def _valid_timestamp_array(values: object, *, expected_count: int) -> np.ndarray | None:
    if values is None:
        return None
    timestamps = np.asarray(values, dtype=np.float32).reshape(-1)
    return timestamps if timestamps.size == int(expected_count) else None


def _contour_masks(
    store: Any,
    *,
    panel: Any,
    start: int,
    stop: int,
    target_timestamps: np.ndarray,
    bmode_metadata: Any,
    source_store: Any,
    geometry: SectorGeometry | None,
    content_type: str,
    output_shape: tuple[int, int],
    source_image_shape: tuple[int, int],
    target_mask_channels: int,
    target_frame_policy: str = "all",
    output_grid: CartesianGrid | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    contour_path, contour_timestamps_path = _contour_paths(panel)
    frame_count = int(stop) - int(start)
    empty = _empty_targets(
        frame_count=frame_count, output_shape=output_shape, target_mask_channels=target_mask_channels
    )
    if contour_path is None or contour_path not in store.group:
        return empty
    contours = np.asarray(store.load_array(contour_path), dtype=np.float32)
    if contours.ndim < 1 or int(contours.shape[0]) == 0:
        return empty
    has_contour_timestamps = bool(contour_timestamps_path and contour_timestamps_path in store.group)
    contour_timestamps = _load_timestamps(store, contour_timestamps_path, start=0, stop=int(contours.shape[0]))
    if target_frame_policy == "annotated":
        points, base_valid = _annotated_contour_points_for_target_frames(
            contours=contours,
            contour_timestamps=contour_timestamps,
            target_timestamps=target_timestamps,
            frame_count=frame_count,
            has_contour_timestamps=has_contour_timestamps,
        )
        return _target_masks_from_contour_points(
            contour_points=points,
            geometry=geometry,
            content_type=content_type,
            role_id=str(panel.role_id),
            image_shape=output_shape,
            source_image_shape=source_image_shape,
            output_grid=output_grid,
            target_mask_channels=target_mask_channels,
            base_valid_mask=base_valid,
        )
    indices = _contour_frame_indices(
        store,
        source_store=source_store,
        panel=panel,
        contour_timestamps=contour_timestamps,
        target_timestamps=target_timestamps,
        bmode_metadata=bmode_metadata,
        contour_frame_count=int(contours.shape[0]),
        target_count=frame_count,
    )
    points = np.asarray(contours[np.clip(indices, 0, int(contours.shape[0]) - 1)], dtype=np.float32)
    return _target_masks_from_contour_points(
        contour_points=points,
        geometry=geometry,
        content_type=content_type,
        role_id=str(panel.role_id),
        image_shape=output_shape,
        source_image_shape=source_image_shape,
        output_grid=output_grid,
        target_mask_channels=target_mask_channels,
        base_valid_mask=np.ones((frame_count,), dtype=np.float32),
    )


def _annotated_contour_points_for_target_frames(
    *,
    contours: np.ndarray,
    contour_timestamps: np.ndarray,
    target_timestamps: np.ndarray,
    frame_count: int,
    has_contour_timestamps: bool,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.full((int(frame_count), *contours.shape[1:]), np.nan, dtype=np.float32)
    base_valid = np.zeros((int(frame_count),), dtype=np.float32)
    if int(frame_count) <= 0:
        return points, base_valid
    if not has_contour_timestamps:
        count = min(int(frame_count), int(contours.shape[0]))
        points[:count] = np.asarray(contours[:count], dtype=np.float32)
        base_valid[:count] = 1.0
        return points, base_valid
    matches = _annotated_target_frame_matches(
        contour_timestamps=contour_timestamps,
        target_timestamps=target_timestamps,
        contour_count=int(contours.shape[0]),
        target_count=int(frame_count),
    )
    for contour_index, target_index in matches:
        points[int(target_index)] = np.asarray(contours[int(contour_index)], dtype=np.float32)
        base_valid[int(target_index)] = 1.0
    return points, base_valid


def _annotated_target_frame_matches(
    *,
    contour_timestamps: np.ndarray,
    target_timestamps: np.ndarray,
    contour_count: int,
    target_count: int,
) -> tuple[tuple[int, int], ...]:
    if int(contour_count) <= 0 or int(target_count) <= 0:
        return ()
    source = np.asarray(contour_timestamps, dtype=np.float64).reshape(-1)
    if source.size != int(contour_count):
        source = np.arange(int(contour_count), dtype=np.float64)
    target = np.asarray(target_timestamps, dtype=np.float64).reshape(-1)
    if target.size != int(target_count):
        target = np.arange(int(target_count), dtype=np.float64)
    finite_target = np.isfinite(target)
    if not bool(np.any(finite_target)):
        return ()
    tolerance = _timestamp_match_tolerance(target[finite_target])
    best_by_target: dict[int, tuple[int, float]] = {}
    for contour_index, timestamp in enumerate(source):
        if not np.isfinite(timestamp):
            continue
        distances = np.abs(target - float(timestamp))
        distances = np.where(finite_target, distances, np.inf)
        target_index = int(np.argmin(distances))
        distance = float(distances[target_index])
        if distance > tolerance:
            continue
        previous = best_by_target.get(target_index)
        if previous is None or distance < previous[1]:
            best_by_target[target_index] = (int(contour_index), distance)
    return tuple(
        (contour_index, target_index) for target_index, (contour_index, _distance) in sorted(best_by_target.items())
    )


def _timestamp_match_tolerance(timestamps: np.ndarray) -> float:
    values = np.sort(np.asarray(timestamps, dtype=np.float64).reshape(-1))
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return float("inf")
    steps = np.diff(values)
    positive = steps[steps > 0.0]
    if positive.size == 0:
        return float("inf")
    return max(float(np.median(positive)) * 0.5 + 1e-6, 1e-6)


def _target_masks_from_contour_points(
    *,
    contour_points: np.ndarray,
    geometry: SectorGeometry | None,
    content_type: str,
    role_id: str,
    image_shape: tuple[int, int],
    source_image_shape: tuple[int, int],
    target_mask_channels: int,
    base_valid_mask: np.ndarray,
    output_grid: CartesianGrid | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(contour_points, dtype=np.float32)
    frame_count = int(points.shape[0]) if points.ndim >= 1 else 0
    target_masks, target_valid = _empty_targets(
        frame_count=frame_count,
        output_shape=image_shape,
        target_mask_channels=target_mask_channels,
    )
    if points.ndim != 3 or points.shape[-1] != 2 or frame_count == 0:
        return target_masks, target_valid
    height, width = int(image_shape[0]), int(image_shape[1])
    pixel_points = _contour_points_to_pixel_array(
        points=points,
        geometry=geometry,
        image_shape=(height, width),
        source_image_shape=source_image_shape,
        output_grid=output_grid,
    )
    group_layout = contour_group_layout_for_metadata(content_type=content_type, point_count=int(points.shape[1]))
    channel_indices = _target_mask_channel_indices(
        content_type=content_type,
        role_id=role_id,
        channel_count=int(target_mask_channels),
    )
    base_valid = np.asarray(base_valid_mask, dtype=np.float32).reshape(-1)
    for frame_index in range(frame_count):
        if frame_index < base_valid.size and float(base_valid[frame_index]) <= 0.0:
            continue
        if int(pixel_points.shape[1]) % int(group_layout.group_size) != 0:
            mask = rasterize_polygon_pixels(pixel_points[frame_index], image_shape=(height, width)).astype(np.float32)
            if not bool(mask.any()):
                continue
            target_masks[0, frame_index] = mask
            target_valid[0, frame_index, 0, 0] = 1.0
            continue
        result = build_contour_masks(
            pixel_points[frame_index],
            image_shape=(height, width),
            group_layout=group_layout,
        )
        if not bool(result.endo_mask.any()) and not bool(result.myo_mask.any()):
            continue
        if target_mask_channels == 1:
            target_masks[0, frame_index] = np.asarray(result.endo_mask | result.myo_mask, dtype=np.float32)
            target_valid[0, frame_index, 0, 0] = 1.0
            continue
        endo_channel, myo_channel = channel_indices
        target_masks[endo_channel, frame_index] = result.endo_mask.astype(np.float32)
        target_masks[myo_channel, frame_index] = result.myo_mask.astype(np.float32)
        target_valid[endo_channel, frame_index, 0, 0] = 1.0
        target_valid[myo_channel, frame_index, 0, 0] = 1.0
    return target_masks, target_valid


def _empty_targets(
    *,
    frame_count: int,
    output_shape: tuple[int, int],
    target_mask_channels: int,
) -> tuple[np.ndarray, np.ndarray]:
    channels = max(1, int(target_mask_channels))
    height, width = int(output_shape[0]), int(output_shape[1])
    return (
        np.zeros((channels, int(frame_count), height, width), dtype=np.float32),
        np.zeros((channels, int(frame_count), 1, 1), dtype=np.float32),
    )


def _contour_points_to_pixel_array(
    *,
    points: np.ndarray,
    geometry: SectorGeometry | None,
    image_shape: tuple[int, int],
    source_image_shape: tuple[int, int],
    output_grid: CartesianGrid | None = None,
) -> np.ndarray:
    height, width = int(image_shape[0]), int(image_shape[1])
    pts = np.asarray(points, dtype=np.float32)
    out = np.full_like(pts, np.nan, dtype=np.float32)
    if pts.ndim != 3 or pts.shape[-1] != 2:
        return out
    flat = pts.reshape(-1, 2)
    finite = np.isfinite(flat).all(axis=1)
    if not bool(np.any(finite)):
        return out
    finite_points = flat[finite]
    if output_grid is not None:
        mapped = CartesianPixelGrid(output_grid, shape=(height, width)).physical_to_pixel_xy(finite_points)
        out.reshape(-1, 2)[finite] = mapped
        return out
    if geometry is None:
        return out
    source_shape = geometry.grid_shape or (int(source_image_shape[0]), int(source_image_shape[1]))
    mapped = BeamspacePixelGrid(geometry=geometry, shape=source_shape).physical_to_pixel_xy(finite_points)
    if source_shape != (height, width):
        mapped = resize_pixel_xy(mapped, source_shape=source_shape, target_shape=(height, width))
    out.reshape(-1, 2)[finite] = mapped
    return out


def _target_mask_channel_indices(*, content_type: str, role_id: str, channel_count: int) -> tuple[int, int]:
    content = str(content_type).strip().lower()
    role = str(role_id).strip().lower()
    indices = {
        "2d_left_ventricular_strain": (0, 1),
        "2d_left_atrial_strain": (2, 3),
        "2d_right_ventricular_strain": (4, 5 if int(channel_count) >= 6 else 1),
    }
    pair = indices.get(content)
    if pair is None:
        if role in {"rv", "afirv"}:
            pair = (4, 5 if int(channel_count) >= 6 else 1)
        elif role in {"la", "afila"}:
            pair = (2, 3)
        else:
            pair = (0, 1)
    if max(pair) >= int(channel_count):
        return 0, min(1, max(0, int(channel_count) - 1))
    return pair


def _panel_geometry(panel: Any, *, source_shape: tuple[int, int] | None = None) -> SectorGeometry | None:
    raw = panel.geometry
    if not isinstance(raw, dict):
        return None
    try:
        return sector_geometry_from_mapping(
            raw,
            grid_shape=None if source_shape is None else (int(source_shape[0]), int(source_shape[1])),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _contour_paths(panel: Any) -> tuple[str | None, str | None]:
    for annotation in panel.annotations:
        if _is_contour_annotation(annotation):
            return annotation.value.path, None if annotation.time is None else annotation.time.path
    return None, None


def _contour_shape(store: Any, contour_path: str | None) -> tuple[int, ...] | None:
    if contour_path is None or contour_path not in store.group:
        return None
    shape = tuple(int(dim) for dim in getattr(store.group[contour_path], "shape", ()))
    if not shape:
        return None
    return shape


def _load_timestamps(store: Any, path: str | None, *, start: int, stop: int) -> np.ndarray:
    if not path or path not in store.group:
        return np.arange(int(start), int(stop), dtype=np.float32)
    values = np.asarray(store.load_array_slice(path, start, stop), dtype=np.float32)
    if values.size != int(stop) - int(start):
        return np.arange(int(start), int(stop), dtype=np.float32)
    return values


def _contour_frame_indices(
    store: Any,
    *,
    source_store: Any,
    panel: Any,
    contour_timestamps: np.ndarray,
    target_timestamps: np.ndarray,
    bmode_metadata: Any,
    contour_frame_count: int,
    target_count: int,
) -> np.ndarray:
    qrs = _segmentation_qrs_trigger_times(store, source_store, panel, bmode_metadata=bmode_metadata)
    if qrs.size >= 2:
        aligned = mesh_frame_indices_for_volume_timestamps(
            contour_timestamps,
            target_timestamps,
            _qrs_alignment_metadata(bmode_metadata, qrs),
            mesh_frame_count=contour_frame_count,
            target_count=target_count,
        )
        if len(aligned) == int(target_count):
            return np.asarray(aligned, dtype=np.int64)
    return _nearest_indices(contour_timestamps, target_timestamps)


def _nearest_indices(source_timestamps: np.ndarray, target_timestamps: np.ndarray) -> np.ndarray:
    source = np.asarray(source_timestamps, dtype=np.float64).reshape(-1)
    target = np.asarray(target_timestamps, dtype=np.float64).reshape(-1)
    if source.size == 0:
        return np.zeros(target.shape, dtype=np.int64)
    indices = np.searchsorted(source, target, side="left")
    indices = np.clip(indices, 0, source.size - 1)
    previous = np.clip(indices - 1, 0, source.size - 1)
    use_previous = np.abs(source[previous] - target) <= np.abs(source[indices] - target)
    return np.where(use_previous, previous, indices).astype(np.int64)


def _minmax_normalize_frames_per_frame(frames: np.ndarray) -> np.ndarray:
    finite = np.nan_to_num(np.asarray(frames, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if finite.size and float(np.nanmax(finite)) > 1.5:
        finite = finite / 255.0
    low = np.min(finite, axis=(1, 2), keepdims=True)
    high = np.max(finite, axis=(1, 2), keepdims=True)
    value_range = high - low
    constant = ~np.isfinite(value_range) | (value_range < 1e-6)
    if bool(np.any(constant)):
        constant_frames = np.flatnonzero(constant.reshape(-1))
        preview = ", ".join(str(int(index)) for index in constant_frames[:8])
        suffix = "" if constant_frames.size <= 8 else ", ..."
        warnings.warn(
            "segmentation min-max normalization received "
            f"{int(constant_frames.size)}/{int(value_range.shape[0])} constant frames "
            f"with input shape {tuple(int(value) for value in finite.shape)} "
            f"(frame indices: {preview}{suffix}); returning zeros for those frames",
            RuntimeWarning,
            stacklevel=2,
        )
    value_range = np.where(~constant, value_range, 1.0)
    return cast(np.ndarray, np.clip((finite - low) / value_range, 0.0, 1.0).astype(np.float32))


def _primary_segmentation_content_type(record: RecordingRecord, *, wanted: tuple[str, ...] = ()) -> str:
    wanted_set = {value for value in wanted if value}
    for value in record.content_types:
        content_type = str(value).strip()
        if "strain" in content_type and (not wanted_set or content_type in wanted_set):
            return content_type
    return ""


def _role_matches_content_type(role_id: str, content_type: str) -> bool:
    role = str(role_id).strip().lower()
    content = str(content_type).strip().lower()
    if content == "2d_left_ventricular_strain":
        return role not in {"la", "afirla", "afila", "rv", "afirv"}
    if content == "2d_left_atrial_strain":
        return role in {"la", "afirla", "afila"}
    if content == "2d_right_ventricular_strain":
        return role in {"rv", "afirv"}
    return True


def _reject_obsolete_segmentation_data_config(data_config: dict[str, Any]) -> None:
    obsolete = {
        "sample_unit",
        "sector_resample_enabled",
        "sector_resample_depth_start_cm",
        "sector_resample_depth_end_cm",
        "sector_resample_tilt_deg",
        "sector_resample_width_deg",
    }
    present = sorted(key for key in obsolete if key in data_config and data_config.get(key) not in {None, ""})
    if present:
        raise ValueError(
            "Segmentation now runs in native beamspace; remove obsolete data config keys: " + ", ".join(present)
        )


def _cartesian_sector_mask(
    geometry: SectorGeometry,
    grid: CartesianGrid,
    *,
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
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


def _cartesian_grid_sample_grid(
    geometry: SectorGeometry,
    grid: CartesianGrid,
    *,
    source_shape: tuple[int, int],
) -> np.ndarray:
    out_h, out_w = grid.shape
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


def _cartesian_image_sample_grid(grid: CartesianGrid) -> np.ndarray:
    out_h, out_w = grid.shape
    x = np.linspace(-1.0, 1.0, out_w, dtype=np.float64)
    y = np.linspace(-1.0, 1.0, out_h, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)
    return np.stack([xx, yy], axis=-1).astype(np.float32)


def _string_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if isinstance(value, (tuple, list)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    text = str(value).strip()
    return (text,) if text else ()


def _is_contour_annotation(annotation: Any) -> bool:
    field = str(annotation.field).lower()
    return "contour" in field or str(annotation.value.path).endswith("_contour")


def _segmentation_qrs_trigger_times(store: Any, source_store: Any, panel: Any, *, bmode_metadata: Any) -> np.ndarray:
    values: list[np.ndarray] = []
    for annotation in getattr(panel, "annotations", ()):
        if not _is_qrs_annotation(annotation):
            continue
        if annotation.time is not None and annotation.time.path in store.group:
            values.append(np.asarray(store.group[annotation.time.path][:], dtype=np.float64))
        elif annotation.value.path in store.group:
            values.append(np.asarray(store.group[annotation.value.path][:], dtype=np.float64))
    values.extend(_segmentation_qrs_arrays_from_mapping(getattr(panel, "raw", None)))
    values.extend(_segmentation_qrs_arrays_from_mapping(bmode_metadata))
    values.extend(_segmentation_qrs_arrays_from_paths(store, str(panel.role_id)))
    if source_store is not store:
        values.extend(_segmentation_qrs_arrays_from_paths(source_store, str(panel.role_id)))
    if not values:
        return np.asarray([], dtype=np.float64)
    qrs = np.concatenate([np.asarray(value, dtype=np.float64).reshape(-1) for value in values], axis=0)
    qrs = qrs[np.isfinite(qrs)]
    return np.unique(np.sort(qrs)).astype(np.float64, copy=False)


def _qrs_alignment_metadata(bmode_metadata: Any, qrs: np.ndarray) -> dict[str, Any]:
    raw = dict(bmode_metadata) if isinstance(bmode_metadata, Mapping) else {}
    public = raw.get("metadata")
    public_metadata = dict(public) if isinstance(public, Mapping) else {}
    public_metadata["qrs_trigger_times"] = np.asarray(qrs, dtype=np.float64).reshape(-1).tolist()
    raw["metadata"] = public_metadata
    return raw


def _segmentation_qrs_arrays_from_mapping(raw: Any) -> list[np.ndarray]:
    if not isinstance(raw, Mapping):
        return []
    values: list[np.ndarray] = []
    for key in ("ecg_qrs_trigger_times", "qrs_trigger_times", "volume_qrs_trigger_times"):
        if key in raw:
            value = _optional_float_array(raw[key])
            if value is not None:
                values.append(value)
    public = raw.get("metadata")
    if isinstance(public, Mapping):
        for key in ("ecg_qrs_trigger_times", "qrs_trigger_times"):
            if key in public:
                value = _optional_float_array(public[key])
                if value is not None:
                    values.append(value)
    linked = raw.get("linked_recording")
    if isinstance(linked, Mapping):
        for key in ("ecg_qrs_trigger_times", "qrs_trigger_times"):
            if key in linked:
                value = _optional_float_array(linked[key])
                if value is not None:
                    values.append(value)
    return values


def _segmentation_qrs_arrays_from_paths(store: Any, role_id: str) -> list[np.ndarray]:
    candidate_pairs = (
        (f"timestamps/{role_id}_ecg_qrs", f"data/{role_id}_ecg_qrs"),
        (f"timestamps/{role_id}_qrs", f"data/{role_id}_qrs"),
        ("timestamps/ecg_qrs", "data/ecg_qrs"),
    )
    values: list[np.ndarray] = []
    for timestamp_path, data_path in candidate_pairs:
        path = timestamp_path if timestamp_path in store.group else data_path
        if path in store.group:
            values.append(np.asarray(store.group[path][:], dtype=np.float64))
    return values


def _optional_float_array(value: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    arr = arr[np.isfinite(arr)]
    return arr if arr.size else None


def _is_qrs_annotation(annotation: Any) -> bool:
    field = str(annotation.field).lower()
    path = str(annotation.value.path).lower()
    time_path = "" if annotation.time is None else str(annotation.time.path).lower()
    return "qrs" in field or path.endswith("_ecg_qrs") or time_path.endswith("_ecg_qrs")
