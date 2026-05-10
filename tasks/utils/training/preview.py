from __future__ import annotations

import logging
import random
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn

from echoxflow import RecordingArray, RecordingRecord, open_recording
from echoxflow.loading import LoadedArray
from echoxflow.plotting import PlotViewMode, blend_segmentation_rgb
from echoxflow.plotting.cartesian import (
    _compose_tissue_overlay,
    _outside_middle_colormap_band,
    _tissue_rgba_frame,
)
from echoxflow.plotting.gating import (
    BloodGateConfig,
    TissueGateConfig,
    blood_gate,
    normalize_doppler_power,
    tissue_gate,
)
from echoxflow.preview import common_preview_arrays, write_preview_pair
from echoxflow.scan import ImageLayer, compose_layers
from tasks.utils.training.device import _autocast_context, move_to_device
from tasks.utils.training.io import (
    _color_full_region_preview_attrs,
    _encode_tissue_doppler_for_source,
    _finite_array,
    _native_tdi_extra_arrays,
    _preview_attrs,
    _resize_video,
    _resolve_segmentation_panel,
    _sample_scalar,
    _sample_timestamps,
    _sector_semantic_id,
    _symmetric_limit,
    _tensor_video,
)
from tasks.utils.training.runtime import _COLOR_POWER_FLOOR, _PREVIEW_MAX_FPS, _PREVIEW_STYLE, AmpConfig


@dataclass(frozen=True)
class _PreviewRecordingSpecs:
    common: tuple[RecordingArray, ...]
    real: tuple[RecordingArray, ...]
    predicted: tuple[RecordingArray, ...]
    attrs: Mapping[str, Any]
    modalities: tuple[str, ...]
    view_mode: PlotViewMode

    def arrays(self, suffix: str) -> tuple[RecordingArray, ...]:
        if suffix == "real":
            return self.real
        if suffix == "predicted":
            return self.predicted
        raise ValueError(f"unsupported preview suffix {suffix!r}")


def _write_epoch_previews(
    *,
    module: nn.Module,
    samples: Mapping[str, object | None],
    run_dir: Path | None,
    epoch: int,
    amp: AmpConfig,
    device: torch.device,
    logger: logging.Logger,
) -> None:
    if run_dir is None:
        return
    preview_dir = run_dir / "previews"
    was_training = module.training
    module.eval()
    try:
        for split, sample in samples.items():
            if sample is None:
                continue
            try:
                _write_parseable_epoch_previews(
                    module=module,
                    sample=sample,
                    preview_dir=preview_dir,
                    epoch=epoch,
                    split=split,
                    amp=amp,
                    device=device,
                )
            except Exception as exc:
                logger.warning("could not write %s preview for epoch %d: %s", split, epoch, exc)
    finally:
        module.train(was_training)


def _random_preview_sample(dataloader: Iterable[object] | None, *, device: torch.device) -> object | None:
    if dataloader is None:
        return None
    dataset = getattr(dataloader, "dataset", None)
    if dataset is None:
        return None
    try:
        sample_count = len(dataset)
    except TypeError:
        return None
    if int(sample_count) <= 0:
        return None
    index = int(np.random.randint(0, int(sample_count)))
    return move_to_device(dataset[index], device)


def _stable_preview_sample(
    dataloader: Iterable[object] | None,
    *,
    device: torch.device,
    seed: int,
    index_fraction: float = 0.5,
) -> object | None:
    if dataloader is None:
        return None
    dataset = getattr(dataloader, "dataset", None)
    if dataset is None:
        return None
    try:
        sample_count = len(dataset)
    except TypeError:
        return None
    if int(sample_count) <= 0:
        return None
    index = min(int(sample_count) - 1, max(0, int(round((int(sample_count) - 1) * float(index_fraction)))))
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        random.seed(int(seed))
        np.random.seed(int(seed) % (2**32))
        torch.manual_seed(int(seed))
        sample = dataset[index]
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
    return move_to_device(sample, device)


def _write_parseable_epoch_previews(
    *,
    module: nn.Module,
    sample: object,
    preview_dir: Path,
    epoch: int,
    split: str,
    amp: AmpConfig,
    device: torch.device,
) -> bool:
    record = getattr(sample, "record", None)
    task_kind = str(getattr(sample, "task_kind", ""))
    preview_spec = _task_preview_spec(task_kind)
    if preview_spec is None or not isinstance(record, RecordingRecord):
        return False
    with torch.no_grad(), _autocast_context(amp, device):
        prediction = preview_spec.prediction_fn(module, sample)
    preview_spec.writer(
        sample=sample,
        record=record,
        prediction=prediction,
        preview_dir=preview_dir,
        epoch=epoch,
        split=split,
    )
    return True


def _task_preview_spec(task_kind: str) -> Any | None:
    try:
        from tasks.registry import task_preview_spec

        return task_preview_spec(task_kind)
    except ValueError:
        return None


def _preview_prediction(module: nn.Module, sample: object, task_kind: str) -> Tensor:
    frames = getattr(sample, "frames", None)
    if not isinstance(frames, Tensor):
        raise TypeError(f"{task_kind} preview sample is missing frames")
    if task_kind == "segmentation":
        return cast(Tensor, cast(Any, module)(frames, None))
    if task_kind == "tissue_doppler":
        return cast(
            Tensor,
            cast(Any, module)(
                frames,
                velocity_scale_mps_per_px_frame=getattr(sample, "velocity_scale_mps_per_px_frame"),
            ),
        )
    if task_kind == "color_doppler":
        return cast(Tensor, cast(Any, module)(frames, conditioning=getattr(sample, "conditioning")))
    raise ValueError(f"unsupported preview task kind {task_kind!r}")


def segmentation_preview_prediction(module: nn.Module, sample: object) -> Tensor:
    frames = getattr(sample, "frames", None)
    if not isinstance(frames, Tensor):
        raise TypeError("segmentation preview sample is missing frames")
    return cast(Tensor, cast(Any, module)(frames, None))


def tissue_doppler_preview_prediction(module: nn.Module, sample: object) -> Tensor:
    frames = getattr(sample, "frames", None)
    if not isinstance(frames, Tensor):
        raise TypeError("tissue_doppler preview sample is missing frames")
    return cast(
        Tensor,
        cast(Any, module)(
            frames,
            velocity_scale_mps_per_px_frame=getattr(sample, "velocity_scale_mps_per_px_frame"),
        ),
    )


def color_doppler_preview_prediction(module: nn.Module, sample: object) -> Tensor:
    frames = getattr(sample, "frames", None)
    if not isinstance(frames, Tensor):
        raise TypeError("color_doppler preview sample is missing frames")
    return cast(Tensor, cast(Any, module)(frames, conditioning=getattr(sample, "conditioning")))


def _write_segmentation_recording_previews(
    *,
    sample: object,
    record: RecordingRecord,
    prediction: Tensor,
    preview_dir: Path,
    epoch: int,
    split: str,
) -> None:
    specs = _segmentation_recording_preview_specs(sample=sample, record=record, prediction=prediction)
    write_preview_pair(
        record=record,
        preview_dir=preview_dir,
        epoch=epoch,
        split=split,
        common=specs.common,
        build_modality_arrays=specs.arrays,
        attrs=specs.attrs,
        modalities=specs.modalities,
        view_mode=specs.view_mode,
        max_fps=_PREVIEW_MAX_FPS,
        style=_PREVIEW_STYLE,
    )


def _segmentation_recording_preview_specs(
    *,
    sample: object,
    record: RecordingRecord,
    prediction: Tensor,
) -> _PreviewRecordingSpecs:
    store = open_recording(record, root=getattr(sample, "data_root", None))
    source_store, bmode_path, panel = _resolve_segmentation_panel(store, getattr(sample, "role_id", None))
    target = getattr(sample, "target_masks")
    frame_video = _tensor_video(getattr(sample, "frames"))
    if frame_video.shape[1] < 1:
        raise ValueError("segmentation preview sample frames must include at least one image channel")
    bmode_video = frame_video[:, 0]
    output_shape = (int(bmode_video.shape[-2]), int(bmode_video.shape[-1]))
    common_arrays = common_preview_arrays(source_store, sample, bmode_path=bmode_path)
    timing_array = common_arrays[0]
    timestamps = timing_array.timestamps
    expected_video = blend_segmentation_rgb(
        foreground_video=_tensor_video(target),
        bmode_video=bmode_video,
        output_shape=output_shape,
    )
    predicted_video = blend_segmentation_rgb(
        foreground_video=_segmentation_foreground_video(
            prediction,
            target,
            valid_mask=getattr(sample, "target_mask_valid", None),
        ),
        bmode_video=bmode_video,
        output_shape=output_shape,
    )
    classes = _segmentation_class_names(int(target.shape[1]))

    def build_modality_arrays(video: np.ndarray) -> tuple[RecordingArray, ...]:
        return (
            RecordingArray(
                data_path="data/segmentation_overlay",
                values=np.clip(video, 0.0, 1.0).astype(np.float32),
                timestamps=timestamps,
                timestamps_path=timing_array.timestamps_path,
                content_type="segmentation_overlay",
                attrs={
                    "display": "okabe_ito_blended_rgb",
                    "classes": classes,
                },
            ),
        )

    return _PreviewRecordingSpecs(
        common=tuple(common_arrays[1:]),
        real=build_modality_arrays(expected_video),
        predicted=build_modality_arrays(predicted_video),
        attrs=_segmentation_preview_attrs(
            source_store,
            panel=panel,
            output_shape=output_shape,
            timestamps_path=timing_array.timestamps_path,
        ),
        modalities=("segmentation_overlay",),
        view_mode=_preview_view_mode(sample),
    )


def _preview_view_mode(sample: object) -> PlotViewMode:
    if str(getattr(sample, "coordinate_space", "")).strip().lower() == "cartesian":
        return "beamspace"
    return "both"


_segmentation_preview_view_mode = _preview_view_mode


def _segmentation_preview_attrs(
    source_store: Any,
    *,
    panel: Any | None,
    output_shape: tuple[int, int],
    timestamps_path: str | None,
) -> Mapping[str, Any]:
    attrs = dict(_preview_attrs(source_store))
    geometry = getattr(panel, "geometry", None)
    if not isinstance(geometry, Mapping):
        raise ValueError("segmentation preview requires panel geometry for scan-converted overlay")
    manifest = attrs.get("recording_manifest")
    if not isinstance(manifest, Mapping):
        manifest = {"manifest_type": "2d"}
    else:
        manifest = deepcopy(dict(manifest))
    sectors = manifest.get("sectors")
    updated_sectors = (
        [dict(sector) for sector in sectors if isinstance(sector, Mapping)] if isinstance(sectors, list) else []
    )
    overlay_sector = _segmentation_overlay_sector(
        geometry=geometry,
        output_shape=output_shape,
        timestamps_path=timestamps_path,
    )
    updated_sectors = [sector for sector in updated_sectors if _sector_semantic_id(sector) != "segmentation_overlay"]
    updated_sectors.append(overlay_sector)
    manifest["sectors"] = updated_sectors
    attrs["recording_manifest"] = manifest
    return attrs


def _segmentation_overlay_sector(
    *,
    geometry: Mapping[str, Any],
    output_shape: tuple[int, int],
    timestamps_path: str | None,
) -> dict[str, Any]:
    overlay_geometry = deepcopy(dict(geometry))
    overlay_geometry["grid_size"] = [int(output_shape[0]), int(output_shape[1])]
    sector: dict[str, Any] = {
        "semantic_id": "segmentation_overlay",
        "sector_role_id": "segmentation_overlay",
        "geometry": overlay_geometry,
        "value_range": [0.0, 1.0],
    }
    if timestamps_path is not None:
        sector["timestamps"] = {"array_path": timestamps_path, "format": "zarr_array"}
    return sector


def _write_tissue_recording_previews(
    *,
    sample: object,
    record: RecordingRecord,
    prediction: Tensor,
    preview_dir: Path,
    epoch: int,
    split: str,
) -> None:
    specs = _tissue_recording_preview_specs(sample=sample, record=record, prediction=prediction)
    write_preview_pair(
        record=record,
        preview_dir=preview_dir,
        epoch=epoch,
        split=split,
        common=specs.common,
        build_modality_arrays=specs.arrays,
        attrs=specs.attrs,
        modalities=specs.modalities,
        view_mode=specs.view_mode,
        max_fps=_PREVIEW_MAX_FPS,
        style=_PREVIEW_STYLE,
    )


def _tissue_recording_preview_specs(
    *,
    sample: object,
    record: RecordingRecord,
    prediction: Tensor,
) -> _PreviewRecordingSpecs:
    store = open_recording(record, root=getattr(sample, "data_root", None))
    target = getattr(sample, "doppler_target")
    expected_video = _tensor_video(target[:, :1])[:, 0]
    predicted_video = _tensor_video(prediction[:, :1])[:, 0]
    target_timestamps = _sample_timestamps(sample, "doppler_timestamps", count=int(expected_video.shape[0]))
    limit = _sample_scalar(sample, "sector_velocity_limit_mps") or _symmetric_limit(expected_video, predicted_video)
    predicted_video = _wrap_velocity_band(predicted_video, velocity_limit_mps=limit)
    source_stream = store.load_stream("data/tissue_doppler")
    extra_arrays = _native_tdi_extra_arrays(store, source_stream)
    common_arrays = _common_preview_arrays(store, sample, extra=extra_arrays)
    videos = {"real": expected_video, "predicted": predicted_video}
    uses_cartesian_prediction = _preview_view_mode(sample) == "beamspace"

    def build_modality_arrays(suffix: str) -> Sequence[RecordingArray]:
        encoded = _encode_tissue_doppler_for_source(
            store=store,
            values=videos[suffix],
            source_stream=source_stream,
            velocity_limit_mps=limit,
        )
        arrays = [
            RecordingArray(
                data_path="data/tissue_doppler",
                values=encoded,
                timestamps=target_timestamps,
                timestamps_path="timestamps/tissue_doppler",
                content_type="tissue_doppler",
            )
        ]
        if uses_cartesian_prediction:
            arrays.append(
                RecordingArray(
                    data_path="data/cartesian_tissue_doppler",
                    values=_cartesian_tissue_overlay_video(
                        sample=sample,
                        tissue_video=videos[suffix],
                        velocity_limit_mps=limit,
                    ),
                    timestamps=target_timestamps,
                    timestamps_path="timestamps/cartesian_tissue_doppler",
                    content_type="cartesian_tissue_doppler",
                    attrs={
                        "cartesian_colorbar_data_path": "data/tissue_doppler",
                        "cartesian_colorbar_value_range": [-float(limit), float(limit)],
                    },
                )
            )
        return tuple(arrays)

    return _PreviewRecordingSpecs(
        common=common_arrays,
        real=tuple(build_modality_arrays("real")),
        predicted=tuple(build_modality_arrays("predicted")),
        attrs=_preview_attrs(store),
        modalities=(
            ("cartesian_tissue_doppler",) if uses_cartesian_prediction else ("2d_brightness_mode", "tissue_doppler")
        ),
        view_mode=_preview_view_mode(sample),
    )


def _wrap_velocity_band(video: np.ndarray, *, velocity_limit_mps: float) -> np.ndarray:
    limit = max(float(velocity_limit_mps), 1e-6)
    period = 2.0 * limit
    values = _finite_array(video)
    return cast(np.ndarray, (np.remainder(values + limit, period) - limit).astype(np.float32))


def _segmentation_foreground_video(prediction: Tensor, target: Tensor, valid_mask: Tensor | None = None) -> np.ndarray:
    target_channels = int(target.shape[1])
    expected_channels = target_channels + 1
    if int(prediction.shape[1]) != expected_channels:
        raise ValueError(
            f"segmentation prediction has {int(prediction.shape[1])} channels, "
            f"expected {expected_channels} (background + {target_channels} targets)"
        )
    if valid_mask is not None:
        valid = valid_mask.to(device=prediction.device, dtype=prediction.dtype)
        valid = valid.expand(-1, -1, -1, prediction.shape[-2], prediction.shape[-1])
        prediction = prediction.clone()
        prediction[:, 1:] = prediction[:, 1:].masked_fill(valid <= 0.0, torch.finfo(prediction.dtype).min)
    foreground = torch.softmax(prediction, dim=1)[:, 1:]
    if valid_mask is not None:
        foreground = foreground * valid
    return _tensor_video(foreground)


def _write_color_recording_previews(
    *,
    sample: object,
    record: RecordingRecord,
    prediction: Tensor,
    preview_dir: Path,
    epoch: int,
    split: str,
) -> None:
    specs = _color_recording_preview_specs(sample=sample, record=record, prediction=prediction)
    write_preview_pair(
        record=record,
        preview_dir=preview_dir,
        epoch=epoch,
        split=split,
        common=specs.common,
        build_modality_arrays=specs.arrays,
        attrs=specs.attrs,
        modalities=specs.modalities,
        view_mode=specs.view_mode,
        max_fps=_PREVIEW_MAX_FPS,
        style=_PREVIEW_STYLE,
    )


def _color_recording_preview_specs(
    *,
    sample: object,
    record: RecordingRecord,
    prediction: Tensor,
) -> _PreviewRecordingSpecs:
    store = open_recording(record, root=getattr(sample, "data_root", None))
    target = getattr(sample, "doppler_target")
    expected_video = _tensor_video(target)
    predicted_video = _tensor_video(prediction)
    velocity_limit = _sample_scalar(sample, "nyquist_mps") or _symmetric_limit(
        expected_video[:, 0],
        predicted_video[:, 0],
    )
    common_arrays = _common_preview_arrays(store, sample)
    videos = {"real": expected_video, "predicted": predicted_video}

    def build_modality_arrays(suffix: str) -> Sequence[RecordingArray]:
        video = videos[suffix]
        velocity = _color_velocity(video, velocity_limit_mps=velocity_limit).astype(np.float32)
        power = np.clip(_color_power(video), 0.0, 1.0).astype(np.float32)
        doppler_timestamps = _sample_timestamps(sample, "target_timestamps", count=int(velocity.shape[0]))
        arrays = [
            RecordingArray(
                data_path="data/2d_color_doppler_velocity",
                values=velocity.astype(np.float32),
                timestamps=doppler_timestamps,
                timestamps_path="timestamps/2d_color_doppler",
                content_type="2d_color_doppler_velocity",
            ),
            RecordingArray(
                data_path="data/2d_color_doppler_power",
                values=power,
                timestamps=doppler_timestamps,
                timestamps_path="timestamps/2d_color_doppler",
                content_type="2d_color_doppler_power",
            ),
        ]
        if _preview_view_mode(sample) == "beamspace":
            arrays.append(
                RecordingArray(
                    data_path="data/cartesian_color_doppler",
                    values=_cartesian_color_overlay_video(
                        sample=sample,
                        velocity_video=velocity,
                        power_video=power,
                        velocity_limit_mps=velocity_limit,
                    ),
                    timestamps=doppler_timestamps,
                    timestamps_path="timestamps/cartesian_color_doppler",
                    content_type="cartesian_color_doppler",
                    attrs={
                        "cartesian_colorbar_data_path": "data/2d_color_doppler_velocity",
                        "cartesian_colorbar_value_range": [-float(velocity_limit), float(velocity_limit)],
                    },
                )
            )
        return tuple(arrays)

    attrs = _color_full_region_preview_attrs(
        store,
        velocity_limit_mps=velocity_limit,
        output_shape=(int(expected_video.shape[-2]), int(expected_video.shape[-1])),
    )
    return _PreviewRecordingSpecs(
        common=common_arrays,
        real=tuple(build_modality_arrays("real")),
        predicted=tuple(build_modality_arrays("predicted")),
        attrs=attrs,
        modalities=(
            ("cartesian_color_doppler",)
            if _preview_view_mode(sample) == "beamspace"
            else ("2d_brightness_mode", "2d_color_doppler_velocity", "2d_color_doppler_power")
        ),
        view_mode=_preview_view_mode(sample),
    )


_PREVIEW_SPEC_BUILDERS: dict[str, Any] = {
    "segmentation": _segmentation_recording_preview_specs,
    "tissue_doppler": _tissue_recording_preview_specs,
    "color_doppler": _color_recording_preview_specs,
}


def _prediction_recording_preview_specs(
    *,
    task_kind: str,
    sample: object,
    record: RecordingRecord,
    prediction: Tensor,
) -> _PreviewRecordingSpecs:
    builder = _PREVIEW_SPEC_BUILDERS.get(task_kind)
    if builder is None:
        raise ValueError(f"unsupported preview task kind {task_kind!r}")
    return cast(_PreviewRecordingSpecs, builder(sample=sample, record=record, prediction=prediction))


def _common_preview_arrays(
    store: Any,
    sample: object,
    *,
    extra: Sequence[RecordingArray] = (),
) -> tuple[RecordingArray, ...]:
    common = common_preview_arrays(store, sample, extra=extra)
    if _preview_view_mode(sample) != "beamspace":
        return common
    bmode = _tensor_video(getattr(sample, "frames"))[:, 0]
    return (
        RecordingArray(
            data_path="data/2d_brightness_mode",
            values=_preview_bmode_uint8(bmode),
            timestamps=_sample_timestamps(sample, "frame_timestamps", count=int(bmode.shape[0])),
            timestamps_path="timestamps/2d_brightness_mode",
            content_type="2d_brightness_mode",
        ),
        *common[1:],
    )


def _preview_bmode_uint8(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype == np.uint8:
        return arr
    finite = np.nan_to_num(arr.astype(np.float32, copy=False), nan=0.0, posinf=255.0, neginf=0.0)
    if finite.size and float(finite.min()) >= 0.0 and float(finite.max()) <= 1.0:
        finite = finite * 255.0
    return np.clip(np.rint(finite), 0.0, 255.0).astype(np.uint8)


def _cartesian_color_overlay_video(
    *,
    sample: object,
    velocity_video: np.ndarray,
    power_video: np.ndarray,
    velocity_limit_mps: float,
) -> np.ndarray:
    output_shape = _cartesian_preview_bmode_shape(sample)
    velocity = _resize_video(_finite_array(velocity_video), output_shape)
    power = np.clip(_resize_video(_finite_array(power_video), output_shape), 0.0, 1.0)
    bmode = _cartesian_overlay_bmode(sample, count=int(velocity.shape[0]), output_shape=output_shape)
    frames = []
    velocity_range = (-float(velocity_limit_mps), float(velocity_limit_mps))
    for bmode_frame, velocity_frame, power_frame in zip(bmode, velocity, power):
        region = np.isfinite(bmode_frame) & np.isfinite(velocity_frame) & np.isfinite(power_frame)
        gate = blood_gate(
            bmode_frame,
            power_frame,
            config=BloodGateConfig(bmode_value_range=(0.0, 1.0), power_value_range=(0.0, 1.0)),
            region_mask=region,
        )
        velocity_gate = gate & _outside_middle_colormap_band(
            velocity_frame,
            value_range=velocity_range,
            hidden_fraction=0.20,
        )
        power_alpha = normalize_doppler_power(power_frame) * velocity_gate.astype(np.float32)
        frames.append(
            compose_layers(
                [
                    ImageLayer(
                        data=bmode_frame,
                        cmap="grayscale",
                        value_range=(0.0, 1.0),
                        mask=region,
                    ),
                    ImageLayer(
                        data=velocity_frame,
                        cmap="color_doppler_velocity",
                        alpha=0.9,
                        mask=velocity_gate,
                        value_range=velocity_range,
                    ),
                    ImageLayer(
                        data=power_frame,
                        cmap="color_doppler_power",
                        alpha=0.25,
                        mask=power_alpha,
                        value_range=(0.0, 1.0),
                    ),
                ],
                background=None,
            )
        )
    return np.asarray(frames, dtype=np.float32)


def _cartesian_tissue_overlay_video(
    *,
    sample: object,
    tissue_video: np.ndarray,
    velocity_limit_mps: float,
) -> np.ndarray:
    output_shape = _cartesian_preview_bmode_shape(sample)
    tissue = _resize_video(_finite_array(tissue_video), output_shape)
    bmode_video = _cartesian_overlay_bmode(sample, count=int(tissue.shape[0]), output_shape=output_shape)
    bmode = LoadedArray(
        name="2d_brightness_mode",
        data_path="data/2d_brightness_mode",
        data=bmode_video,
        timestamps_path=None,
        timestamps=None,
        sample_rate_hz=None,
        attrs={},
        stream=cast(Any, SimpleNamespace(metadata=SimpleNamespace(value_range=(0.0, 1.0)))),
    )
    tissue_loaded = LoadedArray(
        name="tissue_doppler",
        data_path="data/tissue_doppler",
        data=tissue,
        timestamps_path=None,
        timestamps=None,
        sample_rate_hz=None,
        attrs={},
        stream=cast(
            Any,
            SimpleNamespace(
                metadata=SimpleNamespace(value_range=(-float(velocity_limit_mps), float(velocity_limit_mps)))
            ),
        ),
    )
    frames = []
    for bmode_frame, tissue_frame in zip(bmode_video, tissue):
        tissue_rgba = _tissue_rgba_frame(tissue_loaded, tissue_frame)
        gate = tissue_gate(
            bmode_frame,
            config=TissueGateConfig(bmode_value_range=(0.0, 1.0)),
            region_mask=np.isfinite(bmode_frame) & np.isfinite(tissue_frame),
        )
        frames.append(
            _compose_tissue_overlay(
                bmode=bmode,
                bmode_data=bmode_frame,
                bmode_mask=np.isfinite(bmode_frame),
                tissue_rgba=tissue_rgba,
                tissue_mask=gate,
            )
        )
    return np.asarray(frames, dtype=np.float32)


def _cartesian_overlay_bmode(sample: object, *, count: int, output_shape: tuple[int, int]) -> np.ndarray:
    frames = _tensor_video(getattr(sample, "frames"))[:, 0]
    bmode = _aligned_interval_bmode(frames, count=int(count))
    return _resize_video(np.clip(_finite_array(bmode), 0.0, 1.0), output_shape)


def _cartesian_preview_bmode_shape(sample: object) -> tuple[int, int]:
    frames = _tensor_video(getattr(sample, "frames"))
    return int(frames.shape[-2]), int(frames.shape[-1])


def _aligned_interval_bmode(frames: np.ndarray, *, count: int) -> np.ndarray:
    bmode = _finite_array(frames)
    if bmode.ndim != 3:
        raise ValueError(f"Cartesian preview B-mode must be [T,H,W], got {bmode.shape}")
    target_count = int(count)
    if target_count <= 0:
        return bmode[:0]
    if int(bmode.shape[0]) == target_count:
        return bmode
    if int(bmode.shape[0]) >= 2:
        intervals = 0.5 * (bmode[:-1] + bmode[1:])
        if int(intervals.shape[0]) == target_count:
            return np.asarray(intervals, dtype=np.float32)
        if target_count % int(intervals.shape[0]) == 0:
            return np.repeat(intervals, target_count // int(intervals.shape[0]), axis=0).astype(np.float32, copy=False)
    indices = np.rint(np.linspace(0, max(0, int(bmode.shape[0]) - 1), target_count)).astype(np.int64)
    return np.asarray(bmode[np.clip(indices, 0, int(bmode.shape[0]) - 1)], dtype=np.float32)


def _segmentation_class_names(channel_count: int) -> list[str]:
    names = [
        "LV blood",
        "LV myocardium",
        "LA blood",
        "LA myocardium",
        "RV blood",
        "RV myocardium",
    ]
    return names[: max(0, int(channel_count))]


def _color_velocity(video: np.ndarray, *, velocity_limit_mps: float) -> np.ndarray:
    if video.shape[1] <= 0:
        return np.zeros((int(video.shape[0]), int(video.shape[2]), int(video.shape[3])), dtype=np.float32)
    raw_velocity = _finite_array(video[:, 0])
    power = _color_power(video)
    velocity = np.where(
        power > _COLOR_POWER_FLOOR,
        raw_velocity,
        np.zeros_like(raw_velocity, dtype=np.float32),
    )
    return cast(np.ndarray, np.clip(velocity, -velocity_limit_mps, velocity_limit_mps).astype(np.float32))


def _color_power(video: np.ndarray) -> np.ndarray:
    if video.shape[1] <= 1:
        return np.ones((int(video.shape[0]), int(video.shape[2]), int(video.shape[3])), dtype=np.float32)
    return cast(np.ndarray, np.clip(_finite_array(video[:, 1]), 0.0, 1.0))
