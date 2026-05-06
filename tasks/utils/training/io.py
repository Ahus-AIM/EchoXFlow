from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from echoxflow import RecordingArray, RecordingRecord
from echoxflow.preview import (
    common_preview_arrays,
    source_bmode_array,
    source_ecg_arrays,
    write_preview_pair,
    write_preview_recording_video,
    zero_ecg_array,
)
from echoxflow.streams import (
    TDI_LINEAR_UINT8_STORAGE_ENCODING,
    TDI_NATIVE_STORAGE_ENCODING,
    encode_tdi_linear_uint8_codes,
    encode_tdi_native_codes,
)
from tasks.utils.training.runtime import _PREVIEW_FPS, _PREVIEW_MAX_FPS, _PREVIEW_STYLE, _PREVIEW_VIEW_MODE

_common_preview_arrays = common_preview_arrays
_write_preview_pair = write_preview_pair


def _write_preview_recording_video(
    *,
    record: RecordingRecord,
    preview_dir: Path,
    epoch: int,
    split: str,
    suffix: str,
    arrays: Sequence[RecordingArray],
    attrs: Mapping[str, Any],
    modalities: tuple[str, ...],
) -> None:
    render_options: dict[str, Any] = {
        "view_mode": _PREVIEW_VIEW_MODE,
        "max_fps": _PREVIEW_MAX_FPS,
        "style": _PREVIEW_STYLE,
    }
    write_preview_recording_video(
        record=record,
        preview_dir=preview_dir,
        epoch=epoch,
        split=split,
        suffix=suffix,
        arrays=arrays,
        attrs=attrs,
        modalities=modalities,
        **render_options,
    )


def _source_bmode_array(store: Any, sample: object, data_path: str = "data/2d_brightness_mode") -> RecordingArray:
    return source_bmode_array(store, sample, data_path=data_path)


def _source_ecg_arrays(store: Any) -> tuple[RecordingArray, ...]:
    return source_ecg_arrays(store)


def _zero_ecg_array(timestamps: np.ndarray | None, *, frame_count: int) -> RecordingArray:
    return zero_ecg_array(timestamps, frame_count=frame_count, fallback_fps=_PREVIEW_FPS)


def _resolve_segmentation_panel(store: Any, role_id: str | None) -> tuple[Any, str, Any | None]:
    try:
        obj = store.load_object()
        panel = next(panel for panel in obj.panels if role_id is None or panel.role_id == role_id)
        return store.open_reference(panel.bmode.recording), panel.bmode.data_path, panel
    except (StopIteration, ValueError, KeyError):
        return store, "data/2d_brightness_mode", None


def _resolve_segmentation_bmode(store: Any, role_id: str | None) -> tuple[Any, str]:
    source_store, bmode_path, _panel = _resolve_segmentation_panel(store, role_id)
    return source_store, bmode_path


def _native_tdi_extra_arrays(store: Any, source_stream: Any) -> tuple[RecordingArray, ...]:
    metadata = getattr(source_stream, "metadata", None)
    fenc_table_path = getattr(metadata, "fenc_table_path", None)
    if not fenc_table_path:
        return ()
    return (
        RecordingArray(
            data_path=str(fenc_table_path),
            values=store.load_array(str(fenc_table_path)),
            content_type=str(fenc_table_path).removeprefix("data/"),
        ),
    )


def _encode_tissue_doppler_for_source(
    *,
    store: Any,
    values: np.ndarray,
    source_stream: Any,
    velocity_limit_mps: float,
) -> np.ndarray:
    metadata = getattr(source_stream, "metadata", None)
    storage_encoding = getattr(metadata, "storage_encoding", None)
    if storage_encoding == TDI_NATIVE_STORAGE_ENCODING:
        fenc_table_path = getattr(metadata, "fenc_table_path", None)
        if not fenc_table_path:
            raise ValueError("Native TDI preview source is missing fenc_table_path metadata")
        return encode_tdi_native_codes(
            values,
            fenc_table=store.load_array(str(fenc_table_path)),
            velocity_scale_mps=velocity_limit_mps,
        )
    if storage_encoding == TDI_LINEAR_UINT8_STORAGE_ENCODING:
        return encode_tdi_linear_uint8_codes(values, velocity_scale_mps=velocity_limit_mps)
    return _finite_array(values)


def _preview_attrs(store: Any) -> Mapping[str, Any]:
    return deepcopy(dict(getattr(store.group, "attrs", {})))


def _color_full_region_preview_attrs(
    store: Any,
    *,
    velocity_limit_mps: float,
    output_shape: tuple[int, int],
) -> Mapping[str, Any]:
    attrs = dict(_preview_attrs(store))
    manifest = attrs.get("recording_manifest")
    if not isinstance(manifest, dict):
        return attrs
    sectors = manifest.get("sectors")
    if not isinstance(sectors, list):
        return attrs
    bmode_sector = _find_sector(sectors, "bmode")
    if bmode_sector is None:
        return attrs
    replacements = {
        "2d_color_doppler_velocity": _full_region_doppler_sector(
            bmode_sector,
            semantic_id="2d_color_doppler_velocity",
            velocity_limit_mps=velocity_limit_mps,
            output_shape=output_shape,
        ),
        "2d_color_doppler_power": _full_region_doppler_sector(
            bmode_sector,
            semantic_id="2d_color_doppler_power",
            velocity_limit_mps=None,
            output_shape=output_shape,
        ),
    }
    updated_sectors = []
    seen: set[str] = set()
    for sector in sectors:
        semantic_id = _sector_semantic_id(sector)
        if semantic_id in replacements:
            updated_sectors.append(replacements[semantic_id])
            seen.add(semantic_id)
        else:
            updated_sectors.append(sector)
    for semantic_id, sector in replacements.items():
        if semantic_id not in seen:
            updated_sectors.append(sector)
    manifest["sectors"] = updated_sectors
    return attrs


def _full_region_doppler_sector(
    bmode_sector: Mapping[str, Any],
    *,
    semantic_id: str,
    velocity_limit_mps: float | None,
    output_shape: tuple[int, int],
) -> dict[str, Any]:
    sector = deepcopy(dict(bmode_sector))
    sector["semantic_id"] = semantic_id
    if "sector_role_id" in sector:
        sector["sector_role_id"] = semantic_id
    if "track_role_id" in sector:
        sector["track_role_id"] = semantic_id
    for key in (
        "colormap",
        "native_tdi_lookup_tables",
        "tdi_tables",
        "storage_encoding",
        "velocity_limit_source",
        "velocity_scale_mps",
        "velocity_scale_source",
    ):
        sector.pop(key, None)
    geometry = sector.get("geometry")
    if isinstance(geometry, Mapping):
        updated_geometry = deepcopy(dict(geometry))
        updated_geometry["grid_size"] = [int(output_shape[0]), int(output_shape[1])]
        sector["geometry"] = updated_geometry
    sector["timestamps"] = {"array_path": "timestamps/2d_color_doppler", "format": "zarr_array"}
    if velocity_limit_mps is None:
        sector.pop("velocity_limit_mps", None)
        sector["value_range"] = [0.0, 1.0]
    else:
        sector["velocity_limit_mps"] = float(velocity_limit_mps)
        sector["value_range"] = [-float(velocity_limit_mps), float(velocity_limit_mps)]
    return sector


def _find_sector(sectors: Sequence[object], semantic_id: str) -> Mapping[str, Any] | None:
    for sector in sectors:
        if isinstance(sector, Mapping) and _sector_semantic_id(sector) == semantic_id:
            return sector
    return None


def _sector_semantic_id(sector: object) -> str:
    if not isinstance(sector, Mapping):
        return ""
    return str(sector.get("semantic_id") or sector.get("sector_role_id") or sector.get("track_role_id") or "")


def _resize_video(video: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    values = _finite_array(video)
    if tuple(values.shape[-2:]) == tuple(output_shape):
        return values
    tensor = torch.from_numpy(values[:, None])
    resized = torch.nn.functional.interpolate(tensor, size=output_shape, mode="bilinear", align_corners=False)
    return np.asarray(resized[:, 0].numpy(), dtype=np.float32)


def _resize_channel_video(video: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    values = _finite_array(video)
    if values.ndim != 4:
        raise ValueError(f"channel video must be [T,C,H,W], got {values.shape}")
    if tuple(values.shape[-2:]) == tuple(output_shape):
        return values
    tensor = torch.from_numpy(values)
    resized = torch.nn.functional.interpolate(tensor, size=output_shape, mode="bilinear", align_corners=False)
    return np.asarray(resized.numpy(), dtype=np.float32)


def _resize_rgb_video(video: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    values = _finite_array(video)
    if values.ndim != 4 or values.shape[-1] not in {3, 4}:
        raise ValueError(f"RGB video must be [T,H,W,3|4], got {values.shape}")
    if tuple(values.shape[-3:-1]) == tuple(output_shape):
        return values
    tensor = torch.from_numpy(values[..., :3].transpose(0, 3, 1, 2))
    resized = torch.nn.functional.interpolate(tensor, size=output_shape, mode="bilinear", align_corners=False)
    return np.asarray(resized.numpy().transpose(0, 2, 3, 1), dtype=np.float32)


def _is_ecg_path(path: str) -> bool:
    data_path = str(path).strip("/")
    return data_path == "data/ecg" or (data_path.startswith("data/") and data_path.endswith("_ecg"))


def _safe_recording_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-", "."} else "_" for char in value)


def _sample_timestamps(sample: object, field_name: str, *, count: int) -> np.ndarray:
    value = getattr(sample, field_name, None)
    if isinstance(value, Tensor):
        array = value.detach().cpu().numpy()
        timestamps = np.asarray(array[0] if array.ndim >= 2 else array, dtype=np.float32).reshape(-1)
        if int(timestamps.size) == int(count):
            return timestamps
    return np.arange(int(count), dtype=np.float32) / _PREVIEW_FPS


def _preview_output_timestamps(*, count: int) -> np.ndarray:
    return np.arange(int(count), dtype=np.float32) / _PREVIEW_FPS


def _sample_scalar(sample: object, field_name: str) -> float | None:
    value = getattr(sample, field_name, None)
    if not isinstance(value, Tensor):
        return None
    array = value.detach().float().cpu().numpy().reshape(-1)
    if array.size == 0 or not np.isfinite(array[0]):
        return None
    return max(float(abs(array[0])), 1e-6)


def _tensor_video(value: Tensor) -> np.ndarray:
    array = value.detach().float().cpu().numpy()
    if array.ndim != 5:
        raise ValueError(f"preview tensors must be [B,C,T,H,W], got shape {array.shape}")
    return np.asarray(array[0], dtype=np.float32).transpose(1, 0, 2, 3)


def _finite_array(value: np.ndarray) -> np.ndarray:
    return np.asarray(np.where(np.isfinite(value), value, 0.0), dtype=np.float32)


def _symmetric_limit(*arrays: np.ndarray) -> float:
    values = np.concatenate([np.ravel(_finite_array(array)) for array in arrays])
    if values.size == 0:
        return 1.0
    return max(float(np.max(np.abs(values))), 1e-6)
