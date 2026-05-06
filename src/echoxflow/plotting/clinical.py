"""Clinical-view rendering data preparation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import numpy as np

from echoxflow.loading import LoadedArray
from echoxflow.plotting.gating import (
    BloodGateConfig,
    TissueGateConfig,
    blood_gate,
    normalize_bmode_intensity,
    normalize_doppler_power,
    tissue_gate,
)
from echoxflow.scan import CartesianGrid, ImageLayer, compose_layers, layer_to_rgba, sector_to_cartesian
from echoxflow.streams import default_value_range_for_path

_COLOR_DOPPLER_HIDDEN_CENTER_FRACTION = 0.20


def clinical_loaded_arrays(loaded_arrays: tuple[LoadedArray, ...]) -> tuple[LoadedArray, ...]:
    """Build Cartesian clinical-view arrays from loaded 2D streams."""
    by_path = {loaded.data_path: loaded for loaded in loaded_arrays}
    bmode = _first_available(
        by_path,
        (
            "data/2d_brightness_mode",
            "data/2d_brightness_mode_0",
            "data/2d_biplane_brightness_mode",
            "data/2d_triplane_brightness_mode",
        ),
    )
    if bmode is None or not _has_geometry(bmode):
        converted = tuple(_clinical_single(loaded) for loaded in loaded_arrays if _has_geometry(loaded))
        return converted or loaded_arrays

    overlays = _clinical_overlays(bmode, by_path)
    if overlays:
        return overlays

    return (
        _clinical_single(bmode),
        *tuple(_clinical_single(loaded) for loaded in loaded_arrays if loaded is not bmode and _has_geometry(loaded)),
    )


def _clinical_overlays(bmode: LoadedArray, by_path: dict[str, LoadedArray]) -> tuple[LoadedArray, ...]:
    overlays: list[LoadedArray] = []
    velocity = by_path.get("data/2d_color_doppler_velocity")
    power = by_path.get("data/2d_color_doppler_power")
    if velocity is not None and power is not None and _has_geometry(velocity) and _has_geometry(power):
        overlays.append(_clinical_color_doppler(bmode, velocity, power))
    tissue = by_path.get("data/tissue_doppler")
    if tissue is not None and _has_geometry(tissue):
        overlays.append(_clinical_tissue_doppler(bmode, tissue))
    return tuple(overlays)


def _clinical_color_doppler(bmode: LoadedArray, velocity: LoadedArray, power: LoadedArray) -> LoadedArray:
    bmode_geometry = _geometry(bmode)
    velocity_geometry = _geometry(velocity)
    power_geometry = _geometry(power)
    bmode_data = np.asarray(bmode.data)
    timeline_source, timeline_timestamps = _clinical_timeline_source(bmode, velocity, power)
    grid = CartesianGrid.from_sector_height(bmode_geometry, int(_frame_at(bmode_data, 0).shape[0]))
    frames = []
    for index, time_s in enumerate(timeline_timestamps):
        bmode_frame = _frame_at_time(bmode, float(time_s), fallback_index=index)
        velocity_frame = _frame_at_time(velocity, time_s, fallback_index=index)
        power_frame = _frame_at_time(power, time_s, fallback_index=index)
        bmode_cart = sector_to_cartesian(bmode_frame, bmode_geometry, grid=grid, interpolation="linear")
        velocity_cart = sector_to_cartesian(velocity_frame, velocity_geometry, grid=grid, interpolation="linear")
        power_cart = sector_to_cartesian(power_frame, power_geometry, grid=grid, interpolation="linear")
        region = bmode_cart.mask & velocity_cart.mask & power_cart.mask
        velocity_range = _value_range(velocity, velocity_cart.data)
        gate = blood_gate(
            bmode_cart.data,
            power_cart.data,
            config=BloodGateConfig(),
            region_mask=region,
        )
        velocity_gate = gate & _outside_middle_colormap_band(
            velocity_cart.data,
            value_range=velocity_range,
            hidden_fraction=_COLOR_DOPPLER_HIDDEN_CENTER_FRACTION,
        )
        power_alpha = normalize_doppler_power(power_cart.data) * velocity_gate.astype(np.float32)
        frames.append(
            compose_layers(
                [
                    ImageLayer(
                        data=bmode_cart.data,
                        cmap="grayscale",
                        value_range=_value_range(bmode, bmode_cart.data),
                        mask=bmode_cart.mask,
                    ),
                    ImageLayer(
                        data=velocity_cart.data,
                        cmap="color_doppler_velocity",
                        alpha=0.9,
                        mask=velocity_gate,
                        value_range=velocity_range,
                    ),
                    ImageLayer(
                        data=power_cart.data,
                        cmap="color_doppler_power",
                        alpha=0.25,
                        mask=power_alpha,
                        value_range=_value_range(power, power_cart.data),
                    ),
                ],
                background=None,
            )
        )
    return _clinical_loaded(
        source=timeline_source,
        name="clinical_color_doppler",
        data=np.asarray(frames, dtype=np.float32),
        label_path="data/clinical_color_doppler",
        grid=grid,
        timestamps=timeline_timestamps,
        attrs={
            "annotation_overlays": _combined_annotation_overlays(bmode, velocity, power),
            "clinical_colorbar_data_path": velocity.data_path,
            "clinical_colorbar_value_range": _value_range(velocity, np.asarray(velocity.data)),
            "clinical_color_doppler_geometry": velocity_geometry,
            "clinical_color_doppler_sector": None if velocity.stream is None else velocity.stream.metadata.raw,
        },
    )


def _clinical_tissue_doppler(bmode: LoadedArray, tissue: LoadedArray) -> LoadedArray:
    bmode_geometry = _geometry(bmode)
    tissue_geometry = _geometry(tissue)
    bmode_data = np.asarray(bmode.data)
    timeline_source, timeline_timestamps = _clinical_timeline_source(bmode, tissue)
    grid = CartesianGrid.from_sector_height(bmode_geometry, int(_frame_at(bmode_data, 0).shape[0]))
    frames = []
    for index, time_s in enumerate(timeline_timestamps):
        bmode_frame = _frame_at_time(bmode, float(time_s), fallback_index=index)
        tissue_frame = _frame_at_time(tissue, time_s, fallback_index=index)
        bmode_cart = sector_to_cartesian(bmode_frame, bmode_geometry, grid=grid, interpolation="linear")
        tissue_rgba = _tissue_rgba_frame(tissue, tissue_frame)
        tissue_cart = sector_to_cartesian(tissue_rgba, tissue_geometry, grid=grid, interpolation="linear")
        region = bmode_cart.mask & tissue_cart.mask
        gate = tissue_gate(
            bmode_cart.data,
            config=TissueGateConfig(),
            region_mask=region,
        )
        frames.append(
            _compose_tissue_overlay(
                bmode=bmode,
                bmode_data=bmode_cart.data,
                bmode_mask=bmode_cart.mask,
                tissue_rgba=tissue_cart.data,
                tissue_mask=gate,
            )
        )
    return _clinical_loaded(
        source=timeline_source,
        name="clinical_tissue_doppler",
        data=np.asarray(frames, dtype=np.float32),
        label_path="data/clinical_tissue_doppler",
        grid=grid,
        timestamps=timeline_timestamps,
        attrs={
            "annotation_overlays": _combined_annotation_overlays(bmode, tissue),
            "clinical_colorbar_data_path": tissue.data_path,
            "clinical_colorbar_value_range": _value_range(tissue, np.asarray(tissue.data)),
        },
    )


def _clinical_single(loaded: LoadedArray) -> LoadedArray:
    geometry = _geometry(loaded)
    data = np.asarray(loaded.data)
    frame_count = _frame_count(data)
    grid = CartesianGrid.from_sector_height(geometry, int(_frame_at(data, 0).shape[0]))
    converted = []
    for index in range(frame_count):
        cart = sector_to_cartesian(_frame_at(data, index), geometry, grid=grid, interpolation="linear")
        converted.append(_masked_rgba(loaded, cart.data, cart.mask))
    return _clinical_loaded(
        source=loaded,
        name=f"clinical_{loaded.name}",
        data=np.asarray(converted),
        label_path=loaded.data_path,
        grid=grid,
    )


def _masked_rgba(loaded: LoadedArray, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cast(
        np.ndarray,
        layer_to_rgba(
            ImageLayer(
                data=data,
                cmap=loaded.data_path,
                value_range=_value_range(loaded, data),
                mask=mask,
            )
        ),
    )


def _clinical_loaded(
    *,
    source: LoadedArray,
    name: str,
    data: np.ndarray,
    label_path: str,
    grid: CartesianGrid,
    timestamps: np.ndarray | None = None,
    attrs: Mapping[str, object] | None = None,
) -> LoadedArray:
    resolved_timestamps = source.timestamps if timestamps is None else np.asarray(timestamps, dtype=np.float64)
    return LoadedArray(
        name=name,
        data_path=label_path,
        data=data,
        timestamps_path=source.timestamps_path,
        timestamps=resolved_timestamps,
        sample_rate_hz=_sample_rate_from_timestamps(resolved_timestamps) or source.sample_rate_hz,
        attrs={**dict(source.attrs), "clinical_source": source.data_path, "clinical_grid": grid, **dict(attrs or {})},
        stream=source.stream,
    )


def _combined_annotation_overlays(*loaded_arrays: LoadedArray) -> tuple[object, ...]:
    overlays: list[object] = []
    seen_singletons: set[object] = set()
    for loaded in loaded_arrays:
        raw = loaded.attrs.get("annotation_overlays")
        if isinstance(raw, tuple):
            for overlay in raw:
                if isinstance(overlay, Mapping) and overlay.get("kind") == "sampling_gate":
                    if "sampling_gate" in seen_singletons:
                        continue
                    seen_singletons.add("sampling_gate")
                if isinstance(overlay, Mapping) and overlay.get("kind") == "sampling_line":
                    key = ("sampling_line", _sampling_line_key(overlay))
                    if key in seen_singletons:
                        continue
                    seen_singletons.add(key)
                overlays.append(overlay)
    return tuple(overlays)


def _sampling_line_key(overlay: Mapping[object, object]) -> tuple[object, ...]:
    points = np.asarray(overlay.get("points"), dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
        return ()
    rounded = np.round(points[:, :2].astype(np.float64), decimals=5)
    return tuple(float(value) for value in rounded.reshape(-1))


def _compose_tissue_overlay(
    *,
    bmode: LoadedArray,
    bmode_data: np.ndarray,
    bmode_mask: np.ndarray,
    tissue_rgba: np.ndarray,
    tissue_mask: np.ndarray,
) -> np.ndarray:
    base = layer_to_rgba(
        ImageLayer(
            data=bmode_data,
            cmap="grayscale",
            value_range=_value_range(bmode, bmode_data),
            mask=bmode_mask,
        )
    )
    visible = np.asarray(tissue_mask, dtype=bool)
    intensity = _tissue_bmode_multiplier(bmode_data, value_range=_value_range(bmode, bmode_data))
    out = np.asarray(base, dtype=np.float32).copy()
    out[visible, :3] = np.asarray(tissue_rgba, dtype=np.float32)[visible, :3] * intensity[visible, None]
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def _tissue_bmode_multiplier(bmode_data: np.ndarray, *, value_range: tuple[float, float]) -> np.ndarray:
    normalized = normalize_bmode_intensity(bmode_data, value_range=value_range)
    return np.asarray(np.clip(normalized / 0.8, 0.0, 1.0), dtype=np.float32)


def _tissue_rgba_frame(tissue: LoadedArray, tissue_frame: np.ndarray) -> np.ndarray:
    return cast(
        np.ndarray,
        layer_to_rgba(
            ImageLayer(
                data=tissue_frame,
                cmap="tissue_doppler",
                value_range=_value_range(tissue, tissue_frame),
            )
        ),
    )


def _outside_middle_colormap_band(
    values: np.ndarray,
    *,
    value_range: tuple[float, float],
    hidden_fraction: float,
) -> np.ndarray:
    low, high = float(value_range[0]), float(value_range[1])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.ones(np.asarray(values).shape, dtype=bool)
    half_width = float(hidden_fraction) * 0.5
    normalized = (np.asarray(values, dtype=np.float32) - low) / (high - low)
    return (normalized <= 0.5 - half_width) | (normalized >= 0.5 + half_width)


def _first_available(by_path: dict[str, LoadedArray], paths: tuple[str, ...]) -> LoadedArray | None:
    for path in paths:
        if path in by_path:
            return by_path[path]
    return None


def _has_geometry(loaded: LoadedArray) -> bool:
    return (
        loaded.stream is not None and loaded.stream.metadata.geometry is not None and np.asarray(loaded.data).ndim >= 2
    )


def _geometry(loaded: LoadedArray):
    if loaded.stream is None or loaded.stream.metadata.geometry is None:
        raise ValueError(f"{loaded.data_path} is missing sector geometry metadata")
    return loaded.stream.metadata.geometry


def _value_range(loaded: LoadedArray, values: np.ndarray) -> tuple[float, float]:
    if loaded.stream is not None and loaded.stream.metadata.value_range is not None:
        return loaded.stream.metadata.value_range
    return default_value_range_for_path(loaded.data_path, values) or (0.0, 1.0)


def _clinical_timeline_source(*loaded_arrays: LoadedArray) -> tuple[LoadedArray, np.ndarray]:
    candidates: list[tuple[float, LoadedArray, np.ndarray]] = []
    for loaded in loaded_arrays:
        timestamps = _valid_frame_timestamps(loaded)
        if timestamps is None:
            continue
        fps = _sample_rate_from_timestamps(timestamps)
        if fps is not None and np.isfinite(fps) and fps > 0.0:
            candidates.append((float(fps), loaded, timestamps))
    if candidates:
        _fps, loaded, timestamps = max(candidates, key=lambda item: item[0])
        return loaded, timestamps
    source = loaded_arrays[0]
    count = _frame_count(np.asarray(source.data))
    return source, np.arange(max(1, count), dtype=np.float64)


def _valid_frame_timestamps(loaded: LoadedArray) -> np.ndarray | None:
    if loaded.timestamps is None:
        return None
    timestamps = np.asarray(loaded.timestamps, dtype=np.float64).reshape(-1)
    if timestamps.size != _frame_count(np.asarray(loaded.data)) or timestamps.size <= 1:
        return None
    if not np.all(np.isfinite(timestamps)):
        return None
    return timestamps


def _sample_rate_from_timestamps(timestamps: np.ndarray | None) -> float | None:
    if timestamps is None:
        return None
    values = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if values.size < 2:
        return None
    diffs = np.diff(values)
    valid = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if valid.size == 0:
        return None
    return float(1.0 / np.median(valid))


def _frame_count(data: np.ndarray) -> int:
    arr = np.asarray(data)
    return int(arr.shape[0]) if arr.ndim >= 3 else 1


def _frame_at(data: np.ndarray, index: int) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim >= 3:
        return np.asarray(arr[int(np.clip(index, 0, arr.shape[0] - 1))])
    return arr


def _frame_at_time(loaded: LoadedArray, time_s: float, *, fallback_index: int) -> np.ndarray:
    data = np.asarray(loaded.data)
    if loaded.timestamps is None:
        return _frame_at(data, fallback_index)
    timestamps = np.asarray(loaded.timestamps, dtype=np.float64).reshape(-1)
    if timestamps.size == 0:
        return _frame_at(data, fallback_index)
    index = int(np.argmin(np.abs(timestamps - float(time_s))))
    return _frame_at(data, index)


def _frame_time(loaded: LoadedArray, index: int) -> float:
    if loaded.timestamps is None:
        return float(index)
    timestamps = np.asarray(loaded.timestamps, dtype=np.float64).reshape(-1)
    if timestamps.size == 0:
        return float(index)
    return float(timestamps[int(np.clip(index, 0, timestamps.size - 1))])
