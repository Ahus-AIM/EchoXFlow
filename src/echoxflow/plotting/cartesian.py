"""Cartesian-view rendering data preparation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import cast

import numpy as np
from matplotlib.colors import Colormap

from echoxflow.loading import LoadedArray
from echoxflow.plotting.gating import (
    BloodGateConfig,
    TissueGateConfig,
    blood_gate,
    normalize_bmode_intensity,
    tissue_gate,
)
from echoxflow.scan import (
    CartesianGrid,
    CartesianImage,
    ImageLayer,
    Interpolation,
    SectorGeometry,
    compose_layers,
    layer_to_rgba,
    sector_stack_to_cartesian_array,
)
from echoxflow.schema import color_flow_velocity_codes
from echoxflow.streams import default_value_range_for_path

_COLOR_DOPPLER_HIDDEN_CENTER_FRACTION = 0.20
_COLOR_DOPPLER_VELOCITY_ALPHA = 0.9
# The scanner indexes the tissue table's brightness axis with k = 255 * (slope * B-mode + offset), B-mode in [0, 1].
_TISSUE_TABLE_SLOPE = 0.85
_TISSUE_TABLE_OFFSET = 0.02


def cartesian_loaded_arrays(
    loaded_arrays: tuple[LoadedArray, ...],
    *,
    blood_gate: BloodGateConfig | None = None,
    max_depth_m: float | None = None,
    tissue_brightness_lift: float = 0.0,
    tissue_gate: TissueGateConfig | None = None,
) -> tuple[LoadedArray, ...]:
    """Build Cartesian cartesian-view arrays from loaded 2D streams.

    ``blood_gate`` and ``tissue_gate`` override the default Doppler gates; ``max_depth_m`` crops the sectors below
    that depth; ``tissue_brightness_lift`` brightens the tissue Doppler colours (see :class:`PlotStyle`).
    """
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
        converted = tuple(
            _cartesian_single(loaded, max_depth_m=max_depth_m) for loaded in loaded_arrays if _has_geometry(loaded)
        )
        return converted or loaded_arrays

    overlays = _cartesian_overlays(
        bmode,
        by_path,
        blood_gate=blood_gate,
        max_depth_m=max_depth_m,
        tissue_brightness_lift=tissue_brightness_lift,
        tissue_gate=tissue_gate,
    )
    if overlays:
        return overlays

    return (
        _cartesian_single(bmode, max_depth_m=max_depth_m),
        *tuple(
            _cartesian_single(loaded, max_depth_m=max_depth_m)
            for loaded in loaded_arrays
            if loaded is not bmode and _has_geometry(loaded)
        ),
    )


def _cartesian_overlays(
    bmode: LoadedArray,
    by_path: dict[str, LoadedArray],
    *,
    blood_gate: BloodGateConfig | None = None,
    max_depth_m: float | None = None,
    tissue_brightness_lift: float = 0.0,
    tissue_gate: TissueGateConfig | None = None,
) -> tuple[LoadedArray, ...]:
    overlays: list[LoadedArray] = []
    velocity = by_path.get("data/2d_color_doppler_velocity")
    power = by_path.get("data/2d_color_doppler_power")
    if velocity is not None and power is not None and _has_geometry(velocity) and _has_geometry(power):
        overlays.append(
            _cartesian_color_doppler(bmode, velocity, power, gate_config=blood_gate, max_depth_m=max_depth_m)
        )
    tissue = by_path.get("data/tissue_doppler")
    if tissue is not None and _has_geometry(tissue):
        overlays.append(
            _cartesian_tissue_doppler(
                bmode,
                tissue,
                max_depth_m=max_depth_m,
                brightness_lift=tissue_brightness_lift,
                gate_config=tissue_gate,
            )
        )
    return tuple(overlays)


def _display_grid(geometry: SectorGeometry, height: int, max_depth_m: float | None) -> CartesianGrid:
    """The Cartesian grid of a sector, cropped to ``max_depth_m`` when that is shallower than the sector."""
    if max_depth_m is not None and float(max_depth_m) < float(geometry.depth_end_m):
        geometry = replace(geometry, depth_end_m=max(float(max_depth_m), float(geometry.depth_start_m) + 1e-3))
    return CartesianGrid.from_sector_height(geometry, height)


def _within_depth(image: CartesianImage, max_depth_m: float | None) -> np.ndarray:
    """The image mask without the pixels deeper than ``max_depth_m`` from the apex at the grid origin."""
    if max_depth_m is None:
        return np.asarray(image.mask, dtype=bool)
    height, width = image.grid.shape
    xs = np.linspace(image.grid.x_range_m[0], image.grid.x_range_m[1], width, dtype=np.float64)
    ys = np.linspace(image.grid.y_range_m[0], image.grid.y_range_m[1], height, dtype=np.float64)
    radii = np.hypot(xs[np.newaxis, :], ys[:, np.newaxis])
    return np.asarray(image.mask, dtype=bool) & (radii <= float(max_depth_m))


def _cartesian_color_doppler(
    bmode: LoadedArray,
    velocity: LoadedArray,
    power: LoadedArray,
    *,
    gate_config: BloodGateConfig | None = None,
    max_depth_m: float | None = None,
) -> LoadedArray:
    bmode_geometry = _geometry(bmode)
    velocity_geometry = _geometry(velocity)
    power_geometry = _geometry(power)
    bmode_data = np.asarray(bmode.data)
    _, timeline_timestamps = _cartesian_timeline_source(bmode, velocity, power)
    grid = _display_grid(bmode_geometry, int(_frame_at(bmode_data, 0).shape[0]), max_depth_m)
    # Velocity and power are sampled with nearest-neighbour so wrapped velocities across the Nyquist boundary are
    # never blended into false mid-range values; the colormap is applied only after sampling.
    bmode_stack = _cartesian_stack(bmode, bmode_geometry, grid, timeline_timestamps, interpolation="linear")
    velocity_stack = _cartesian_stack(velocity, velocity_geometry, grid, timeline_timestamps, interpolation="nearest")
    power_stack = _cartesian_stack(power, power_geometry, grid, timeline_timestamps, interpolation="nearest")
    bmode_mask = _within_depth(bmode_stack, max_depth_m)
    region = bmode_mask & velocity_stack.mask & power_stack.mask
    flow_colors = _cartesian_flow_color_stack(velocity, velocity_geometry, grid, timeline_timestamps)
    frames = [
        _color_doppler_frame(
            bmode=bmode,
            velocity=velocity,
            bmode_data=bmode_stack.data[index],
            bmode_mask=bmode_mask,
            velocity_data=velocity_stack.data[index],
            power_data=power_stack.data[index],
            region=region,
            flow_rgb=None if flow_colors is None else flow_colors[index],
            gate_config=gate_config,
        )
        for index in range(len(timeline_timestamps))
    ]
    return _cartesian_loaded(
        source=bmode,
        name="cartesian_color_doppler",
        data=np.asarray(frames, dtype=np.float32),
        label_path="data/cartesian_color_doppler",
        grid=grid,
        timestamps=timeline_timestamps,
        attrs={
            "annotation_overlays": _combined_annotation_overlays(bmode, velocity, power),
            "colormap": velocity.attrs.get("colormap"),
            "cartesian_colorbar_data_path": velocity.data_path,
            "cartesian_colorbar_table": velocity.attrs.get("flow_color_table"),
            "cartesian_colorbar_value_range": _value_range(velocity, np.asarray(velocity.data)),
            "cartesian_color_doppler_geometry": velocity_geometry,
            "cartesian_color_doppler_sector": None if velocity.stream is None else velocity.stream.metadata.raw,
        },
    )


def _color_doppler_frame(
    *,
    bmode: LoadedArray,
    velocity: LoadedArray,
    bmode_data: np.ndarray,
    bmode_mask: np.ndarray,
    velocity_data: np.ndarray,
    power_data: np.ndarray,
    region: np.ndarray,
    flow_rgb: np.ndarray | None,
    gate_config: BloodGateConfig | None = None,
) -> np.ndarray:
    """Composite one scan-converted colour Doppler frame: B-mode with the gated velocity colours on top.

    All arrays are on the same Cartesian grid. `flow_rgb` is the `[0, 1]` RGB the recording's two-dimensional
    FlowColorTable gives each pixel, or None to colour `velocity_data` through the velocity colormap. The scanner
    draws velocity colour opaque over B-mode and adds no power tint, so there is no further layer.
    """
    velocity_range = _value_range(velocity, velocity_data)
    gate = blood_gate(bmode_data, power_data, config=gate_config or BloodGateConfig(), region_mask=region)
    velocity_gate = gate & _outside_middle_colormap_band(
        velocity_data, value_range=velocity_range, hidden_fraction=_COLOR_DOPPLER_HIDDEN_CENTER_FRACTION
    )
    return cast(
        np.ndarray,
        compose_layers(
            [
                ImageLayer(
                    data=bmode_data,
                    cmap="grayscale",
                    value_range=_value_range(bmode, bmode_data),
                    mask=bmode_mask,
                ),
                ImageLayer(
                    data=velocity_data if flow_rgb is None else flow_rgb,
                    cmap=_own_colormap(velocity, "color_doppler_velocity"),
                    alpha=_COLOR_DOPPLER_VELOCITY_ALPHA,
                    mask=velocity_gate,
                    value_range=velocity_range,
                ),
            ],
            background=None,
        ),
    )


def _cartesian_tissue_doppler(
    bmode: LoadedArray,
    tissue: LoadedArray,
    *,
    max_depth_m: float | None = None,
    brightness_lift: float = 0.0,
    gate_config: TissueGateConfig | None = None,
) -> LoadedArray:
    bmode_geometry = _geometry(bmode)
    tissue_geometry = _geometry(tissue)
    bmode_data = np.asarray(bmode.data)
    timeline_source, timeline_timestamps = _cartesian_timeline_source(bmode, tissue)
    grid = _display_grid(bmode_geometry, int(_frame_at(bmode_data, 0).shape[0]), max_depth_m)
    # The velocity is interpolated as a phase on the aliasing circle, so neighbours across the Nyquist boundary blend
    # into a velocity near the limit instead of a false value near zero; colouring happens afterwards.
    bmode_stack = _cartesian_stack(bmode, bmode_geometry, grid, timeline_timestamps, interpolation="linear")
    tissue_stack = _cartesian_wrapped_stack(
        tissue, tissue_geometry, grid, timeline_timestamps, limit=abs(_value_range(tissue, np.asarray(tissue.data))[1])
    )
    bmode_mask = _within_depth(bmode_stack, max_depth_m)
    region = bmode_mask & tissue_stack.mask
    frames = [
        _tissue_doppler_frame(
            bmode=bmode,
            tissue=tissue,
            bmode_data=bmode_stack.data[index],
            bmode_mask=bmode_mask,
            tissue_data=tissue_stack.data[index],
            region=region,
            brightness_lift=brightness_lift,
            gate_config=gate_config,
        )
        for index in range(len(timeline_timestamps))
    ]
    return _cartesian_loaded(
        source=timeline_source,
        name="cartesian_tissue_doppler",
        data=np.asarray(frames, dtype=np.float32),
        label_path="data/cartesian_tissue_doppler",
        grid=grid,
        timestamps=timeline_timestamps,
        attrs={
            "annotation_overlays": _combined_annotation_overlays(bmode, tissue),
            "colormap": tissue.attrs.get("colormap"),
            "cartesian_colorbar_data_path": tissue.data_path,
            "cartesian_colorbar_table": tissue.attrs.get("tissue_color_table"),
            "cartesian_colorbar_value_range": _value_range(tissue, np.asarray(tissue.data)),
        },
    )


def _tissue_doppler_frame(
    *,
    bmode: LoadedArray,
    tissue: LoadedArray,
    bmode_data: np.ndarray,
    bmode_mask: np.ndarray,
    tissue_data: np.ndarray,
    region: np.ndarray,
    brightness_lift: float = 0.0,
    gate_config: TissueGateConfig | None = None,
) -> np.ndarray:
    """Composite one scan-converted tissue Doppler frame.

    With the recording's brightness-style FlowColorTable the scanner indexes the table by the B-mode brightness of
    each pixel (first axis) and the velocity phase code (second axis); without one the velocity ramp is scaled by
    the B-mode brightness instead. Only pixels the tissue gate passes are coloured. ``brightness_lift`` moves the
    normalised brightness that fraction of the way towards 1 before the table lookup, so dark tissue takes brighter
    colours while bright tissue barely changes and near-black pixels stay dark.
    """
    gate = tissue_gate(bmode_data, config=gate_config or TissueGateConfig(), region_mask=region)
    table = tissue.attrs.get("tissue_color_table")
    if not isinstance(table, np.ndarray):
        return _compose_tissue_overlay(
            bmode=bmode,
            bmode_data=bmode_data,
            bmode_mask=bmode_mask,
            tissue_rgba=_tissue_rgba_frame(tissue, tissue_data),
            tissue_mask=gate,
        )
    base = np.asarray(
        layer_to_rgba(
            ImageLayer(
                data=bmode_data,
                cmap="grayscale",
                value_range=_value_range(bmode, bmode_data),
                mask=bmode_mask,
            )
        ),
        dtype=np.float32,
    )
    low, high = _value_range(bmode, bmode_data)
    brightness = np.clip((np.nan_to_num(bmode_data) - low) / (high - low), 0.0, 1.0)
    # Lift dark tissue towards brighter table rows, bright tissue hardly, and the darkest pixels not at all.
    brightness = brightness + float(brightness_lift) * (1.0 - brightness) * np.minimum(1.0, brightness / 0.15)
    rows = np.clip(np.rint(255.0 * (_TISSUE_TABLE_SLOPE * brightness + _TISSUE_TABLE_OFFSET)), 0, 255).astype(np.intp)
    limit = abs(_value_range(tissue, tissue_data)[1])
    columns = color_flow_velocity_codes(np.nan_to_num(tissue_data), velocity_limit_mps=limit).astype(np.intp)
    base[gate, :3] = table[rows[gate], columns[gate]].astype(np.float32) / 255.0
    return base


def _cartesian_single(loaded: LoadedArray, *, max_depth_m: float | None = None) -> LoadedArray:
    geometry = _geometry(loaded)
    data = np.asarray(loaded.data)
    grid = _display_grid(geometry, int(_frame_at(data, 0).shape[0]), max_depth_m)
    stack = sector_stack_to_cartesian_array(
        data if data.ndim >= 3 else data[np.newaxis], geometry, grid=grid, interpolation=_interpolation_for(loaded)
    )
    mask = _within_depth(stack, max_depth_m)
    converted = [_masked_rgba(loaded, frame, mask) for frame in stack.data]
    return _cartesian_loaded(
        source=loaded,
        name=f"cartesian_{loaded.name}",
        data=np.asarray(converted),
        label_path=loaded.data_path,
        grid=grid,
    )


def _interpolation_for(loaded: LoadedArray) -> Interpolation:
    """Nearest sampling for Doppler scalars, linear for everything else."""
    return "nearest" if "doppler" in loaded.data_path and "brightness" not in loaded.data_path else "linear"


def _cartesian_stack(
    loaded: LoadedArray,
    geometry: SectorGeometry,
    grid: CartesianGrid,
    timestamps: np.ndarray,
    *,
    interpolation: Interpolation,
) -> CartesianImage:
    """Scan-convert the frames nearest to each timeline time in one call, sharing a single lookup."""
    data = np.asarray(loaded.data)
    stack = data if data.ndim >= 3 else data[np.newaxis]
    indices = [_frame_index_at_time(loaded, float(t), fallback_index=i) for i, t in enumerate(timestamps)]
    return sector_stack_to_cartesian_array(stack[indices], geometry, grid=grid, interpolation=interpolation)


def _cartesian_wrapped_stack(
    loaded: LoadedArray, geometry: SectorGeometry, grid: CartesianGrid, timestamps: np.ndarray, *, limit: float
) -> CartesianImage:
    """Scan-convert a velocity that wraps at ``±limit`` with linear interpolation on the aliasing circle."""
    data = np.asarray(loaded.data, dtype=np.float32)
    stack = data if data.ndim >= 3 else data[np.newaxis]
    indices = [_frame_index_at_time(loaded, float(t), fallback_index=i) for i, t in enumerate(timestamps)]
    sampled = sector_stack_to_cartesian_array(
        wrapped_velocity_to_phasor(stack[indices], limit=limit), geometry, grid=grid, interpolation="linear"
    )
    return CartesianImage(data=phasor_to_wrapped_velocity(sampled.data, limit=limit), mask=sampled.mask, grid=grid)


def wrapped_velocity_to_phasor(velocity: np.ndarray, *, limit: float) -> np.ndarray:
    """``[..., 2]`` cosine and sine of the velocity's phase ``pi * v / limit``; NaN velocities map to zero."""
    phase = np.pi * np.nan_to_num(np.asarray(velocity, dtype=np.float32)) / np.float32(max(float(limit), 1e-12))
    return np.stack((np.cos(phase), np.sin(phase)), axis=-1).astype(np.float32)


def phasor_to_wrapped_velocity(phasor: np.ndarray, *, limit: float) -> np.ndarray:
    """Inverse of :func:`wrapped_velocity_to_phasor`; interpolated phasors give the velocity of their mean phase."""
    phasor = np.asarray(phasor, dtype=np.float32)
    velocity: np.ndarray = np.arctan2(phasor[..., 1], phasor[..., 0]) / np.pi * np.float32(limit)
    return velocity.astype(np.float32)


def _cartesian_flow_color_stack(
    velocity: LoadedArray, geometry: SectorGeometry, grid: CartesianGrid, timestamps: np.ndarray
) -> np.ndarray | None:
    """Colour native codes through the table, then interpolate displayed RGB onto the Cartesian grid.

    The table is looked up on exact native codes. Interpolating displayed colours preserves mixed green/cyan
    boundaries; packed indices and wrapped scalar velocities must never be interpolated as ordinary numbers.
    Returns `float32` `(T, H, W, 3)` in `[0, 1]`, or None when the velocity array carries no codes or table.
    """
    codes = velocity.attrs.get("flow_color_codes")
    table = velocity.attrs.get("flow_color_table")
    if not isinstance(codes, np.ndarray) or not isinstance(table, np.ndarray):
        return None
    rgb = velocity.attrs.get("flow_color_rgb")
    if not isinstance(rgb, np.ndarray) or rgb.shape != (*codes.shape, 3):
        rgb = table[(codes >> 8).astype(np.intp), (codes & 255).astype(np.intp)]
    sampled = _cartesian_stack(replace(velocity, data=rgb), geometry, grid, timestamps, interpolation="linear")
    return np.asarray(sampled.data, dtype=np.float32) / np.float32(255.0)


def _own_colormap(loaded: LoadedArray, fallback: str) -> Colormap | str:
    """Return the colormap loaded with the array, else the configured name."""
    own = loaded.attrs.get("colormap")
    return own if isinstance(own, Colormap) else fallback


def _masked_rgba(loaded: LoadedArray, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cast(
        np.ndarray,
        layer_to_rgba(
            ImageLayer(
                data=data,
                cmap=_own_colormap(loaded, loaded.data_path),
                value_range=_value_range(loaded, data),
                mask=mask,
            )
        ),
    )


def _cartesian_loaded(
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
        attrs={**dict(source.attrs), "cartesian_source": source.data_path, "cartesian_grid": grid, **dict(attrs or {})},
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
                cmap=_own_colormap(tissue, "tissue_doppler"),
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


def _cartesian_timeline_source(*loaded_arrays: LoadedArray) -> tuple[LoadedArray, np.ndarray]:
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


def _frame_index_at_time(loaded: LoadedArray, time_s: float, *, fallback_index: int) -> int:
    if loaded.timestamps is None:
        return fallback_index
    timestamps = np.asarray(loaded.timestamps, dtype=np.float64).reshape(-1)
    if timestamps.size == 0:
        return fallback_index
    return int(np.argmin(np.abs(timestamps - float(time_s))))
