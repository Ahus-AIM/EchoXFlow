"""Focused spectral Doppler metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from echoxflow.manifest import manifest_documents


@dataclass(frozen=True)
class SpectralCursorBox:
    """Spectral Doppler cursor/sample box in image coordinates."""

    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class SpectralMetadata:
    """Metadata needed to interpret a spectral Doppler matrix."""

    data_path: str
    cursor_box: SpectralCursorBox | None = None
    baseline_row: float | None = None
    nyquist_limit_mps: float | None = None
    row_velocity_mps: np.ndarray | None = None
    raw: Mapping[str, Any] | None = None


def spectral_metadata_from_attrs(
    data_path: str,
    group_attrs: Mapping[str, Any],
    *,
    row_count: int | None = None,
) -> SpectralMetadata:
    """Extract typed spectral metadata from recording-level manifest attributes."""
    raw = _spectral_document(data_path, group_attrs)
    if raw is None:
        return SpectralMetadata(data_path=data_path)
    baseline = _optional_float(raw.get("baseline_row"))
    nyquist = _optional_float(_first(raw, "nyquist_limit_mps", "velocity_limit_mps", "velocity_scale_mps"))
    axis = _velocity_axis(raw, row_count=row_count, baseline_row=baseline, nyquist_limit_mps=nyquist)
    return SpectralMetadata(
        data_path=data_path,
        cursor_box=_cursor_box(raw),
        baseline_row=baseline,
        nyquist_limit_mps=nyquist,
        row_velocity_mps=axis,
        raw=raw,
    )


def _spectral_document(data_path: str, group_attrs: Mapping[str, Any]) -> Mapping[str, Any] | None:
    name = data_path.strip("/").removeprefix("data/")
    candidates = {name}
    if name in {"1d_pulsed_wave_doppler", "1d_continuous_wave_doppler"}:
        candidates.add("spectral_doppler")
        candidates.add("pw_doppler" if name == "1d_pulsed_wave_doppler" else "cw_doppler")
    for document in _manifest_documents(group_attrs):
        for key in ("spectral_metadata", "spectral", "doppler_spectrum"):
            value = document.get(key)
            if isinstance(value, dict):
                return value
        for track in _items(document.get("tracks")):
            role = _semantic_id(track)
            data_ref = track.get("data")
            array_path = _array_ref_path(data_ref)
            if role in candidates or array_path == data_path:
                return track
        sectors = document.get("sectors")
        if not isinstance(sectors, list):
            continue
        for sector in sectors:
            if not isinstance(sector, dict):
                continue
            role = _semantic_id(sector)
            if role in candidates or role.startswith("1d_"):
                return sector
    return None


def _manifest_documents(group_attrs: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    return manifest_documents(group_attrs)


def _array_ref_path(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    path = value.get("array_path") or value.get("zarr_path") or value.get("path")
    return str(path).strip("/") if path else None


def _cursor_box(raw: Mapping[str, Any]) -> SpectralCursorBox | None:
    value = raw.get("cursor_box")
    if isinstance(value, Mapping):
        values = [
            value.get("x"),
            value.get("y"),
            value.get("width"),
            value.get("height"),
        ]
    elif isinstance(value, (list, tuple)) and len(value) == 4:
        values = list(value)
    else:
        return None
    try:
        x, y, width, height = _float_quad(values)
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite([x, y, width, height])) or width <= 0.0 or height <= 0.0:
        return None
    return SpectralCursorBox(x=x, y=y, width=width, height=height)


def _float_quad(values: list[Any]) -> tuple[float, float, float, float]:
    if len(values) != 4 or any(value is None for value in values):
        raise ValueError("Expected four finite cursor box values")
    return float(values[0]), float(values[1]), float(values[2]), float(values[3])


def _velocity_axis(
    raw: Mapping[str, Any],
    *,
    row_count: int | None,
    baseline_row: float | None,
    nyquist_limit_mps: float | None,
) -> np.ndarray | None:
    value = raw.get("row_velocity_mps")
    if isinstance(value, (list, tuple)):
        axis = np.asarray(value, dtype=np.float32).reshape(-1)
        return axis if axis.size and np.all(np.isfinite(axis)) else None
    if row_count is None or baseline_row is None or nyquist_limit_mps is None:
        return None
    rows = np.arange(int(row_count), dtype=np.float32)
    baseline = float(baseline_row)
    scale = np.where(rows <= baseline, max(1.0, baseline), max(1.0, row_count - 1 - baseline))
    return np.asarray((baseline - rows) / scale * float(nyquist_limit_mps), dtype=np.float32)


def _first(mapping: Mapping[str, Any], *keys: str) -> Any | None:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _items(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _semantic_id(value: Mapping[str, Any]) -> str:
    return str(value.get("semantic_id") or value.get("sector_role_id") or value.get("track_role_id") or "").strip()


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None
