"""Focused spectral Doppler metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, cast

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
    baseline = _baseline_row(raw, row_count=row_count)
    nyquist = _optional_float(
        next(
            (
                raw[key]
                for key in ("nyquist_mps", "nyquist_limit_mps", "velocity_limit_mps", "velocity_scale_mps")
                if key in raw
            ),
            None,
        )
    )
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
    if name == "1d_pulsed_wave_doppler":
        candidates.update(("spectral_doppler", "pw_doppler", "pulsed_wave", "pulsed_wave_doppler"))
    elif name == "1d_continuous_wave_doppler":
        candidates.update(("spectral_doppler", "cw_doppler", "continuous_wave", "continuous_wave_doppler"))
    for document in manifest_documents(group_attrs):
        for key in ("spectral_metadata", "spectral", "doppler_spectrum"):
            value = document.get(key)
            if isinstance(value, dict):
                return value
        for track in document.get("tracks") or ():
            if not isinstance(track, Mapping):
                continue
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
        x, y, width, height = (float(cast(Any, value)) for value in values)
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite([x, y, width, height])) or width <= 0.0 or height <= 0.0:
        return None
    return SpectralCursorBox(x=x, y=y, width=width, height=height)


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
    scale = max(1.0, float(row_count - 1) / 2.0)
    return np.asarray((baseline - rows) / scale * float(nyquist_limit_mps), dtype=np.float32)


def _baseline_row(raw: Mapping[str, Any], *, row_count: int | None) -> float | None:
    baseline = _optional_float(raw.get("baseline_row"))
    if baseline is not None or row_count is None:
        return baseline
    baseline_frac = _optional_float(
        next((raw[key] for key in ("spectral_row_baseline_frac", "baseline_frac") if key in raw), None)
    )
    return None if baseline_frac is None else float(baseline_frac) * max(0, int(row_count) - 1)


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
