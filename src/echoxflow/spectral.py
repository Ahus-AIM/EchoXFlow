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
class SpectralCalibration:
    """Per-column spectral calibration, one entry per time column of the matrix.

    Each array is NaN where the source left a column uncalibrated. ``baseline_fraction`` is the position of zero
    velocity measured from the first row; the column spans ``2 * nyquist_mps`` with ``+2 * nyquist * baseline`` at
    the first row edge, the exporter's convention. ``angle_correction_rad`` is reported as published:
    the export declares no convention for whether the velocity scale already includes it, so it is never applied.
    """

    baseline_fraction: np.ndarray | None = None
    nyquist_mps: np.ndarray | None = None
    angle_correction_rad: np.ndarray | None = None

    def sliced(self, start: int, stop: int | None) -> SpectralCalibration:
        def cut(values: np.ndarray | None) -> np.ndarray | None:
            return None if values is None else np.asarray(values)[int(start) : stop]

        return SpectralCalibration(
            baseline_fraction=cut(self.baseline_fraction),
            nyquist_mps=cut(self.nyquist_mps),
            angle_correction_rad=cut(self.angle_correction_rad),
        )

    def calibrated_columns(self) -> np.ndarray | None:
        """Boolean mask of columns with both a valid baseline and a valid Nyquist limit."""
        if self.baseline_fraction is None or self.nyquist_mps is None:
            return None
        baseline = np.asarray(self.baseline_fraction, dtype=np.float64)
        nyquist = np.asarray(self.nyquist_mps, dtype=np.float64)
        return np.asarray(
            np.isfinite(baseline) & (baseline >= 0.0) & (baseline <= 1.0) & np.isfinite(nyquist) & (nyquist > 0.0)
        )

    @property
    def status(self) -> str:
        """``constant``, ``varying``, ``partial`` (some columns uncalibrated) or ``missing``."""
        calibrated = self.calibrated_columns()
        if calibrated is None or not np.any(calibrated):
            return "missing"
        if not np.all(calibrated):
            return "partial"
        baseline = np.asarray(self.baseline_fraction, dtype=np.float64)
        nyquist = np.asarray(self.nyquist_mps, dtype=np.float64)
        constant = np.all(baseline == baseline[0]) and np.all(nyquist == nyquist[0])
        return "constant" if constant else "varying"


@dataclass(frozen=True)
class SpectralMetadata:
    """Metadata needed to interpret a spectral Doppler matrix.

    ``row_velocity_mps`` is one velocity axis shared by every column and is only set when one calibration applies
    to all selected columns. ``column_velocity_mps`` (``[time, rows]``) gives each column its own axis, with NaN for
    columns whose calibration is unknown; nothing is fabricated for them.
    """

    data_path: str
    cursor_box: SpectralCursorBox | None = None
    baseline_row: float | None = None
    nyquist_limit_mps: float | None = None
    row_velocity_mps: np.ndarray | None = None
    raw: Mapping[str, Any] | None = None
    calibration: SpectralCalibration | None = None
    column_velocity_mps: np.ndarray | None = None


def spectral_metadata_from_attrs(
    data_path: str,
    group_attrs: Mapping[str, Any],
    *,
    row_count: int | None = None,
    calibration: SpectralCalibration | None = None,
) -> SpectralMetadata:
    """Extract typed spectral metadata from recording-level manifest attributes.

    ``calibration`` is the per-column calibration loaded from the arrays the track references
    (:func:`spectral_calibration_refs`); with it, physical axes come from the columns' own values and the scalar
    manifest fields are only a fallback for older recordings.
    """
    raw = _spectral_document(data_path, group_attrs)
    if raw is None:
        return SpectralMetadata(data_path=data_path)
    if calibration is not None and calibration.calibrated_columns() is not None:
        return _calibrated_spectral_metadata(data_path, raw, calibration=calibration, row_count=row_count)
    baseline = _baseline_row(raw, row_count=row_count)
    nyquist = _optional_float(
        _first(raw, "nyquist_limit_mps", "nyquist_mps", "velocity_limit_mps", "velocity_scale_mps")
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


#: Track `spectral_calibration` keys holding one value per time column.
SPECTRAL_CALIBRATION_COLUMNS: tuple[str, ...] = ("baseline_fraction", "nyquist_mps", "angle_correction_rad")


def spectral_calibration_refs(data_path: str, group_attrs: Mapping[str, Any]) -> dict[str, str]:
    """Return the per-column calibration arrays a spectral track references, by calibration key."""
    raw = _spectral_document(data_path, group_attrs)
    calibration = None if raw is None else raw.get("spectral_calibration")
    if not isinstance(calibration, Mapping):
        return {}
    refs = {key: _array_ref_path(calibration.get(key)) for key in SPECTRAL_CALIBRATION_COLUMNS}
    return {key: path for key, path in refs.items() if path is not None}


def spectral_segment_offsets_ref(data_path: str, group_attrs: Mapping[str, Any]) -> str | None:
    """Return the source-segment boundary array of a spectral track, if it declares one."""
    raw = _spectral_document(data_path, group_attrs)
    return None if raw is None else _array_ref_path(raw.get("segment_sample_offsets"))


def column_velocity_axes(calibration: SpectralCalibration, *, row_count: int) -> np.ndarray:
    """Return ``[columns, rows]`` row-centre velocities in m/s; uncalibrated columns are NaN."""
    calibrated = calibration.calibrated_columns()
    if calibrated is None:
        raise ValueError("Spectral calibration needs both baseline_fraction and nyquist_mps")
    baseline = np.asarray(calibration.baseline_fraction, dtype=np.float64)[:, None]
    nyquist = np.asarray(calibration.nyquist_mps, dtype=np.float64)[:, None]
    rows = (np.arange(int(row_count), dtype=np.float64)[None, :] + 0.5) / max(1, int(row_count))
    axes = 2.0 * nyquist * (baseline - rows)
    axes[~calibrated] = np.nan
    return np.asarray(axes, dtype=np.float32)


def _calibrated_spectral_metadata(
    data_path: str,
    raw: Mapping[str, Any],
    *,
    calibration: SpectralCalibration,
    row_count: int | None,
) -> SpectralMetadata:
    columns = None if row_count is None else column_velocity_axes(calibration, row_count=row_count)
    baseline_row = nyquist = axis = None
    if calibration.status == "constant":
        baseline_fraction = float(np.asarray(calibration.baseline_fraction)[0])
        nyquist = float(np.asarray(calibration.nyquist_mps)[0])
        if row_count is not None and columns is not None:
            baseline_row = baseline_fraction * int(row_count) - 0.5
            axis = np.asarray(columns[0], dtype=np.float32)
    return SpectralMetadata(
        data_path=data_path,
        cursor_box=_cursor_box(raw),
        baseline_row=baseline_row,
        nyquist_limit_mps=nyquist,
        row_velocity_mps=axis,
        raw=raw,
        calibration=calibration,
        column_velocity_mps=columns,
    )


def _spectral_document(data_path: str, group_attrs: Mapping[str, Any]) -> Mapping[str, Any] | None:
    name = data_path.strip("/").removeprefix("data/")
    candidates = {name}
    if name == "1d_pulsed_wave_doppler":
        candidates.update(("spectral_doppler", "pw_doppler", "pulsed_wave", "pulsed_wave_doppler"))
    elif name == "1d_continuous_wave_doppler":
        candidates.update(("spectral_doppler", "cw_doppler", "continuous_wave", "continuous_wave_doppler"))
    for document in manifest_documents(group_attrs):
        for track in document.get("tracks") or ():
            if isinstance(track, Mapping) and _array_ref_path(track.get("data")) == data_path.strip("/"):
                return track
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
    baseline = float(np.clip(baseline_row, 0.0, max(0, int(row_count) - 1)))
    offsets = baseline - rows
    positive_scale = max(1.0, baseline)
    negative_scale = max(1.0, float(row_count - 1) - baseline)
    scale = np.where(offsets >= 0.0, positive_scale, negative_scale)
    return np.asarray(offsets / scale * float(nyquist_limit_mps), dtype=np.float32)


def _baseline_row(raw: Mapping[str, Any], *, row_count: int | None) -> float | None:
    baseline = _optional_float(raw.get("baseline_row"))
    if baseline is not None:
        return baseline
    baseline_frac = _optional_float(_first(raw, "baseline_frac", "spectral_row_baseline_frac"))
    if baseline_frac is None or row_count is None:
        return None
    return float(np.clip(baseline_frac, 0.0, 1.0) * max(0, int(row_count) - 1))


def _first(mapping: Mapping[str, Any], *keys: str) -> Any | None:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


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
