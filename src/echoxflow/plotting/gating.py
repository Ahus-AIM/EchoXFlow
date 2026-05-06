"""Reusable clinical-view gating masks for Doppler overlays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from echoxflow.streams import default_value_range_for_path


@dataclass(frozen=True)
class TissueGateConfig:
    """B-mode intensity gate for tissue overlays."""

    min_bmode_intensity: float = 0.05
    max_bmode_intensity: float = 1.0
    bmode_value_range: tuple[float, float] | None = None


@dataclass(frozen=True)
class BloodGateConfig:
    """B-mode and Doppler-power gate for blood-flow overlays."""

    min_doppler_power: float = 0.30
    min_power_minus_bmode: float = 0.12
    max_bmode_intensity: float = 0.90
    bmode_value_range: tuple[float, float] | None = None
    power_value_range: tuple[float, float] | None = None


def tissue_gate(
    bmode: np.ndarray,
    *,
    config: TissueGateConfig | None = None,
    region_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return a tissue overlay mask from normalized B-mode intensity."""
    cfg = config or TissueGateConfig()
    normalized_bmode = normalize_bmode_intensity(bmode, value_range=cfg.bmode_value_range)
    mask = (normalized_bmode >= float(cfg.min_bmode_intensity)) & (normalized_bmode <= float(cfg.max_bmode_intensity))
    return _apply_region_mask(mask, region_mask)


def blood_gate(
    bmode: np.ndarray,
    doppler_power: np.ndarray,
    *,
    config: BloodGateConfig | None = None,
    region_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return a blood-flow mask from Doppler power and B-mode suppression.

    This mirrors the color Doppler sibling script:
    ``power > min_doppler_power`` and
    ``power - bmode > min_power_minus_bmode``.
    """
    cfg = config or BloodGateConfig()
    normalized_bmode = normalize_bmode_intensity(bmode, value_range=cfg.bmode_value_range)
    normalized_power = normalize_doppler_power(doppler_power, value_range=cfg.power_value_range)
    _ensure_same_shape(normalized_bmode, normalized_power)
    mask = (
        (normalized_power > float(cfg.min_doppler_power))
        & ((normalized_power - normalized_bmode) > float(cfg.min_power_minus_bmode))
        & (normalized_bmode <= float(cfg.max_bmode_intensity))
    )
    return _apply_region_mask(mask, region_mask)


def normalize_bmode_intensity(
    bmode: np.ndarray,
    *,
    value_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Normalize B-mode using a fixed expected value range."""
    return _normalize_fixed(
        bmode,
        value_range=value_range or default_value_range_for_path("data/2d_brightness_mode") or (0.0, 255.0),
    )


def normalize_doppler_power(
    doppler_power: np.ndarray,
    *,
    value_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Normalize Doppler power using a fixed expected value range."""
    return _normalize_fixed(
        doppler_power,
        value_range=value_range or default_value_range_for_path("data/2d_color_doppler_power") or (0.0, 1.0),
    )


def _normalize_fixed(values: np.ndarray, *, value_range: tuple[float, float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    vmin, vmax = float(value_range[0]), float(value_range[1])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise ValueError(f"Invalid fixed normalization range: {value_range!r}")
    normalized = (np.nan_to_num(arr, nan=vmin, posinf=vmax, neginf=vmin) - vmin) / (vmax - vmin)
    clipped = np.asarray(np.clip(normalized, 0.0, 1.0), dtype=np.float32)
    return clipped


def _apply_region_mask(mask: np.ndarray, region_mask: np.ndarray | None) -> np.ndarray:
    result = np.asarray(mask, dtype=bool)
    if region_mask is None:
        return result
    region = np.asarray(region_mask, dtype=bool)
    _ensure_same_shape(result, region)
    return np.asarray(result & region, dtype=bool)


def _ensure_same_shape(left: np.ndarray, right: np.ndarray) -> None:
    if left.shape != right.shape:
        raise ValueError(f"Expected matching shapes, got {left.shape} and {right.shape}")
