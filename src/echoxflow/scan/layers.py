"""Layer compositing for direct arrays and Cartesian scan-converted images."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Colormap

from echoxflow.colors import named_listed_colormap


@dataclass(frozen=True)
class ImageLayer:
    data: np.ndarray
    cmap: str | Colormap | None = None
    alpha: float = 1.0
    mask: np.ndarray | None = None
    value_range: tuple[float, float] | None = None


def compose_layers(
    layers: tuple[ImageLayer, ...] | list[ImageLayer], *, background: str | None = "#222222"
) -> np.ndarray:
    """Alpha-compose scalar/RGB layers into an RGBA image."""
    if not layers:
        raise ValueError("At least one image layer is required")
    first = np.asarray(layers[0].data)
    shape = first.shape[:2]
    canvas_rgb = np.zeros((*shape, 3), dtype=np.float32)
    canvas_alpha = np.zeros(shape, dtype=np.float32)
    if background is not None:
        canvas_rgb[...] = _hex_to_rgb(background)
        canvas_alpha[...] = 1.0
    for layer in layers:
        rgba = layer_to_rgba(layer)
        layer_alpha = rgba[..., 3]
        out_alpha = layer_alpha + canvas_alpha * (1.0 - layer_alpha)
        numerator = rgba[..., :3] * layer_alpha[..., None] + canvas_rgb * canvas_alpha[..., None] * (
            1.0 - layer_alpha[..., None]
        )
        canvas_rgb = np.asarray(
            np.divide(
                numerator,
                out_alpha[..., None],
                out=np.zeros_like(canvas_rgb),
                where=out_alpha[..., None] > 0.0,
            ),
            dtype=np.float32,
        )
        canvas_alpha = np.asarray(out_alpha, dtype=np.float32)
    return np.asarray(np.clip(np.dstack([canvas_rgb, canvas_alpha]), 0.0, 1.0), dtype=np.float32)


def layer_to_rgba(layer: ImageLayer) -> np.ndarray:
    data = np.asarray(layer.data)
    if data.ndim == 3 and data.shape[-1] in {3, 4}:
        rgb = data[..., :3].astype(np.float32)
        if np.nanmax(rgb) > 1.0:
            rgb /= 255.0
        alpha = data[..., 3].astype(np.float32) if data.shape[-1] == 4 else np.ones(data.shape[:2], dtype=np.float32)
    else:
        rgb = _scalar_to_rgb(data, layer)
        alpha = np.ones(data.shape[:2], dtype=np.float32)
    alpha *= float(layer.alpha)
    if layer.mask is not None:
        alpha *= np.asarray(layer.mask, dtype=np.float32)
    return np.dstack([np.clip(rgb, 0.0, 1.0), np.clip(alpha, 0.0, 1.0)])


def opacity_from_values(
    values: np.ndarray,
    *,
    low: float | None = None,
    high: float | None = None,
    invert: bool = False,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    lo = float(np.nanmin(arr) if low is None else low)
    hi = float(np.nanmax(arr) if high is None else high)
    alpha = (arr - lo) / max(1e-9, hi - lo)
    alpha = np.clip(alpha, 0.0, 1.0)
    return 1.0 - alpha if invert else alpha


def _scalar_to_rgb(data: np.ndarray, layer: ImageLayer) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    vmin, vmax = layer.value_range or _default_value_range(np.asarray(data))
    if abs(vmax - vmin) < 1e-9:
        normed = np.ones_like(arr, dtype=np.float32) if vmax > 0.0 else np.zeros_like(arr, dtype=np.float32)
    else:
        normed = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
    cmap = _resolve_cmap(layer.cmap)
    if cmap is not None:
        return np.asarray(cmap(normed))[..., :3].astype(np.float32)
    return np.repeat(normed[..., None], 3, axis=-1)


def _default_value_range(values: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        return float(info.min), float(info.max)
    if np.issubdtype(arr.dtype, np.bool_):
        return 0.0, 1.0
    return 0.0, 1.0


def _resolve_cmap(cmap: str | Colormap | None) -> Colormap | None:
    if cmap is None:
        return None
    if isinstance(cmap, Colormap):
        return cmap
    return named_listed_colormap(cmap) or colormaps[cmap]


def _hex_to_rgb(value: str) -> np.ndarray:
    text = value.lstrip("#")
    if len(text) != 6:
        raise ValueError(f"Expected #RRGGBB color, got {value!r}")
    return np.asarray([int(text[i : i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float32)
