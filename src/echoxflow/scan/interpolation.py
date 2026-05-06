"""Small NumPy interpolation kernels for image and volume sampling."""

from __future__ import annotations

from typing import Literal

import numpy as np

Interpolation = Literal["nearest", "linear"]


def sample_image(image: np.ndarray, rows: np.ndarray, cols: np.ndarray, *, interpolation: Interpolation) -> np.ndarray:
    img = np.asarray(image)
    if img.ndim not in {2, 3}:
        raise ValueError(f"Expected image shape [H,W] or [H,W,C], got {img.shape}")
    if interpolation == "nearest":
        return _sample_image_nearest(img, rows, cols)
    if interpolation == "linear":
        return _sample_image_linear(img, rows, cols)
    raise ValueError(f"Unsupported interpolation: {interpolation}")


def sample_volume(volume: np.ndarray, zyx: np.ndarray, *, interpolation: Interpolation) -> np.ndarray:
    vol = np.asarray(volume)
    if vol.ndim not in {3, 4}:
        raise ValueError(f"Expected volume shape [Z,Y,X] or [Z,Y,X,C], got {vol.shape}")
    if interpolation == "nearest":
        return _sample_volume_nearest(vol, zyx)
    if interpolation == "linear":
        return _sample_volume_linear(vol, zyx)
    raise ValueError(f"Unsupported interpolation: {interpolation}")


def _sample_image_nearest(image: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    rr = np.clip(np.rint(rows).astype(np.int64), 0, image.shape[0] - 1)
    cc = np.clip(np.rint(cols).astype(np.int64), 0, image.shape[1] - 1)
    return np.asarray(image[rr, cc])


def _sample_image_linear(image: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    r0 = np.floor(rows).astype(np.int64)
    c0 = np.floor(cols).astype(np.int64)
    r1 = r0 + 1
    c1 = c0 + 1
    wr = rows - r0
    wc = cols - c0
    r0 = np.clip(r0, 0, image.shape[0] - 1)
    r1 = np.clip(r1, 0, image.shape[0] - 1)
    c0 = np.clip(c0, 0, image.shape[1] - 1)
    c1 = np.clip(c1, 0, image.shape[1] - 1)
    top = _blend(image[r0, c0], image[r0, c1], wc)
    bottom = _blend(image[r1, c0], image[r1, c1], wc)
    return _blend(top, bottom, wr)


def _sample_volume_nearest(volume: np.ndarray, zyx: np.ndarray) -> np.ndarray:
    coords = np.asarray(zyx, dtype=np.float64)
    zz = np.clip(np.rint(coords[..., 0]).astype(np.int64), 0, volume.shape[0] - 1)
    yy = np.clip(np.rint(coords[..., 1]).astype(np.int64), 0, volume.shape[1] - 1)
    xx = np.clip(np.rint(coords[..., 2]).astype(np.int64), 0, volume.shape[2] - 1)
    return np.asarray(volume[zz, yy, xx])


def _sample_volume_linear(volume: np.ndarray, zyx: np.ndarray) -> np.ndarray:
    coords = np.asarray(zyx, dtype=np.float64)
    z0 = np.floor(coords[..., 0]).astype(np.int64)
    y0 = np.floor(coords[..., 1]).astype(np.int64)
    x0 = np.floor(coords[..., 2]).astype(np.int64)
    z1 = z0 + 1
    y1 = y0 + 1
    x1 = x0 + 1
    wz = coords[..., 0] - z0
    wy = coords[..., 1] - y0
    wx = coords[..., 2] - x0
    z0 = np.clip(z0, 0, volume.shape[0] - 1)
    z1 = np.clip(z1, 0, volume.shape[0] - 1)
    y0 = np.clip(y0, 0, volume.shape[1] - 1)
    y1 = np.clip(y1, 0, volume.shape[1] - 1)
    x0 = np.clip(x0, 0, volume.shape[2] - 1)
    x1 = np.clip(x1, 0, volume.shape[2] - 1)
    c00 = _blend(volume[z0, y0, x0], volume[z0, y0, x1], wx)
    c01 = _blend(volume[z0, y1, x0], volume[z0, y1, x1], wx)
    c10 = _blend(volume[z1, y0, x0], volume[z1, y0, x1], wx)
    c11 = _blend(volume[z1, y1, x0], volume[z1, y1, x1], wx)
    c0 = _blend(c00, c01, wy)
    c1 = _blend(c10, c11, wy)
    return _blend(c0, c1, wz)


def _blend(a: np.ndarray, b: np.ndarray, weight: np.ndarray) -> np.ndarray:
    wa = np.asarray(weight, dtype=np.float32)
    while wa.ndim < np.asarray(a).ndim:
        wa = wa[..., None]
    blended = np.asarray(a, dtype=np.float32) * (1.0 - wa) + np.asarray(b, dtype=np.float32) * wa
    return np.asarray(blended)
