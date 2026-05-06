from __future__ import annotations

from typing import cast

import numpy as np

from echoxflow.colors import categorical_palette


def blend_segmentation_rgb(
    *,
    foreground_video: np.ndarray,
    bmode_video: np.ndarray,
    output_shape: tuple[int, int],
    colors: np.ndarray | None = None,
    alpha_scale: float = 0.78,
) -> np.ndarray:
    weights = _resize_channel_video(foreground_video, output_shape)
    if weights.ndim != 4:
        raise ValueError(f"segmentation foreground video must be [T,C,H,W], got {weights.shape}")
    palette = _palette(colors, channel_count=int(weights.shape[1]))
    weights = np.clip(_finite_array(weights), 0.0, 1.0)
    weight_sum = np.sum(weights, axis=1)
    blended_color = np.einsum("tchw,cr->thwr", weights, palette, optimize=True)
    normalized_color = blended_color / np.maximum(weight_sum[..., None], 1e-6)
    alpha = np.clip(weight_sum, 0.0, 1.0) * float(alpha_scale)
    bmode = normalized_bmode_rgb(bmode_video, frame_count=int(weights.shape[0]), output_shape=output_shape)
    return cast(np.ndarray, (bmode * (1.0 - alpha[..., None]) + normalized_color * alpha[..., None]).astype(np.float32))


def normalized_bmode_rgb(
    bmode_video: np.ndarray,
    *,
    frame_count: int,
    output_shape: tuple[int, int],
) -> np.ndarray:
    values = np.asarray(bmode_video)
    if values.ndim == 4 and values.shape[-1] in {3, 4}:
        values = values[..., :3].astype(np.float32)
        if values.shape[0] != frame_count:
            values = values[:frame_count]
        if tuple(values.shape[-3:-1]) != tuple(output_shape):
            values = _resize_rgb_video(values, output_shape)
        if values.size and float(np.nanmax(values)) > 1.5:
            values = values / 255.0
        return cast(np.ndarray, np.clip(_finite_array(values), 0.0, 1.0))
    if values.ndim != 3:
        values = np.asarray(values).reshape((-1, *values.shape[-2:]))
    values = values[:frame_count].astype(np.float32, copy=False)
    if values.size and float(np.nanmax(values)) > 1.5:
        values = values / 255.0
    gray = _resize_video(values, output_shape)
    lo = np.nanmin(gray, axis=(1, 2), keepdims=True)
    hi = np.nanmax(gray, axis=(1, 2), keepdims=True)
    gray = (gray - lo) / np.maximum(hi - lo, 1e-6)
    return cast(np.ndarray, np.repeat(np.clip(_finite_array(gray), 0.0, 1.0)[..., None], 3, axis=-1).astype(np.float32))


def _palette(colors: np.ndarray | None, *, channel_count: int) -> np.ndarray:
    palette = categorical_palette(max_colors=channel_count) if colors is None else np.asarray(colors, dtype=np.float32)
    palette = palette[:channel_count]
    if palette.shape[0] < channel_count:
        repeats = int(np.ceil(channel_count / max(1, palette.shape[0])))
        palette = np.tile(palette, (repeats, 1))[:channel_count]
    return palette


def _resize_video(video: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    import torch

    values = _finite_array(video)
    if tuple(values.shape[-2:]) == tuple(output_shape):
        return values
    tensor = torch.from_numpy(values[:, None])
    resized = torch.nn.functional.interpolate(tensor, size=output_shape, mode="bilinear", align_corners=False)
    return np.asarray(resized[:, 0].numpy(), dtype=np.float32)


def _resize_channel_video(video: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    import torch

    values = _finite_array(video)
    if values.ndim != 4:
        raise ValueError(f"channel video must be [T,C,H,W], got {values.shape}")
    if tuple(values.shape[-2:]) == tuple(output_shape):
        return values
    tensor = torch.from_numpy(values)
    resized = torch.nn.functional.interpolate(tensor, size=output_shape, mode="bilinear", align_corners=False)
    return np.asarray(resized.numpy(), dtype=np.float32)


def _resize_rgb_video(video: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    import torch

    values = _finite_array(video)
    if values.ndim != 4 or values.shape[-1] not in {3, 4}:
        raise ValueError(f"RGB video must be [T,H,W,3|4], got {values.shape}")
    if tuple(values.shape[-3:-1]) == tuple(output_shape):
        return values
    tensor = torch.from_numpy(values[..., :3].transpose(0, 3, 1, 2))
    resized = torch.nn.functional.interpolate(tensor, size=output_shape, mode="bilinear", align_corners=False)
    return np.asarray(resized.numpy().transpose(0, 2, 3, 1), dtype=np.float32)


def _finite_array(value: np.ndarray) -> np.ndarray:
    return np.asarray(np.where(np.isfinite(value), value, 0.0), dtype=np.float32)
