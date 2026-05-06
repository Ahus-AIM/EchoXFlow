from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor


def write_input_target_debug_pngs(sample: object, *, output_dir: Path, split: str, epoch: int, step: int) -> None:
    """Write first-batch input and target tensors as frame-grid PNGs."""
    frames = _sample_tensor(sample, "frames")
    targets = _sample_tensor(sample, "target_masks")
    if frames is None or targets is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_tensor_channels(
        frames,
        output_dir=output_dir,
        stem=f"{split}_epoch{epoch:03d}_step{step:04d}_input",
        label="input frames",
        max_channels=frames.shape[1],
    )
    _write_tensor_channels(
        targets,
        output_dir=output_dir,
        stem=f"{split}_epoch{epoch:03d}_step{step:04d}_target",
        label="target masks",
        max_channels=targets.shape[1],
    )
    valid = _sample_tensor(sample, "target_mask_valid")
    if valid is not None:
        _write_tensor_channels(
            valid,
            output_dir=output_dir,
            stem=f"{split}_epoch{epoch:03d}_step{step:04d}_target_valid",
            label="target mask valid",
            max_channels=valid.shape[1],
        )


def _sample_tensor(sample: object, name: str) -> Tensor | None:
    candidate = cast(object, getattr(sample, name, None))
    if candidate is None or not isinstance(candidate, Tensor):
        return None
    if candidate.ndim != 5:
        return None
    return candidate.detach().to(device="cpu", dtype=torch.float32)


def _write_tensor_channels(
    tensor: Tensor,
    *,
    output_dir: Path,
    stem: str,
    label: str,
    max_channels: int,
) -> None:
    batch_index = 0
    channel_count = min(int(tensor.shape[1]), int(max_channels))
    for channel_index in range(channel_count):
        values = tensor[batch_index, channel_index].numpy()
        _write_frame_grid(
            values,
            output=output_dir / f"{stem}_b{batch_index:02d}_c{channel_index:02d}.png",
            title=(
                f"{label} b={batch_index} c={channel_index} " f"dims=[B,C,T,H,W]={tuple(int(v) for v in tensor.shape)}"
            ),
        )


def _write_frame_grid(values: Any, *, output: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    frame_count = int(values.shape[0])
    cols = min(4, max(1, frame_count))
    rows = int(math.ceil(frame_count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols + 0.8, 3.0 * rows + 0.6), squeeze=False)
    finite = values[np.isfinite(values)]
    if finite.size:
        vmin = float(finite.min())
        vmax = float(finite.max())
    else:
        vmin, vmax = 0.0, 1.0
    if vmin == vmax:
        vmax = vmin + 1.0
    image = None
    for frame_index, axis in enumerate(axes.flat):
        axis.set_xticks([])
        axis.set_yticks([])
        if frame_index >= frame_count:
            axis.axis("off")
            continue
        image = axis.imshow(values[frame_index], cmap="viridis", vmin=vmin, vmax=vmax)
        axis.set_title(f"T={frame_index} HxW={values.shape[1]}x{values.shape[2]}", fontsize=8)
    fig.suptitle(title, fontsize=10)
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.75, pad=0.02)
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)
