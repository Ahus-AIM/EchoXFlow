"""Task 4: ECG-conditioned frame-to-video diffusion.

Training does not use the shared ``tasks.train`` loop (it needs EMA weights and a diffusion objective); run
``python -m tasks.video_diffusion.train`` instead. ``TASK_SPEC`` registers the task and exposes the training loss.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor

from tasks.registry import TaskSpec
from tasks.utils.training.module import LossFn
from tasks.video_diffusion.model import (
    CosineDiffusionSchedule,
    physical_coordinate_context,
    pixel_pack_video,
    pixel_unpack_video,
)


@dataclass(frozen=True)
class PreparedBatch:
    """A clip in diffusion space (pixel-packed) and in image space (normalized to ``[-1, 1]``)."""

    first_frame: Tensor  # [B, C, H/f, W/f]
    clean_future: Tensor  # [B, C, T-1, H/f, W/f]
    first_frame_image: Tensor  # [B, 1, H, W]
    clean_future_image: Tensor  # [B, 1, T-1, H, W]
    coordinate_context: Tensor  # [B, 2, H/f, W/f]
    ecg_waveform: Tensor  # [B, T, 2, W]
    pixel_pack_factor: int

    @property
    def image_size(self) -> tuple[int, int]:
        return (int(self.first_frame_image.shape[-2]), int(self.first_frame_image.shape[-1]))


def prepare_batch(batch: Mapping[str, Any], *, device: torch.device, config: Any) -> PreparedBatch:
    """Move a collated dataset batch to ``device`` and build the diffusion inputs."""
    factor = int(config["model"].get("pixel_pack_factor", 2))
    frames = batch["frames"].to(device, non_blocking=True).float()  # [B, T, 1, H, W] in [0, 1]
    normalized = frames.mul(2.0).sub(1.0)
    packed = pixel_pack_video(normalized, factor)
    context = physical_coordinate_context(
        coordinate_space=str(config["data"].get("coordinate_space", "beamspace")),
        shape=(int(packed.shape[-2]), int(packed.shape[-1])),
        depth_start_cm=batch["depth_start_cm"].to(device),
        depth_end_cm=batch["depth_end_cm"].to(device),
        sector_width_rad=batch["sector_width_rad"].to(device),
    )
    return PreparedBatch(
        first_frame=packed[:, :, 0],
        clean_future=packed[:, :, 1:],
        first_frame_image=normalized[:, 0],
        clean_future_image=normalized[:, 1:].permute(0, 2, 1, 3, 4),
        coordinate_context=context,
        ecg_waveform=batch["ecg_waveform"].to(device, non_blocking=True).float(),
        pixel_pack_factor=factor,
    )


@lru_cache(maxsize=4)
def _schedule(steps: int, logsnr_shift: float = 0.0) -> CosineDiffusionSchedule:
    return CosineDiffusionSchedule(steps, logsnr_shift=logsnr_shift)


def diffusion_schedule(config: Any, device: torch.device) -> CosineDiffusionSchedule:
    diffusion = config.get("diffusion", {})
    return _schedule(int(diffusion.get("steps", 1000)), float(diffusion.get("logsnr_shift", 0.0))).to(device)


def diffusion_loss_terms(
    model: torch.nn.Module,
    batch: Mapping[str, Any] | PreparedBatch,
    config: Any,
    *,
    timesteps: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, Tensor]:
    """v-prediction MSE, with the image-space L1 of the predicted clean clip as a diagnostic."""
    device = next(model.parameters()).device
    prepared = batch if isinstance(batch, PreparedBatch) else prepare_batch(batch, device=device, config=config)
    schedule = diffusion_schedule(config, device)
    clean = prepared.clean_future
    if timesteps is None:
        timesteps = torch.randint(schedule.steps, (len(clean),), device=device, generator=generator)
    noise = torch.randn(clean.shape, dtype=clean.dtype, device=device, generator=generator)
    noisy = schedule.add_noise(clean, timesteps, noise)
    target = schedule.velocity_target(clean, timesteps, noise)
    use_autocast = device.type == "cuda" and str(config["trainer"].get("precision", "bf16-mixed")).startswith("bf16")
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_autocast):
        velocity = model(
            noisy,
            timesteps,
            prepared.first_frame,
            prepared.coordinate_context,
            prepared.ecg_waveform,
        )
    velocity = velocity.float()
    velocity_mse = F.mse_loss(velocity, target)
    predicted_image = pixel_unpack_video(
        schedule.predict_clean(noisy, timesteps, velocity), prepared.pixel_pack_factor, output_size=prepared.image_size
    )
    return {
        "loss": velocity_mse,
        "velocity_mse": velocity_mse,
        "x0_l1": F.l1_loss(predicted_image, prepared.clean_future_image),
    }


TASK_SPEC = TaskSpec(name="video_diffusion", loss_fn=cast(LossFn, diffusion_loss_terms))
