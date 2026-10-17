from __future__ import annotations

import math

import torch

from tasks.video_diffusion.model import CosineDiffusionSchedule
from tasks.video_diffusion.task import diffusion_schedule


def _logsnr(alpha_bar: torch.Tensor) -> torch.Tensor:
    return torch.log(alpha_bar / (1.0 - alpha_bar))


def test_zero_shift_is_the_plain_cosine_schedule() -> None:
    torch.testing.assert_close(
        CosineDiffusionSchedule(1000).alpha_bar, CosineDiffusionSchedule(1000, logsnr_shift=0.0).alpha_bar
    )
    default = diffusion_schedule({"diffusion": {"steps": 1000}}, torch.device("cpu"))
    torch.testing.assert_close(default.alpha_bar, CosineDiffusionSchedule(1000).alpha_bar)


def test_negative_shift_moves_every_step_towards_noise_by_the_shift() -> None:
    plain = CosineDiffusionSchedule(1000).alpha_bar.double()
    shift = -2.0 * math.log(2.0)
    shifted = diffusion_schedule({"diffusion": {"steps": 1000, "logsnr_shift": shift}}, torch.device("cpu"))
    shifted_bar = shifted.alpha_bar.double()
    assert torch.all(shifted_bar[1:-1] < plain[1:-1])
    middle = slice(100, 900)  # away from the clamped ends, the log-SNR moves by exactly the shift
    torch.testing.assert_close(
        _logsnr(shifted_bar[middle]) - _logsnr(plain[middle]),
        torch.full_like(plain[middle], shift),
        atol=2e-3,
        rtol=0.0,
    )
