"""ECG-conditioned frame-to-video diffusion model.

A factorized spatiotemporal 3D U-Net denoises the 31 future frames of a 32-frame clip. It is conditioned on the clean
first frame and per-pixel physical coordinates (concatenated at the input) and on per-frame ECG windows encoded by a
residual 1D CNN.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint


def normalization_groups(channels: int) -> int:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return groups


class SinusoidalDiffusionTimeEmbedding(nn.Module):
    frequencies: Tensor

    def __init__(self, channels: int) -> None:
        super().__init__()
        half = channels // 2
        frequencies = torch.exp(-math.log(10_000.0) * torch.arange(half, dtype=torch.float32) / max(half - 1, 1))
        self.register_buffer("frequencies", frequencies)

    def forward(self, timesteps: Tensor) -> Tensor:
        phases = timesteps.float().unsqueeze(-1) * self.frequencies
        return torch.cat((torch.sin(phases), torch.cos(phases)), dim=-1)


class ECGResidualBlock1d(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, *, kernel_size: int, stride: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.normalization_1 = nn.GroupNorm(normalization_groups(input_channels), input_channels)
        self.convolution_1 = nn.Conv1d(
            input_channels, output_channels, kernel_size, stride=stride, padding=padding, padding_mode="reflect"
        )
        self.normalization_2 = nn.GroupNorm(normalization_groups(output_channels), output_channels)
        self.convolution_2 = nn.Conv1d(
            output_channels, output_channels, kernel_size, padding=padding, padding_mode="reflect"
        )
        self.skip = (
            nn.Conv1d(input_channels, output_channels, 1, stride=stride)
            if input_channels != output_channels or stride != 1
            else nn.Identity()
        )

    def forward(self, features: Tensor) -> Tensor:
        residual = self.convolution_1(F.silu(self.normalization_1(features)))
        residual = self.convolution_2(F.silu(self.normalization_2(residual)))
        return cast(Tensor, self.skip(features) + residual)


class ECGWaveformEncoder(nn.Module):
    """Encode one fixed-rate ECG window ``[B, T, 1, W]`` into one embedding per frame."""

    def __init__(
        self,
        *,
        window_samples: int,
        channels: tuple[int, ...],
        kernel_size: int,
        pooled_samples: int,
        embedding_channels: int,
    ) -> None:
        super().__init__()
        self.window_samples = int(window_samples)
        self.input_convolution = nn.Conv1d(
            1, channels[0], kernel_size, padding=kernel_size // 2, padding_mode="reflect"
        )
        blocks: list[nn.Module] = []
        input_channels = channels[0]
        for output_channels in channels:
            blocks.append(ECGResidualBlock1d(input_channels, output_channels, kernel_size=kernel_size, stride=2))
            input_channels = output_channels
        self.blocks = nn.Sequential(*blocks)
        self.output_normalization = nn.GroupNorm(normalization_groups(channels[-1]), channels[-1])
        self.pooled_samples = int(pooled_samples)
        self.output_projection = nn.Sequential(
            nn.Linear(channels[-1] * pooled_samples, embedding_channels),
            nn.SiLU(),
            nn.Linear(embedding_channels, embedding_channels),
        )

    def forward(self, waveform: Tensor) -> Tensor:
        if waveform.ndim != 4 or waveform.shape[2:] != (1, self.window_samples):
            raise ValueError(f"ECG waveform must have shape [B, T, 1, {self.window_samples}], got {waveform.shape}")
        batch, frames = waveform.shape[:2]
        features = waveform.reshape(batch * frames, 1, self.window_samples)
        features = self.blocks(self.input_convolution(features))
        features = F.silu(self.output_normalization(features))
        features = F.adaptive_avg_pool1d(features, self.pooled_samples).flatten(1)
        return cast(Tensor, self.output_projection(features).reshape(batch, frames, -1))


class FactorizedSpatioTemporalConvolution(nn.Module):
    """Spatial ``1xkxk`` convolution followed by a temporal ``kx1x1`` convolution."""

    def __init__(
        self, input_channels: int, output_channels: int, *, spatial_kernel_size: int, temporal_kernel_size: int
    ) -> None:
        super().__init__()
        self.spatial = nn.Conv3d(
            input_channels,
            output_channels,
            (1, spatial_kernel_size, spatial_kernel_size),
            padding=(0, spatial_kernel_size // 2, spatial_kernel_size // 2),
            padding_mode="reflect",
        )
        self.temporal = nn.Conv3d(
            output_channels,
            output_channels,
            (temporal_kernel_size, 1, 1),
            padding=(temporal_kernel_size // 2, 0, 0),
            padding_mode="reflect",
        )

    def forward(self, features: Tensor) -> Tensor:
        return cast(Tensor, self.temporal(self.spatial(features)))


class VideoTimeResidualBlock(nn.Module):
    """Residual block whose per-frame embedding is added between the two factorized convolutions."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        embedding_channels: int,
        spatial_kernel_size: int,
        temporal_kernel_size: int,
    ) -> None:
        super().__init__()
        kernels = {"spatial_kernel_size": spatial_kernel_size, "temporal_kernel_size": temporal_kernel_size}
        self.normalization_1 = nn.GroupNorm(normalization_groups(input_channels), input_channels)
        self.convolution_1 = FactorizedSpatioTemporalConvolution(input_channels, output_channels, **kernels)
        self.embedding_projection = nn.Linear(embedding_channels, output_channels)
        self.normalization_2 = nn.GroupNorm(normalization_groups(output_channels), output_channels)
        self.convolution_2 = FactorizedSpatioTemporalConvolution(output_channels, output_channels, **kernels)
        self.skip = (
            nn.Conv3d(input_channels, output_channels, 1) if input_channels != output_channels else nn.Identity()
        )

    def forward(self, features: Tensor, frame_embedding: Tensor) -> Tensor:
        residual = self.convolution_1(F.silu(self.normalization_1(features)))
        residual = residual + self.embedding_projection(frame_embedding).permute(0, 2, 1)[..., None, None]
        residual = self.convolution_2(F.silu(self.normalization_2(residual)))
        return cast(Tensor, self.skip(features) + residual)


def spatial_interpolate(features: Tensor, size: tuple[int, int]) -> Tensor:
    batch, channels, frames = features.shape[:3]
    flattened = features.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, *features.shape[-2:])
    resized = F.interpolate(flattened, size=size, mode="bilinear", align_corners=False)
    return resized.reshape(batch, frames, channels, *size).permute(0, 2, 1, 3, 4)


class VideoDiffusionUNet(nn.Module):
    """Jointly denoise ``clip_frames - 1`` future frames given one clean first frame and per-frame ECG windows.

    Three encoder levels (``model_channels * channel_multipliers``), a bottleneck with 2x spatial downsampling and two
    residual blocks, and three decoder levels with skips. The clean first frame enters as input channels.
    """

    def __init__(
        self,
        *,
        clip_frames: int = 32,
        data_channels: int = 4,
        model_channels: int = 128,
        channel_multipliers: tuple[int, int, int] = (1, 2, 4),
        embedding_channels: int = 512,
        spatial_kernel_size: int = 3,
        temporal_kernel_size: int = 3,
        ecg_window_samples: int = 841,
        ecg_encoder_channels: tuple[int, ...] = (16, 32, 64),
        ecg_encoder_kernel_size: int = 15,
        ecg_encoder_pooled_samples: int = 8,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if len(channel_multipliers) != 3:
            raise ValueError("the video U-Net requires exactly three channel multipliers")
        self.clip_frames = int(clip_frames)
        self.data_channels = int(data_channels)
        self.activation_checkpointing = bool(activation_checkpointing)
        channels = (
            int(model_channels) * int(channel_multipliers[0]),
            int(model_channels) * int(channel_multipliers[1]),
            int(model_channels) * int(channel_multipliers[2]),
        )
        kernel = spatial_kernel_size
        padding = kernel // 2
        self.diffusion_time_embedding = nn.Sequential(
            SinusoidalDiffusionTimeEmbedding(model_channels),
            nn.Linear(model_channels, embedding_channels),
            nn.SiLU(),
            nn.Linear(embedding_channels, embedding_channels),
        )
        # Input: noisy future + repeated clean first frame + two physical-coordinate channels.
        self.input_convolution = nn.Conv3d(
            2 * self.data_channels + 2,
            channels[0],
            (1, kernel, kernel),
            padding=(0, padding, padding),
            padding_mode="reflect",
        )

        def residual(inputs: int, outputs: int) -> VideoTimeResidualBlock:
            return VideoTimeResidualBlock(
                inputs,
                outputs,
                embedding_channels=embedding_channels,
                spatial_kernel_size=kernel,
                temporal_kernel_size=temporal_kernel_size,
            )

        def spatial_convolution(inputs: int, outputs: int, *, stride: int = 1) -> nn.Conv3d:
            return nn.Conv3d(
                inputs,
                outputs,
                (1, kernel, kernel),
                stride=(1, stride, stride),
                padding=(0, padding, padding),
                padding_mode="reflect",
            )

        self.encoder_0 = residual(channels[0], channels[0])
        self.downsample_0 = spatial_convolution(channels[0], channels[1], stride=2)
        self.encoder_1 = residual(channels[1], channels[1])
        self.downsample_1 = spatial_convolution(channels[1], channels[2], stride=2)
        self.encoder_2 = residual(channels[2], channels[2])
        self.downsample_2 = spatial_convolution(channels[2], channels[2], stride=2)
        self.middle_0 = residual(channels[2], channels[2])
        self.middle_1 = residual(channels[2], channels[2])
        self.upsample_2 = spatial_convolution(channels[2], channels[2])
        self.decoder_2 = residual(channels[2] * 2, channels[2])
        self.upsample_1 = spatial_convolution(channels[2], channels[1])
        self.decoder_1 = residual(channels[1] * 2, channels[1])
        self.upsample_0 = spatial_convolution(channels[1], channels[0])
        self.decoder_0 = residual(channels[0] * 2, channels[0])
        self.output_normalization = nn.GroupNorm(normalization_groups(channels[0]), channels[0])
        self.output_convolution = spatial_convolution(channels[0], self.data_channels)
        nn.init.zeros_(self.output_convolution.weight)
        nn.init.zeros_(cast(Tensor, self.output_convolution.bias))
        self.ecg_waveform_encoder = ECGWaveformEncoder(
            window_samples=ecg_window_samples,
            channels=tuple(ecg_encoder_channels),
            kernel_size=ecg_encoder_kernel_size,
            pooled_samples=ecg_encoder_pooled_samples,
            embedding_channels=embedding_channels,
        )

    def _run(self, block: nn.Module, *inputs: Tensor) -> Tensor:
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            return cast(Tensor, activation_checkpoint(block, *inputs, use_reentrant=False))
        return cast(Tensor, block(*inputs))

    def forward(
        self,
        noisy_future: Tensor,
        diffusion_timesteps: Tensor,
        first_frame: Tensor,
        coordinate_context: Tensor,
        ecg_waveform: Tensor,
    ) -> Tensor:
        """Predict the v-target for ``noisy_future`` ``[B, C, T-1, H, W]``.

        ``first_frame`` is ``[B, C, H, W]``, ``coordinate_context`` ``[B, 2, H, W]`` and ``ecg_waveform``
        ``[B, T, 1, W]``; the first frame's ECG window is not used.
        """
        future_frames = noisy_future.shape[2]
        if future_frames != self.clip_frames - 1:
            raise ValueError(f"expected {self.clip_frames - 1} future frames, got {future_frames}")
        dtype = noisy_future.dtype
        diffusion_embedding = self.diffusion_time_embedding(diffusion_timesteps).unsqueeze(1)
        frame_embedding = diffusion_embedding + self.ecg_waveform_encoder(ecg_waveform[:, 1:].to(dtype))

        repeated_first_frame = first_frame.unsqueeze(2).expand(-1, -1, future_frames, -1, -1)
        repeated_coordinates = coordinate_context.unsqueeze(2).expand(-1, -1, future_frames, -1, -1)
        level_0 = self.input_convolution(torch.cat((noisy_future, repeated_first_frame, repeated_coordinates), dim=1))
        level_0 = self._run(self.encoder_0, level_0, frame_embedding)
        level_1 = self._run(self.encoder_1, self.downsample_0(level_0), frame_embedding)
        level_2 = self._run(self.encoder_2, self.downsample_1(level_1), frame_embedding)
        middle = self.downsample_2(level_2)
        middle = self._run(self.middle_0, middle, frame_embedding)
        middle = self._run(self.middle_1, middle, frame_embedding)
        decoded = self.upsample_2(F.interpolate(middle, size=level_2.shape[-3:], mode="trilinear", align_corners=False))
        decoded = self._run(self.decoder_2, torch.cat((decoded, level_2), dim=1), frame_embedding)
        decoded = self.upsample_1(spatial_interpolate(decoded, (int(level_1.shape[-2]), int(level_1.shape[-1]))))
        decoded = self._run(self.decoder_1, torch.cat((decoded, level_1), dim=1), frame_embedding)
        decoded = self.upsample_0(spatial_interpolate(decoded, (int(level_0.shape[-2]), int(level_0.shape[-1]))))
        decoded = self._run(self.decoder_0, torch.cat((decoded, level_0), dim=1), frame_embedding)
        return cast(Tensor, self.output_convolution(F.silu(self.output_normalization(decoded))))


def ecg_window_samples(ecg_config: Mapping[str, Any]) -> int:
    rate = float(ecg_config.get("sample_rate_hz", 600.0))
    history = float(ecg_config.get("history_seconds", 1.2))
    future = float(ecg_config.get("future_seconds", 0.2))
    return round(history * rate) + round(future * rate) + 1


def build_model(config: Mapping[str, Any], *, activation_checkpointing: bool | None = None) -> VideoDiffusionUNet:
    """Build the U-Net from a ``train.yaml``-style mapping (``model``, ``data`` and ``ecg`` sections)."""
    model = dict(config["model"])
    pack = int(model.pop("pixel_pack_factor", 2))
    model.pop("kind", None)
    for key in ("channel_multipliers", "ecg_encoder_channels"):
        if key in model:
            model[key] = tuple(int(value) for value in model[key])
    if activation_checkpointing is not None:
        model["activation_checkpointing"] = activation_checkpointing
    return VideoDiffusionUNet(
        clip_frames=int(config.get("data", {}).get("clip_frames", 32)),
        data_channels=pack * pack,
        ecg_window_samples=ecg_window_samples(config.get("ecg", {})),
        **model,
    )


class CosineDiffusionSchedule(nn.Module):
    """Discrete cosine noise schedule (Nichol & Dhariwal) with v-prediction helpers.

    ``logsnr_shift`` adds a constant to the log signal-to-noise ratio at every step (the shifted cosine schedule of
    Hoogeboom et al., 2023). A negative shift puts more of training and sampling at high noise, where the large-scale
    structure of a sample is decided; at higher resolutions a given noise level destroys less of that structure.
    """

    alpha_bar: Tensor

    def __init__(self, steps: int = 1000, *, offset: float = 0.008, logsnr_shift: float = 0.0) -> None:
        super().__init__()
        positions = torch.linspace(0.0, 1.0, steps + 1, dtype=torch.float64)
        alpha_bar = torch.cos((positions + offset) / (1.0 + offset) * math.pi / 2.0).square()
        alpha_bar = alpha_bar / alpha_bar[0]
        if logsnr_shift:
            interior = alpha_bar.clamp(1.0e-12, 1.0 - 1.0e-12)
            shifted = torch.sigmoid(torch.log(interior / (1.0 - interior)) + float(logsnr_shift))
            alpha_bar = torch.where((alpha_bar > 0.0) & (alpha_bar < 1.0), shifted, alpha_bar)
        betas = (1.0 - alpha_bar[1:] / alpha_bar[:-1]).clamp(1.0e-8, 0.999)
        self.register_buffer("alpha_bar", torch.cumprod(1.0 - betas, dim=0).float())

    @property
    def steps(self) -> int:
        return int(self.alpha_bar.numel())

    def coefficients(self, timesteps: Tensor, reference: Tensor) -> tuple[Tensor, Tensor]:
        alpha_bar = self.alpha_bar[timesteps].to(dtype=reference.dtype)
        shape = (-1, *(1 for _ in range(reference.ndim - 1)))
        return alpha_bar.sqrt().view(shape), (1.0 - alpha_bar).sqrt().view(shape)

    def add_noise(self, clean: Tensor, timesteps: Tensor, noise: Tensor) -> Tensor:
        alpha, sigma = self.coefficients(timesteps, clean)
        return alpha * clean + sigma * noise

    def velocity_target(self, clean: Tensor, timesteps: Tensor, noise: Tensor) -> Tensor:
        alpha, sigma = self.coefficients(timesteps, clean)
        return alpha * noise - sigma * clean

    def predict_clean(self, noisy: Tensor, timesteps: Tensor, velocity: Tensor) -> Tensor:
        alpha, sigma = self.coefficients(timesteps, noisy)
        return alpha * noisy - sigma * velocity

    def predict_noise(self, noisy: Tensor, timesteps: Tensor, velocity: Tensor) -> Tensor:
        alpha, sigma = self.coefficients(timesteps, noisy)
        return sigma * noisy + alpha * velocity


class ExponentialMovingAverage:
    """EMA of model weights."""

    def __init__(self, model: nn.Module, *, decay: float) -> None:
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = float(decay)
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        blend = 1.0 - self.decay
        source = dict(model.named_parameters())
        for name, parameter in self.model.named_parameters():
            parameter.lerp_(source[name], blend)
        buffers = dict(model.named_buffers())
        for name, buffer in self.model.named_buffers():
            buffer.copy_(buffers[name])


def pixel_pack_video(frames: Tensor, factor: int) -> Tensor:
    """Losslessly rearrange ``[B, T, C, H, W]`` into ``[B, C*f*f, T, H/f, W/f]`` (replicate-padding odd sizes)."""
    batch, frame_count, channels, height, width = frames.shape
    padded_height, padded_width = math.ceil(height / factor) * factor, math.ceil(width / factor) * factor
    flattened = frames.reshape(batch * frame_count, channels, height, width)
    if (padded_height, padded_width) != (height, width):
        flattened = F.pad(flattened, (0, padded_width - width, 0, padded_height - height), mode="replicate")
    packed = F.pixel_unshuffle(flattened, factor)
    return packed.reshape(batch, frame_count, -1, padded_height // factor, padded_width // factor).permute(
        0, 2, 1, 3, 4
    )


def pixel_unpack_video(packed: Tensor, factor: int, *, output_size: tuple[int, int]) -> Tensor:
    """Invert :func:`pixel_pack_video` to ``[B, C, T, H, W]`` and crop the padding."""
    batch, packed_channels, frame_count, height, width = packed.shape
    flattened = packed.permute(0, 2, 1, 3, 4).reshape(batch * frame_count, packed_channels, height, width)
    unpacked = F.pixel_shuffle(flattened, factor)[..., : output_size[0], : output_size[1]]
    return unpacked.reshape(batch, frame_count, -1, *output_size).permute(0, 2, 1, 3, 4)


def physical_coordinate_context(
    *,
    coordinate_space: str,
    shape: tuple[int, int],
    depth_start_cm: Tensor,
    depth_end_cm: Tensor,
    sector_width_rad: Tensor,
    depth_scale_cm: float = 20.0,
) -> Tensor:
    """Per-pixel physical coordinates ``[B, 2, H, W]`` at pixel centres.

    Beamspace: (depth / 20 cm, beam angle in rad). Cartesian: (axial / 20 cm, lateral / 20 cm) on the square canvas
    spanning ``[0, depth_end]`` axially and ``[-depth_end/2, depth_end/2]`` laterally.
    """
    height, width = shape
    device, dtype = depth_end_cm.device, torch.float32
    batch = depth_end_cm.numel()
    rows = ((torch.arange(height, device=device, dtype=dtype) + 0.5) / height).view(1, height, 1)
    cols = ((torch.arange(width, device=device, dtype=dtype) + 0.5) / width - 0.5).view(1, 1, width)
    depth_end = depth_end_cm.to(dtype).view(batch, 1, 1)
    if coordinate_space == "beamspace":
        depth_start = depth_start_cm.to(dtype).view(batch, 1, 1)
        first = (depth_start + rows * (depth_end - depth_start)) / depth_scale_cm
        second = cols * sector_width_rad.to(dtype).view(batch, 1, 1)
    elif coordinate_space == "cartesian":
        first = rows * depth_end / depth_scale_cm
        second = cols * depth_end / depth_scale_cm
    else:
        raise ValueError(f"unsupported coordinate space {coordinate_space!r}")
    return torch.stack((first.expand(batch, height, width), second.expand(batch, height, width)), dim=1)


@torch.no_grad()
def sample_future_clip(
    model: VideoDiffusionUNet,
    schedule: CosineDiffusionSchedule,
    first_frame: Tensor,
    coordinate_context: Tensor,
    ecg_waveform: Tensor,
    *,
    sampling_steps: int = 32,
    generator: torch.Generator | None = None,
    initial_noise: Tensor | None = None,
    use_autocast: bool = True,
) -> Tensor:
    """Deterministic DDIM sampling of the packed future clip ``[B, C, T-1, H, W]`` from pure noise."""
    shape = (len(first_frame), model.data_channels, model.clip_frames - 1, *first_frame.shape[-2:])
    if initial_noise is None:
        current = torch.randn(shape, dtype=first_frame.dtype, device=first_frame.device, generator=generator)
    else:
        current = initial_noise.to(device=first_frame.device, dtype=first_frame.dtype).reshape(shape).clone()
    timesteps = torch.linspace(schedule.steps - 1, 0, sampling_steps, device=first_frame.device).round().long()
    timesteps = torch.unique_consecutive(timesteps)
    for index, timestep in enumerate(timesteps.tolist()):
        batch_timesteps = torch.full((len(first_frame),), timestep, dtype=torch.long, device=first_frame.device)
        with torch.autocast(device_type=first_frame.device.type, dtype=torch.bfloat16, enabled=use_autocast):
            velocity = model(current, batch_timesteps, first_frame, coordinate_context, ecg_waveform)
        velocity = velocity.float()
        predicted_clean = schedule.predict_clean(current, batch_timesteps, velocity)
        if index + 1 == len(timesteps):
            return predicted_clean
        predicted_noise = schedule.predict_noise(current, batch_timesteps, velocity)
        next_timesteps = torch.full_like(batch_timesteps, int(timesteps[index + 1]))
        alpha, sigma = schedule.coefficients(next_timesteps, current)
        current = alpha * predicted_clean + sigma * predicted_noise
    return current
