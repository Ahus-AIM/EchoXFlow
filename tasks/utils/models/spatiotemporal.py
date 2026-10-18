"""Spatiotemporal 3D dense backbones that mix information across frames.

Every class maps ``[B,C,T,H,W]`` to ``[B,F,T,H,W]``. Video Swin-T and MViT-v2-S reuse the torchvision encoders;
UniFormer-S and the U-Mamba-Bot-style network are implemented here without extra dependencies. All start from
random initialization unless ``pretrained=True`` (where supported).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from tasks.utils.models.backbone import DenseVideoBackbone
from tasks.utils.models.unet import norm_groups as _norm_groups


class VideoSwinTinyBackbone(DenseVideoBackbone):
    """Video Swin-T encoder with a multiscale 3D feature-pyramid decoder."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        decoder_channels: int = 32,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        from torchvision.models.video import Swin3D_T_Weights, swin3d_t

        weights = Swin3D_T_Weights.KINETICS400_V1 if pretrained else None
        self.encoder = swin3d_t(weights=weights)
        self.encoder.patch_embed.proj = _adapt_input_conv3d(
            self.encoder.patch_embed.proj,
            in_channels=int(in_channels),
            preserve_weights=bool(pretrained),
        )
        self.encoder.norm = nn.Identity()
        self.encoder.avgpool = nn.Identity()
        self.encoder.head = nn.Identity()
        self.out_channels = int(out_channels)
        self.strides = ((2, 4, 4), (1, 2, 2), (1, 2, 2), (1, 2, 2))
        self.output_spatial_stride = (1, 1)
        feature_channels = (96, 192, 384, 768)
        hidden_channels = int(decoder_channels)
        self.stage_norms = nn.ModuleList(nn.LayerNorm(channels) for channels in feature_channels)
        self.lateral = nn.ModuleList(
            nn.Conv3d(channels, hidden_channels, kernel_size=1) for channels in feature_channels
        )
        self.smoothing = nn.ModuleList(
            nn.Sequential(
                nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(_norm_groups(hidden_channels), hidden_channels),
                nn.GELU(),
            )
            for _ in feature_channels[:-1]
        )
        self.output_projection = nn.Conv3d(hidden_channels, self.out_channels, kernel_size=1)

    def forward(self, video: Tensor) -> Tensor:
        if video.ndim != 5:
            raise ValueError(f"video must be [B,C,T,H,W], got {tuple(video.shape)}")
        output_size = tuple(int(value) for value in video.shape[2:])
        encoded = self.encoder.pos_drop(self.encoder.patch_embed(video))
        features: list[Tensor] = []
        stage_index = 0
        for layer_index, layer in enumerate(self.encoder.features):
            encoded = layer(encoded)
            if layer_index in {0, 2, 4, 6}:
                normalized = self.stage_norms[stage_index](encoded)
                features.append(normalized.permute(0, 4, 1, 2, 3).contiguous())
                stage_index += 1
        pyramid = self.lateral[-1](features[-1])
        for index in range(len(features) - 2, -1, -1):
            pyramid = F.interpolate(
                pyramid,
                size=tuple(int(value) for value in features[index].shape[2:]),
                mode="trilinear",
                align_corners=False,
            )
            pyramid = self.smoothing[index](pyramid + self.lateral[index](features[index]))
        prediction = self.output_projection(pyramid)
        return F.interpolate(prediction, size=output_size, mode="trilinear", align_corners=False)


class MViTV2SmallBackbone(DenseVideoBackbone):
    """MViTv2-S encoder adapted to dense, dynamically sized video prediction."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        decoder_channels: int = 32,
        pretrained: bool = False,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        from torchvision.models.video import MViT_V2_S_Weights, mvit_v2_s

        weights = MViT_V2_S_Weights.KINETICS400_V1 if pretrained else None
        self.encoder = mvit_v2_s(weights=weights)
        self.encoder.conv_proj = _adapt_input_conv3d(
            self.encoder.conv_proj,
            in_channels=int(in_channels),
            preserve_weights=bool(pretrained),
        )
        if self.encoder.pos_encoding.spatial_pos is not None:
            raise RuntimeError("MViTv2-S dense adaptation requires relative positional embeddings")
        self.encoder.head = nn.Identity()
        final_norm = self.encoder.norm
        self.encoder.norm = nn.Identity()

        self.out_channels = int(out_channels)
        self.strides = ((2, 4, 4), (1, 2, 2), (1, 2, 2), (1, 2, 2))
        self.output_spatial_stride = (1, 1)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.stage_indices = (0, 2, 13, 15)
        feature_channels = (96, 192, 384, 768)
        self.stage_norms = nn.ModuleList(
            (
                nn.LayerNorm(feature_channels[0], eps=1e-6),
                nn.LayerNorm(feature_channels[1], eps=1e-6),
                nn.LayerNorm(feature_channels[2], eps=1e-6),
                final_norm,
            )
        )
        self.decoder = _FeaturePyramidDecoder3d(
            feature_channels=feature_channels,
            hidden_channels=int(decoder_channels),
            out_channels=self.out_channels,
        )

    def forward(self, video: Tensor) -> Tensor:
        if video.ndim != 5:
            raise ValueError(f"video must be [B,C,T,H,W], got {tuple(video.shape)}")
        output_size = tuple(int(value) for value in video.shape[2:])
        embedded = self.encoder.conv_proj(video)
        thw = tuple(int(value) for value in embedded.shape[2:])
        encoded = self.encoder.pos_encoding(embedded.flatten(2).transpose(1, 2))

        features: list[Tensor] = []
        stage_index = 0
        for block_index, block in enumerate(self.encoder.blocks):
            if self.gradient_checkpointing and self.training and encoded.requires_grad:
                encoded, thw = checkpoint(block, encoded, thw, use_reentrant=False)
            else:
                encoded, thw = block(encoded, thw)
            thw = tuple(int(value) for value in thw)
            if block_index == self.stage_indices[stage_index]:
                normalized = self.stage_norms[stage_index](encoded)
                features.append(_mvit_tokens_to_video(normalized, thw=thw))
                stage_index += 1
                if stage_index == len(self.stage_indices):
                    break

        if len(features) != len(self.stage_indices):
            raise RuntimeError(f"MViTv2-S returned {len(features)} feature stages; expected {len(self.stage_indices)}")
        return cast(Tensor, self.decoder(features, output_size=output_size))


class UniFormerSmallBackbone(DenseVideoBackbone):
    """Canonical UniFormer-S video encoder with a dense 3D feature-pyramid decoder."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        embed_dims: Sequence[int] = (64, 128, 320, 512),
        depths: Sequence[int] = (3, 4, 8, 3),
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        decoder_channels: int = 32,
        gradient_checkpointing: bool = True,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        if pretrained:
            raise ValueError("pretrained weights are not available for the dense UniFormer-S adaptation")
        channels = tuple(int(value) for value in embed_dims)
        stage_depths = tuple(int(value) for value in depths)
        attention_head_dim = int(head_dim)
        if len(channels) != 4 or min(channels) <= 0:
            raise ValueError(f"embed_dims must contain four positive values, got {embed_dims}")
        if len(stage_depths) != 4 or min(stage_depths) <= 0:
            raise ValueError(f"depths must contain four positive values, got {depths}")
        if attention_head_dim <= 0 or any(channel % attention_head_dim != 0 for channel in channels):
            raise ValueError(f"each embed_dims value must be divisible by head_dim={head_dim}, got {embed_dims}")
        if float(mlp_ratio) <= 0.0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")

        self.out_channels = int(out_channels)
        self.strides = ((2, 4, 4), (1, 2, 2), (1, 2, 2), (1, 2, 2))
        self.output_spatial_stride = (1, 1)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.patch_embeddings = nn.ModuleList(
            (
                _UniFormerPatchEmbed(
                    in_channels=int(in_channels),
                    out_channels=channels[0],
                    kernel_size=(3, 4, 4),
                    stride=(2, 4, 4),
                    padding=(1, 0, 0),
                ),
                _UniFormerPatchEmbed(
                    in_channels=channels[0],
                    out_channels=channels[1],
                    kernel_size=(1, 2, 2),
                    stride=(1, 2, 2),
                ),
                _UniFormerPatchEmbed(
                    in_channels=channels[1],
                    out_channels=channels[2],
                    kernel_size=(1, 2, 2),
                    stride=(1, 2, 2),
                ),
                _UniFormerPatchEmbed(
                    in_channels=channels[2],
                    out_channels=channels[3],
                    kernel_size=(1, 2, 2),
                    stride=(1, 2, 2),
                ),
            )
        )
        drop_path_rates = torch.linspace(0.0, float(drop_path_rate), sum(stage_depths)).tolist()
        rate_index = 0
        stages: list[nn.Module] = []
        for stage_index, (stage_channels, stage_depth) in enumerate(zip(channels, stage_depths, strict=True)):
            blocks: list[nn.Module] = []
            for _ in range(stage_depth):
                block_kwargs: dict[str, Any] = {
                    "channels": stage_channels,
                    "mlp_ratio": float(mlp_ratio),
                    "dropout": float(dropout_rate),
                    "drop_path": float(drop_path_rates[rate_index]),
                }
                block: nn.Module
                if stage_index < 2:
                    block = _UniFormerLocalBlock(**block_kwargs)
                else:
                    block = _UniFormerGlobalBlock(
                        **block_kwargs,
                        num_heads=stage_channels // attention_head_dim,
                        attention_dropout=float(attention_dropout_rate),
                    )
                blocks.append(block)
                rate_index += 1
            stages.append(nn.ModuleList(blocks))
        self.stages = nn.ModuleList(stages)
        self.decoder = _FeaturePyramidDecoder3d(
            feature_channels=channels,
            hidden_channels=int(decoder_channels),
            out_channels=self.out_channels,
        )
        self.apply(_init_uniformer)

    def forward(self, video: Tensor) -> Tensor:
        if video.ndim != 5:
            raise ValueError(f"video must be [B,C,T,H,W], got {tuple(video.shape)}")
        output_size = tuple(int(value) for value in video.shape[2:])
        encoded = video
        features: list[Tensor] = []
        for patch_embedding, stage in zip(self.patch_embeddings, self.stages, strict=True):
            encoded = patch_embedding(encoded)
            for block in cast(nn.ModuleList, stage):
                if self.gradient_checkpointing and self.training and encoded.requires_grad:
                    encoded = checkpoint(block, encoded, use_reentrant=False)
                else:
                    encoded = block(encoded)
            features.append(encoded)
        return cast(Tensor, self.decoder(features, output_size=output_size))


class _UniFormerPatchEmbed(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int, int],
        stride: tuple[int, int, int],
        padding: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        super().__init__()
        self.projection = nn.Conv3d(
            int(in_channels),
            int(out_channels),
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = nn.LayerNorm(int(out_channels), eps=1e-6)

    def forward(self, video: Tensor) -> Tensor:
        embedded = self.projection(video)
        return cast(Tensor, self.norm(embedded.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3).contiguous())


class _UniFormerLocalBlock(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        mlp_ratio: float,
        dropout: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        from torchvision.ops import StochasticDepth

        hidden_channels = int(round(int(channels) * float(mlp_ratio)))
        self.position = nn.Conv3d(
            int(channels),
            int(channels),
            kernel_size=3,
            padding=1,
            groups=int(channels),
        )
        self.norm1 = nn.BatchNorm3d(int(channels))
        self.pointwise1 = nn.Conv3d(int(channels), int(channels), kernel_size=1)
        self.relation = nn.Conv3d(
            int(channels),
            int(channels),
            kernel_size=5,
            padding=2,
            groups=int(channels),
        )
        self.pointwise2 = nn.Conv3d(int(channels), int(channels), kernel_size=1)
        self.norm2 = nn.BatchNorm3d(int(channels))
        self.mlp = nn.Sequential(
            nn.Conv3d(int(channels), hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Conv3d(hidden_channels, int(channels), kernel_size=1),
            nn.Dropout(float(dropout)),
        )
        self.drop_path = StochasticDepth(float(drop_path), mode="row")

    def forward(self, video: Tensor) -> Tensor:
        video = video + self.position(video)
        relation = self.pointwise2(self.relation(self.pointwise1(self.norm1(video))))
        video = video + self.drop_path(relation)
        return cast(Tensor, video + self.drop_path(self.mlp(self.norm2(video))))


class _UniFormerGlobalBlock(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        attention_dropout: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        from torchvision.ops import StochasticDepth

        hidden_channels = int(round(int(channels) * float(mlp_ratio)))
        self.position = nn.Conv3d(
            int(channels),
            int(channels),
            kernel_size=3,
            padding=1,
            groups=int(channels),
        )
        self.norm1 = nn.LayerNorm(int(channels), eps=1e-6)
        self.attention = _MemoryEfficientAttention(
            channels=int(channels),
            num_heads=int(num_heads),
            attention_dropout=float(attention_dropout),
            projection_dropout=float(dropout),
        )
        self.norm2 = nn.LayerNorm(int(channels), eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(int(channels), hidden_channels),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_channels, int(channels)),
            nn.Dropout(float(dropout)),
        )
        self.drop_path = StochasticDepth(float(drop_path), mode="row")

    def forward(self, video: Tensor) -> Tensor:
        video = video + self.position(video)
        batch_size, channels, time_steps, height, width = (int(value) for value in video.shape)
        tokens = video.flatten(2).transpose(1, 2)
        tokens = tokens + self.drop_path(self.attention(self.norm1(tokens)))
        tokens = tokens + self.drop_path(self.mlp(self.norm2(tokens)))
        return tokens.transpose(1, 2).reshape(batch_size, channels, time_steps, height, width)


class _MemoryEfficientAttention(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        num_heads: int,
        attention_dropout: float,
        projection_dropout: float,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = int(channels) // self.num_heads
        self.attention_dropout = float(attention_dropout)
        self.qkv = nn.Linear(int(channels), 3 * int(channels), bias=True)
        self.projection = nn.Linear(int(channels), int(channels))
        self.projection_dropout = nn.Dropout(float(projection_dropout))

    def forward(self, tokens: Tensor) -> Tensor:
        batch_size, token_count, channels = (int(value) for value in tokens.shape)
        qkv = self.qkv(tokens).reshape(
            batch_size,
            token_count,
            3,
            self.num_heads,
            self.head_dim,
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch_size, token_count, channels)
        return cast(Tensor, self.projection_dropout(self.projection(attended)))


class _FeaturePyramidDecoder3d(nn.Module):
    def __init__(
        self,
        *,
        feature_channels: Sequence[int],
        hidden_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        channels = tuple(int(value) for value in feature_channels)
        if len(channels) < 2 or min(channels) <= 0:
            raise ValueError(f"feature_channels must contain at least two positive values, got {feature_channels}")
        hidden = int(hidden_channels)
        if hidden <= 0:
            raise ValueError(f"hidden_channels must be positive, got {hidden_channels}")
        self.lateral = nn.ModuleList(nn.Conv3d(stage_channels, hidden, kernel_size=1) for stage_channels in channels)
        self.smoothing = nn.ModuleList(
            nn.Sequential(
                nn.Conv3d(hidden, hidden, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(_norm_groups(hidden), hidden),
                nn.GELU(),
            )
            for _ in channels[:-1]
        )
        self.output_projection = nn.Conv3d(hidden, int(out_channels), kernel_size=1)

    def forward(self, features: Sequence[Tensor], *, output_size: tuple[int, int, int]) -> Tensor:
        if len(features) != len(self.lateral):
            raise ValueError(f"decoder received {len(features)} feature stages; expected {len(self.lateral)}")
        pyramid = self.lateral[-1](features[-1])
        for index in range(len(features) - 2, -1, -1):
            pyramid = F.interpolate(
                pyramid,
                size=tuple(int(value) for value in features[index].shape[2:]),
                mode="trilinear",
                align_corners=False,
            )
            pyramid = self.smoothing[index](pyramid + self.lateral[index](features[index]))
        prediction = self.output_projection(pyramid)
        return F.interpolate(prediction, size=output_size, mode="trilinear", align_corners=False)


def _mvit_tokens_to_video(tokens: Tensor, *, thw: Sequence[int]) -> Tensor:
    time_steps, height, width = (int(value) for value in thw)
    expected_tokens = time_steps * height * width
    patch_tokens = tokens[:, 1:]
    if int(patch_tokens.shape[1]) != expected_tokens:
        raise RuntimeError(
            f"MViTv2-S produced {patch_tokens.shape[1]} patch tokens for grid {(time_steps, height, width)}"
        )
    batch_size, _token_count, channels = (int(value) for value in patch_tokens.shape)
    return patch_tokens.reshape(batch_size, time_steps, height, width, channels).permute(0, 4, 1, 2, 3).contiguous()


class UMambaBotBackbone(DenseVideoBackbone):
    """Anisotropic 3D U-Mamba-Bot with a dependency-free selective SSM bottleneck."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        backbone_channels: Sequence[int] = (16, 32, 64, 128, 192, 256),
        backbone_kernel_sizes: Sequence[Sequence[int]] = (
            (1, 3, 3),
            (1, 3, 3),
            (3, 3, 3),
            (3, 3, 3),
            (3, 3, 3),
            (3, 3, 3),
        ),
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        channels = tuple(int(value) for value in backbone_channels)
        if len(channels) < 2 or min(channels) <= 0:
            raise ValueError(f"backbone_channels must contain at least two positive values, got {backbone_channels}")
        kernel_sizes = tuple(_as_3tuple(value) for value in backbone_kernel_sizes)
        if len(kernel_sizes) != len(channels):
            raise ValueError(
                "backbone_kernel_sizes must contain one entry per channel stage, "
                f"got {len(kernel_sizes)} kernels for {len(channels)} stages"
            )
        self.out_channels = int(out_channels)
        self.strides = ((1, 1, 1),) + tuple((1, 2, 2) for _ in channels[1:])
        self.output_spatial_stride = (1, 1)
        self.stem = _ResidualBlock3d(int(in_channels), channels[0], kernel_size=kernel_sizes[0])
        encoder_stages: list[nn.Module] = []
        current_channels = channels[0]
        for stage_channels, stride, kernel_size in zip(channels, self.strides, kernel_sizes, strict=True):
            encoder_stages.append(
                _ResidualBlock3d(
                    current_channels,
                    stage_channels,
                    stride=stride,
                    kernel_size=kernel_size,
                )
            )
            current_channels = stage_channels
        self.encoder_stages = nn.ModuleList(encoder_stages)
        self.mamba = _MambaLayer(channels[-1], d_state=int(d_state), d_conv=int(d_conv), expand=int(expand))
        self.upsample_projections = nn.ModuleList()
        self.decoder_stages = nn.ModuleList()
        current_channels = channels[-1]
        for skip_channels, kernel_size in zip(reversed(channels[:-1]), reversed(kernel_sizes[:-1]), strict=True):
            self.upsample_projections.append(nn.Conv3d(current_channels, skip_channels, kernel_size=1))
            self.decoder_stages.append(_ResidualBlock3d(2 * skip_channels, skip_channels, kernel_size=kernel_size))
            current_channels = skip_channels
        self.output_projection = nn.Conv3d(channels[0], self.out_channels, kernel_size=1)
        self.apply(_init_he)

    def forward(self, video: Tensor) -> Tensor:
        if video.ndim != 5:
            raise ValueError(f"video must be [B,C,T,H,W], got {tuple(video.shape)}")
        encoded = self.stem(video)
        skips: list[Tensor] = []
        for stage in self.encoder_stages:
            encoded = stage(encoded)
            skips.append(encoded)
        decoded = self.mamba(skips[-1])
        for projection, stage, skip in zip(
            self.upsample_projections,
            self.decoder_stages,
            reversed(skips[:-1]),
            strict=True,
        ):
            decoded = F.interpolate(
                decoded,
                size=tuple(int(value) for value in skip.shape[2:]),
                mode="nearest",
            )
            decoded = projection(decoded)
            decoded = stage(torch.cat((decoded, skip), dim=1))
        return cast(Tensor, self.output_projection(decoded))


class _ResidualBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: tuple[int, int, int] = (1, 1, 1),
        kernel_size: tuple[int, int, int] = (3, 3, 3),
    ) -> None:
        super().__init__()
        padding = (int(kernel_size[0]) // 2, int(kernel_size[1]) // 2, int(kernel_size[2]) // 2)
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.norm1 = nn.InstanceNorm3d(out_channels, eps=1e-5, affine=True)
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.norm2 = nn.InstanceNorm3d(out_channels, eps=1e-5, affine=True)
        self.activation = nn.LeakyReLU(inplace=True)
        self.residual = (
            nn.Identity()
            if in_channels == out_channels and stride == (1, 1, 1)
            else nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        )

    def forward(self, values: Tensor) -> Tensor:
        residual = self.residual(values)
        values = self.activation(self.norm1(self.conv1(values)))
        values = self.norm2(self.conv2(values))
        return cast(Tensor, self.activation(values + residual))


class _MambaLayer(nn.Module):
    def __init__(self, channels: int, *, d_state: int, d_conv: int, expand: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.norm = nn.LayerNorm(self.channels)
        self.mamba = _MambaMixer(self.channels, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, values: Tensor) -> Tensor:
        batch_size, channels = (int(value) for value in values.shape[:2])
        if channels != self.channels:
            raise ValueError(f"Mamba bottleneck expected {self.channels} channels, got {channels}")
        spatial_shape = tuple(int(value) for value in values.shape[2:])
        tokens = values.reshape(batch_size, channels, -1).transpose(1, 2)
        output_dtype = tokens.dtype
        with torch.autocast(device_type=values.device.type, enabled=False):
            tokens = self.mamba(self.norm(tokens.float()))
        tokens = tokens.to(dtype=output_dtype)
        return tokens.transpose(1, 2).reshape(batch_size, channels, *spatial_shape)


class _MambaMixer(nn.Module):
    """Reference Mamba-1 mixer with a pure PyTorch selective scan."""

    def __init__(self, d_model: int, *, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(expand) * self.d_model
        self.dt_rank = max(1, math.ceil(self.d_model / 16))
        self.in_projection = nn.Linear(self.d_model, 2 * self.d_inner, bias=False)
        self.depthwise_conv = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=int(d_conv),
            groups=self.d_inner,
            padding=int(d_conv) - 1,
            bias=True,
        )
        self.parameter_projection = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
        self.dt_projection = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.a_log = nn.Parameter(
            torch.log(torch.arange(1, self.d_state + 1, dtype=torch.float32)).unsqueeze(0).repeat(self.d_inner, 1)
        )
        self.skip = nn.Parameter(torch.ones(self.d_inner, dtype=torch.float32))
        self.out_projection = nn.Linear(self.d_inner, self.d_model, bias=False)
        self._initialize_delta()

    def _initialize_delta(self) -> None:
        scale = self.dt_rank**-0.5
        nn.init.uniform_(self.dt_projection.weight, -scale, scale)
        log_min, log_max = math.log(0.001), math.log(0.1)
        delta = torch.exp(torch.rand(self.d_inner) * (log_max - log_min) + log_min).clamp_min(1e-4)
        inverse_softplus = delta + torch.log(-torch.expm1(-delta))
        with torch.no_grad():
            self.dt_projection.bias.copy_(inverse_softplus)

    def forward(self, tokens: Tensor) -> Tensor:
        sequence_length = int(tokens.shape[1])
        values, gate = self.in_projection(tokens).chunk(2, dim=-1)
        values = self.depthwise_conv(values.transpose(1, 2))[..., :sequence_length].transpose(1, 2)
        values = F.silu(values)
        parameters = self.parameter_projection(values)
        raw_delta, input_state, output_state = torch.split(
            parameters,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )
        delta = F.softplus(self.dt_projection(raw_delta))
        continuous_a = -torch.exp(self.a_log).to(device=values.device, dtype=values.dtype)
        discrete_a = torch.exp(delta.unsqueeze(-1) * continuous_a.unsqueeze(0).unsqueeze(0))
        discrete_input = delta.unsqueeze(-1) * input_state.unsqueeze(2) * values.unsqueeze(-1)
        states = _selective_affine_scan(discrete_a, discrete_input)
        prediction = (states * output_state.unsqueeze(2)).sum(dim=-1)
        prediction = prediction + values * self.skip.to(device=values.device, dtype=values.dtype)
        return cast(Tensor, self.out_projection(prediction * F.silu(gate)))


def _selective_affine_scan(multiplier: Tensor, update: Tensor) -> Tensor:
    if multiplier.is_cuda:
        try:
            return cast(Tensor, _CudaAffineScan.apply(multiplier, update))
        except (ImportError, RuntimeError):
            pass
    state = torch.zeros_like(update[:, 0])
    outputs: list[Tensor] = []
    for index in range(int(update.shape[1])):
        state = multiplier[:, index] * state + update[:, index]
        outputs.append(state)
    return torch.stack(outputs, dim=1)


class _CudaAffineScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, multiplier: Tensor, update: Tensor) -> Tensor:
        from torch._higher_order_ops.associative_scan import associative_scan

        _combined_multiplier, states = associative_scan(
            _compose_affine,
            (multiplier, update),
            dim=1,
        )
        ctx.save_for_backward(multiplier, states)
        return cast(Tensor, states)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> tuple[Tensor, Tensor]:
        from torch._higher_order_ops.associative_scan import associative_scan

        (grad_states,) = grad_outputs
        multiplier, states = ctx.saved_tensors
        reversed_multiplier = torch.flip(multiplier, dims=(1,))
        reverse_coefficients = torch.cat(
            (torch.zeros_like(reversed_multiplier[:, :1]), reversed_multiplier[:, :-1]),
            dim=1,
        )
        reversed_grad = torch.flip(grad_states, dims=(1,))
        _combined_multiplier, reversed_adjoint = associative_scan(
            _compose_affine,
            (reverse_coefficients, reversed_grad),
            dim=1,
        )
        adjoint = torch.flip(reversed_adjoint, dims=(1,))
        previous_states = torch.cat((torch.zeros_like(states[:, :1]), states[:, :-1]), dim=1)
        return adjoint * previous_states, adjoint


def _compose_affine(
    left: tuple[Tensor, Tensor],
    right: tuple[Tensor, Tensor],
) -> tuple[Tensor, Tensor]:
    left_multiplier, left_update = left
    right_multiplier, right_update = right
    return (
        right_multiplier * left_multiplier,
        right_update + right_multiplier * left_update,
    )


def _adapt_input_conv3d(conv: nn.Conv3d, *, in_channels: int, preserve_weights: bool) -> nn.Conv3d:
    adapted = nn.Conv3d(
        int(in_channels),
        int(conv.out_channels),
        kernel_size=cast(tuple[int, int, int], conv.kernel_size),
        stride=cast(tuple[int, int, int], conv.stride),
        padding=cast(str | tuple[int, int, int], conv.padding),
        dilation=cast(tuple[int, int, int], conv.dilation),
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )
    if preserve_weights:
        with torch.no_grad():
            if int(in_channels) == 1:
                adapted.weight.copy_(conv.weight.sum(dim=1, keepdim=True))
            elif int(in_channels) == int(conv.in_channels):
                adapted.weight.copy_(conv.weight)
            else:
                mean_weight = conv.weight.mean(dim=1, keepdim=True)
                adapted.weight.copy_(mean_weight.repeat(1, int(in_channels), 1, 1, 1))
            if adapted.bias is not None and conv.bias is not None:
                adapted.bias.copy_(conv.bias)
    return adapted


def _init_he(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.kaiming_normal_(module.weight, a=1e-2)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _init_uniformer(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def _as_3tuple(value: Sequence[int]) -> tuple[int, int, int]:
    result = tuple(int(item) for item in value)
    if len(result) != 3 or min(result) <= 0:
        raise ValueError(f"expected three positive values, got {value}")
    return (result[0], result[1], result[2])


__all__ = ["MViTV2SmallBackbone", "UMambaBotBackbone", "UniFormerSmallBackbone", "VideoSwinTinyBackbone"]
