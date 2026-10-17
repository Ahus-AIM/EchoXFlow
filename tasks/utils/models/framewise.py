"""Framewise 2D dense backbones: one shared image network applied to every frame of a video.

Every class maps ``[B,C,T,H,W]`` to ``[B,F,T,H,W]`` by folding time into the batch, so none of them mixes
information across frames. Weights start from random initialization unless ``pretrained=True``.
"""

from __future__ import annotations

from typing import Any, cast

import torch.nn.functional as F
from torch import Tensor, nn

from tasks.utils.models.backbone import FramewiseDenseVideoBackbone
from tasks.utils.models.unet import norm_groups


class ConvNeXtV2AttoBackbone(FramewiseDenseVideoBackbone):
    """ConvNeXt V2 Atto encoder with the segmentation-models-pytorch feature-pyramid decoder."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        decoder_pyramid_channels: int = 64,
        decoder_segmentation_channels: int = 32,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        import segmentation_models_pytorch as smp

        self.out_channels = int(out_channels)
        self.strides = ((1, 32, 32),)
        self.output_spatial_stride = (1, 1)
        self.model = smp.FPN(
            encoder_name="tu-convnextv2_atto",
            encoder_weights="imagenet" if pretrained else None,
            decoder_pyramid_channels=int(decoder_pyramid_channels),
            decoder_segmentation_channels=int(decoder_segmentation_channels),
            decoder_dropout=0.0,
            in_channels=int(in_channels),
            classes=self.out_channels,
        )

    def forward_frames(self, frames: Tensor) -> Tensor:
        return cast(Tensor, self.model(frames))


class SegFormerB0Backbone(FramewiseDenseVideoBackbone):
    """SegFormer-B0: hierarchical MiT-B0 encoder with its native all-MLP decoder."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        decoder_channels: int = 256,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        import segmentation_models_pytorch as smp

        self.out_channels = int(out_channels)
        self.strides = ((1, 32, 32),)
        self.output_spatial_stride = (1, 1)
        self.model = smp.Segformer(
            encoder_name="mit_b0",
            encoder_weights="imagenet" if pretrained else None,
            decoder_segmentation_channels=int(decoder_channels),
            in_channels=int(in_channels),
            classes=self.out_channels,
        )

    def forward_frames(self, frames: Tensor) -> Tensor:
        return cast(Tensor, self.model(frames))


class TimmFeaturePyramidBackbone(FramewiseDenseVideoBackbone):
    """Any ``timm`` ``features_only`` encoder followed by a small dense feature-pyramid decoder.

    Used for the DINOv3 ViT-S/16 architecture and MambaOut-Femto.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        model_name: str,
        decoder_channels: int = 64,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        import timm

        self.out_channels = int(out_channels)
        self.output_spatial_stride = (1, 1)
        self.encoder = timm.create_model(
            str(model_name),
            pretrained=bool(pretrained),
            in_chans=int(in_channels),
            features_only=True,
        )
        feature_info: Any = self.encoder.feature_info
        feature_channels = tuple(int(value) for value in feature_info.channels())
        reductions = tuple(int(value) for value in feature_info.reduction())
        if not feature_channels or len(feature_channels) != len(reductions):
            raise ValueError(f"timm encoder {model_name!r} did not expose a valid feature hierarchy")
        self.feature_channels = feature_channels
        maximum_reduction = max(reductions)
        self.strides = ((1, maximum_reduction, maximum_reduction),)
        width = int(decoder_channels)
        self.lateral = nn.ModuleList(nn.Conv2d(channels, width, kernel_size=1) for channels in feature_channels)
        self.smoothing = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(norm_groups(width), width),
                nn.GELU(),
            )
            for _ in feature_channels[:-1]
        )
        self.output_projection = nn.Conv2d(width, self.out_channels, kernel_size=1)

    def forward_frames(self, frames: Tensor) -> Tensor:
        output_shape = tuple(int(value) for value in frames.shape[-2:])
        features = [
            _channels_first(feature, channels=channels)
            for feature, channels in zip(self.encoder(frames), self.feature_channels, strict=True)
        ]
        pyramid = self.lateral[-1](features[-1])
        for index in range(len(features) - 2, -1, -1):
            pyramid = F.interpolate(pyramid, size=features[index].shape[-2:], mode="bilinear", align_corners=False)
            pyramid = self.smoothing[index](pyramid + self.lateral[index](features[index]))
        prediction = self.output_projection(pyramid)
        return F.interpolate(prediction, size=output_shape, mode="bilinear", align_corners=False)


def _channels_first(feature: Tensor, *, channels: int) -> Tensor:
    if feature.ndim != 4:
        raise ValueError(f"timm feature must have four dimensions, got {tuple(feature.shape)}")
    if int(feature.shape[1]) == int(channels):
        return feature
    if int(feature.shape[-1]) == int(channels):
        return feature.permute(0, 3, 1, 2).contiguous()
    raise ValueError(f"timm feature shape {tuple(feature.shape)} does not contain the declared {channels} channels")


__all__ = ["ConvNeXtV2AttoBackbone", "SegFormerB0Backbone", "TimmFeaturePyramidBackbone"]
