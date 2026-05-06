from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from importlib import import_module
from typing import Any, cast

import torch
import torch.nn.functional as F
from monai.networks.layers.simplelayers import SkipConnection
from torch import Tensor, nn

Int3 = tuple[int, int, int]
Spec3 = int | Sequence[int]


class TaskHead(nn.Module):
    """Post-processes raw network output for one task."""

    requires_conditioning = False
    extra_kwarg: str | None = None

    @property
    def expected_out_channels(self) -> int:
        raise NotImplementedError

    def forward(
        self,
        prediction: Tensor,
        *,
        frames: Tensor,
        conditioning: Tensor | None = None,
        **extras: object,
    ) -> Tensor:
        raise NotImplementedError


class ColorDopplerHead(TaskHead):
    requires_conditioning = False
    extra_kwarg = "conditioning"

    def __init__(
        self,
        *,
        doppler_channels: int = 3,
        target_spatial_shape: tuple[int, int] | None = None,
        temporal_upsample_factor: int = 2,
    ) -> None:
        super().__init__()
        self.doppler_channels = int(doppler_channels)
        self.target_spatial_shape = (
            None if target_spatial_shape is None else as_2tuple(target_spatial_shape, name="target_spatial_shape")
        )
        self.temporal_upsample_factor = int(temporal_upsample_factor)

    @property
    def expected_out_channels(self) -> int:
        return self.doppler_channels

    def forward(
        self,
        prediction: Tensor,
        *,
        frames: Tensor,
        conditioning: Tensor | None = None,
        **_: object,
    ) -> Tensor:
        if conditioning is None:
            raise ValueError("ColorDopplerHead requires sample conditioning with Nyquist metadata")
        time_steps = int(frames.shape[2])
        target_time = max(1, (time_steps - 1) * self.temporal_upsample_factor)
        interpolation_time = max(target_time, time_steps * self.temporal_upsample_factor)
        target_hw = self.target_spatial_shape or (int(frames.shape[3]), int(frames.shape[4]))
        prediction = F.interpolate(
            prediction,
            size=(interpolation_time, *target_hw),
            mode="trilinear",
            align_corners=False,
        )
        trim = max(0, int(prediction.shape[2]) - target_time)
        trim_left = trim // 2
        prediction = prediction[:, :, trim_left : trim_left + target_time]
        if int(prediction.shape[2]) != target_time:
            raise ValueError(
                f"Color Doppler temporal interpolation produced {prediction.shape[2]} frames, expected {target_time}"
            )
        velocity_raw = prediction[:, 0:1]
        power_raw = prediction[:, 1:2]
        velocity_std_raw = prediction[:, 2:3]
        nyquist = conditioning.to(device=prediction.device, dtype=prediction.dtype)[:, 1].clamp_min(1e-6)
        velocity_mps = velocity_raw * nyquist.reshape(-1, 1, 1, 1, 1)
        return torch.cat([velocity_mps, power_raw, velocity_std_raw, prediction[:, 3:]], dim=1)


class TissueDopplerHead(TaskHead):
    requires_conditioning = False
    extra_kwarg = "velocity_scale_mps_per_px_frame"

    def __init__(
        self,
        *,
        doppler_channels: int = 1,
        marker_channels: int = 0,
        target_spatial_shape: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.doppler_channels = int(doppler_channels)
        self.marker_channels = int(marker_channels)
        self.target_spatial_shape = (
            None if target_spatial_shape is None else as_2tuple(target_spatial_shape, name="target_spatial_shape")
        )

    @property
    def expected_out_channels(self) -> int:
        return self.doppler_channels + self.marker_channels

    def forward(
        self,
        prediction: Tensor,
        *,
        frames: Tensor,
        conditioning: Tensor | None = None,
        velocity_scale_mps_per_px_frame: object = None,
        **_: object,
    ) -> Tensor:
        del frames, conditioning
        prediction = prediction[..., :-1, :, :]
        if (
            self.target_spatial_shape is not None
            and tuple(int(v) for v in prediction.shape[-2:]) != self.target_spatial_shape
        ):
            prediction = F.interpolate(
                prediction,
                size=(int(prediction.shape[2]), *self.target_spatial_shape),
                mode="trilinear",
                align_corners=False,
            )
        if velocity_scale_mps_per_px_frame is None:
            return prediction
        doppler_prediction = prediction[:, : self.doppler_channels]
        marker_prediction = prediction[:, self.doppler_channels :]
        scale = (
            cast(Tensor, velocity_scale_mps_per_px_frame)
            .reshape(-1, 1, 1, 1, 1)
            .to(device=prediction.device, dtype=prediction.dtype)
        )
        if scale.shape[0] != doppler_prediction.shape[0]:
            raise ValueError("velocity_scale_mps_per_px_frame batch dimension does not match frames")
        if not bool(torch.isfinite(scale).all()) or bool(torch.any(scale <= 0.0)):
            raise ValueError("velocity_scale_mps_per_px_frame must be finite and positive")
        physical_doppler = doppler_prediction * scale
        if self.marker_channels <= 0:
            return physical_doppler
        return torch.cat([physical_doppler, marker_prediction], dim=1)


class SegmentationHead(TaskHead):
    requires_conditioning = False

    def __init__(
        self,
        *,
        out_channels: int,
        head_channels: int | None = None,
        target_spatial_shape: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.out_channels = int(out_channels)
        self.head_channels = self.out_channels if head_channels is None else int(head_channels)
        if self.out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")
        if self.head_channels <= 0:
            raise ValueError(f"head_channels must be positive, got {head_channels}")
        self.target_spatial_shape = (
            None if target_spatial_shape is None else as_2tuple(target_spatial_shape, name="target_spatial_shape")
        )
        self.final_projection: nn.Module = (
            nn.Identity()
            if self.head_channels == self.out_channels
            else nn.Conv3d(self.head_channels, self.out_channels, kernel_size=1)
        )

    @property
    def expected_out_channels(self) -> int:
        return self.head_channels

    def forward(
        self,
        prediction: Tensor,
        *,
        frames: Tensor,
        conditioning: Tensor | None = None,
        **_: object,
    ) -> Tensor:
        del conditioning
        time_steps = int(frames.shape[2])
        target_hw = self.target_spatial_shape or (int(frames.shape[3]), int(frames.shape[4]))
        if int(prediction.shape[1]) != self.head_channels:
            raise ValueError(
                f"SegmentationHead expected {self.head_channels} feature channels, got {prediction.shape[1]}"
            )
        if tuple(int(v) for v in prediction.shape[-2:]) != target_hw:
            prediction = F.interpolate(
                prediction,
                size=(int(time_steps), *target_hw),
                mode="trilinear",
                align_corners=False,
            )
        return cast(Tensor, self.final_projection(prediction))


KernelSpec = int | Sequence[int]


class DenseSectorUNet(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[Spec3] | None,
        kernel_size: KernelSpec = 3,
        up_kernel_size: KernelSpec = 3,
        kernel_sizes: Sequence[KernelSpec] | KernelSpec | None = None,
        up_kernel_sizes: Sequence[KernelSpec] | KernelSpec | None = None,
        transpose_post_conv_kernel_size: Spec3 | None = None,
        output_spatial_stride: int | Sequence[int] = (1, 1),
        num_res_units: int = 0,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.channels = tuple(int(channel) for channel in channels)
        if len(self.channels) < 2:
            raise ValueError("DenseSectorUNet requires at least two channel stages")
        self.strides = _unet_strides(strides, level_count=len(self.channels) - 1)
        self.output_spatial_stride = as_2tuple(output_spatial_stride, name="output_spatial_stride")
        self.num_res_units = int(num_res_units)
        self.transpose_post_conv_kernel_size = (
            None
            if transpose_post_conv_kernel_size is None
            else as_3tuple(transpose_post_conv_kernel_size, name="transpose_post_conv_kernel_size")
        )
        self.kernel_sizes = _normalize_kernel_specs(kernel_sizes, count=len(self.channels), default=kernel_size)
        self.up_kernel_sizes = _normalize_kernel_specs(
            up_kernel_sizes,
            count=len(self.channels) - 1,
            default=up_kernel_size,
        )
        self.model = self._create_block(level=0, in_channels=self.in_channels, out_channels=self.out_channels)

    def _create_block(self, *, level: int, in_channels: int, out_channels: int) -> nn.Module:
        channel_count = self.channels[level]
        stride = self.strides[level]
        kernel_size = self.kernel_sizes[level]
        is_top = level == 0
        if level < len(self.channels) - 2:
            subblock = self._create_block(level=level + 1, in_channels=channel_count, out_channels=channel_count)
            up_channels = channel_count * 2
        else:
            subblock = self._get_bottom_layer(
                in_channels=channel_count,
                out_channels=self.channels[level + 1],
                kernel_size=self.kernel_sizes[level + 1],
            )
            up_channels = channel_count + self.channels[level + 1]
        down = self._get_down_layer(
            in_channels=in_channels,
            out_channels=channel_count,
            strides=stride,
            kernel_size=kernel_size,
        )
        up = self._get_up_layer(
            in_channels=up_channels,
            out_channels=out_channels,
            strides=stride,
            up_kernel_size=self.up_kernel_sizes[level],
            conv_kernel_size=kernel_size,
            is_top=is_top,
        )
        return nn.Sequential(down, SkipConnection(subblock), up)

    def _get_down_layer(
        self,
        *,
        in_channels: int,
        out_channels: int,
        strides: Int3,
        kernel_size: Int3,
    ) -> nn.Module:
        if self.num_res_units > 0:
            return ReplicateResidualUnit3d(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                kernel_size=kernel_size,
                subunits=self.num_res_units,
            )
        return ReplicateConv3dBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            kernel_size=kernel_size,
        )

    def _get_bottom_layer(self, *, in_channels: int, out_channels: int, kernel_size: Int3) -> nn.Module:
        return self._get_down_layer(
            in_channels=in_channels,
            out_channels=out_channels,
            strides=(1, 1, 1),
            kernel_size=kernel_size,
        )

    def _get_up_layer(
        self,
        *,
        in_channels: int,
        out_channels: int,
        strides: Int3,
        up_kernel_size: Int3,
        conv_kernel_size: Int3,
        is_top: bool,
    ) -> nn.Module:
        if is_top and self.output_spatial_stride != (1, 1):
            return self._get_top_output_projection_layer(
                in_channels=in_channels,
                out_channels=out_channels,
                down_strides=strides,
                kernel_size=conv_kernel_size,
            )
        conv: nn.Module = ReplicateConvTranspose3dBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            kernel_size=up_kernel_size,
            conv_only=is_top and self.num_res_units == 0,
            post_conv_kernel_size=self.transpose_post_conv_kernel_size,
        )
        if self.num_res_units > 0:
            conv = nn.Sequential(
                conv,
                ReplicateResidualUnit3d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    strides=1,
                    kernel_size=conv_kernel_size,
                    subunits=1,
                    last_conv_only=is_top,
                ),
            )
        return conv

    def _get_top_output_projection_layer(
        self,
        *,
        in_channels: int,
        out_channels: int,
        down_strides: Int3,
        kernel_size: Int3,
    ) -> nn.Module:
        projection_stride = _top_output_projection_stride(
            output_spatial_stride=self.output_spatial_stride,
            top_down_stride=(int(down_strides[1]), int(down_strides[2])),
        )
        conv: nn.Module = ReplicateConv3dBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            strides=(1, projection_stride[0], projection_stride[1]),
            kernel_size=kernel_size,
            conv_only=self.num_res_units == 0,
        )
        if self.num_res_units > 0:
            conv = nn.Sequential(
                conv,
                ReplicateResidualUnit3d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    strides=1,
                    kernel_size=kernel_size,
                    subunits=1,
                    last_conv_only=True,
                ),
            )
        return conv

    def forward(self, tensor: Tensor) -> Tensor:
        return cast(Tensor, self.model(tensor))


class ReplicateConv3dBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        strides: Spec3,
        kernel_size: Spec3,
        conv_only: bool = False,
    ) -> None:
        super().__init__()
        self.padding = _same_padding_3d(kernel_size)
        layers: list[nn.Module] = [
            nn.Conv3d(
                int(in_channels),
                int(out_channels),
                kernel_size=as_3tuple(kernel_size, name="kernel_size"),
                stride=as_3tuple(strides, name="stride"),
                padding=0,
                bias=True,
            )
        ]
        if not conv_only:
            layers.extend([nn.InstanceNorm3d(int(out_channels)), nn.PReLU()])
        self.block = nn.Sequential(*layers)

    def forward(self, tensor: Tensor) -> Tensor:
        return cast(Tensor, self.block(_replicate_pad_3d(tensor, self.padding)))


class ReplicateConvTranspose3dBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        strides: Spec3,
        kernel_size: Spec3,
        conv_only: bool = False,
        post_conv_kernel_size: Spec3 | None = None,
    ) -> None:
        super().__init__()
        stride = as_3tuple(strides, name="stride")
        kernel = as_3tuple(kernel_size, name="kernel_size")
        padding = _same_padding_3d(kernel)
        output_padding: Int3 = (
            max(0, int(stride[0]) - 1),
            max(0, int(stride[1]) - 1),
            max(0, int(stride[2]) - 1),
        )
        self.conv = nn.ConvTranspose3d(
            int(in_channels),
            int(out_channels),
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=True,
        )
        self.post = nn.Identity() if conv_only else nn.Sequential(nn.InstanceNorm3d(int(out_channels)), nn.PReLU())
        self.post_conv = (
            nn.Identity()
            if post_conv_kernel_size is None
            else ReplicateDepthwiseConv3dBlock(channels=int(out_channels), kernel_size=post_conv_kernel_size)
        )

    def forward(self, tensor: Tensor) -> Tensor:
        return cast(Tensor, self.post_conv(self.post(self.conv(tensor))))


class ReplicateDepthwiseConv3dBlock(nn.Module):
    def __init__(self, *, channels: int, kernel_size: Spec3) -> None:
        super().__init__()
        channel_count = int(channels)
        kernel = as_3tuple(kernel_size, name="kernel_size")
        self.padding = _same_padding_3d(kernel)
        self.conv = nn.Conv3d(
            channel_count,
            channel_count,
            kernel_size=kernel,
            stride=1,
            padding=0,
            groups=channel_count,
            bias=True,
        )
        _init_depthwise_identity(self.conv)

    def forward(self, tensor: Tensor) -> Tensor:
        return cast(Tensor, self.conv(_replicate_pad_3d(tensor, self.padding)))


class ReplicateResidualUnit3d(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        strides: Spec3,
        kernel_size: Spec3,
        subunits: int,
        last_conv_only: bool = False,
    ) -> None:
        super().__init__()
        stride = as_3tuple(strides, name="stride")
        unit_count = max(1, int(subunits))
        blocks: list[nn.Module] = []
        current_channels = int(in_channels)
        for index in range(unit_count):
            is_last = index == unit_count - 1
            blocks.append(
                ReplicateConv3dBlock(
                    in_channels=current_channels,
                    out_channels=int(out_channels),
                    strides=stride if index == 0 else (1, 1, 1),
                    kernel_size=kernel_size,
                    conv_only=last_conv_only and is_last,
                )
            )
            current_channels = int(out_channels)
        self.blocks = nn.Sequential(*blocks)
        self.residual = (
            nn.Identity()
            if int(in_channels) == int(out_channels) and stride == (1, 1, 1)
            else nn.Conv3d(int(in_channels), int(out_channels), kernel_size=1, stride=stride, bias=True)
        )

    def forward(self, tensor: Tensor) -> Tensor:
        return cast(Tensor, self.blocks(tensor) + self.residual(tensor))


class TaskModel(nn.Module):
    """Shared sector-video pipeline: optional conditioning, padded backbone, crop, head."""

    def __init__(
        self,
        *,
        head: TaskHead,
        in_channels_per_frame: int = 1,
        channels: Sequence[int] = (8, 16),
        strides: Sequence[Spec3] | None = None,
        kernel_sizes: Sequence[Spec3] | None = None,
        up_kernel_sizes: Sequence[Spec3] | None = None,
        transpose_post_conv_kernel_size: Spec3 | None = None,
        num_res_units: int = 0,
        output_spatial_stride: int | Sequence[int] = (1, 1),
        conditioning_features: int = 0,
    ) -> None:
        super().__init__()
        self.head = head
        if isinstance(head, TissueDopplerHead) and int(conditioning_features) != 0:
            raise ValueError(f"TissueDopplerHead requires conditioning_features=0, got {conditioning_features}")
        self.in_channels_per_frame = int(in_channels_per_frame)
        self.conditioning_features = int(conditioning_features)
        self.output_spatial_stride = as_2tuple(output_spatial_stride, name="output_spatial_stride")
        self.conditioning_projection = (
            nn.Linear(self.conditioning_features, self.in_channels_per_frame)
            if self.conditioning_features > 0
            else None
        )
        self.network = DenseSectorUNet(
            in_channels=self.in_channels_per_frame,
            out_channels=head.expected_out_channels,
            channels=channels,
            strides=strides,
            kernel_sizes=kernel_sizes,
            up_kernel_sizes=up_kernel_sizes,
            transpose_post_conv_kernel_size=transpose_post_conv_kernel_size,
            output_spatial_stride=self.output_spatial_stride,
            num_res_units=int(num_res_units),
        )

    @property
    def out_channels(self) -> int:
        return int(getattr(self.head, "out_channels", self.network.out_channels))

    @property
    def head_channels(self) -> int:
        return int(getattr(self.head, "head_channels", self.network.out_channels))

    def forward(self, frames: Tensor, *, conditioning: Tensor | None = None, **head_kwargs: object) -> Tensor:
        if frames.ndim != 5:
            raise ValueError(f"frames must be [B,C,T,H,W], got {tuple(frames.shape)}")
        _batch, _channels, time_steps, height, width = frames.shape
        conditioned = _condition_video_frames(
            frames,
            conditioning=conditioning,
            projection=self.conditioning_projection,
            require_conditioning=self.head.requires_conditioning,
        )
        padded_input, padding = _pad_to_stride_multiple(conditioned, stride_multiple=_stride_multiple(self.network))
        prediction = self.network(padded_input)
        output_hw = output_spatial_shape(
            input_shape=(int(height), int(width)),
            output_spatial_stride=self.output_spatial_stride,
        )
        pad_t, pad_h, pad_w = padding
        if pad_t != 0 or pad_h != 0 or pad_w != 0:
            prediction = prediction[..., : int(time_steps), : output_hw[0], : output_hw[1]]
        return cast(Tensor, self.head(prediction, frames=frames, conditioning=conditioning, **head_kwargs))


_HEAD_REGISTRY = {
    "color_doppler": (
        ColorDopplerHead,
        {"doppler_channels", "target_spatial_shape", "temporal_upsample_factor"},
    ),
    "tissue_doppler": (TissueDopplerHead, {"doppler_channels", "marker_channels", "target_spatial_shape"}),
    "segmentation": (SegmentationHead, {"out_channels", "head_channels", "target_spatial_shape"}),
}

_BACKBONE_KEYS = {
    "in_channels_per_frame",
    "channels",
    "strides",
    "kernel_sizes",
    "up_kernel_sizes",
    "transpose_post_conv_kernel_size",
    "num_res_units",
    "output_spatial_stride",
    "conditioning_features",
}


def build_model(*, kind: str, head: str | type[TaskHead] | None = None, **kwargs: object) -> TaskModel:
    if kind in _HEAD_REGISTRY and head is None:
        head_cls, head_keys = _HEAD_REGISTRY[kind]
        head_kwargs = {key: kwargs.pop(key) for key in list(kwargs) if key in head_keys}
    else:
        head_cls = _resolve_head_class(head or kind)
        head_kwargs = {key: kwargs.pop(key) for key in list(kwargs) if key not in _BACKBONE_KEYS}
    backbone_kwargs = {key: kwargs.pop(key) for key in list(kwargs) if key in _BACKBONE_KEYS}
    if kwargs:
        raise ValueError(f"Unknown model config keys for kind={kind!r}: {sorted(kwargs)}")
    head_module = cast(TaskHead, head_cls(**cast(Any, head_kwargs)))
    return TaskModel(head=head_module, **cast(Any, backbone_kwargs))


def framewise_temporal_config(raw: dict[str, object]) -> dict[str, object]:
    config = deepcopy(raw)
    model = config.get("model")
    if not isinstance(model, dict):
        raise TypeError("framewise_temporal_config requires a mapping `model` section")
    channels = model.get("channels")
    level_count = len(channels) if isinstance(channels, Sequence) and not isinstance(channels, (str, bytes)) else None
    if "kernel_sizes" in model:
        model["kernel_sizes"] = _framewise_kernel_specs(cast(object, model["kernel_sizes"]))
    elif level_count is not None:
        model["kernel_sizes"] = [[1, 3, 3] for _ in range(int(level_count))]
    else:
        raise ValueError("framewise_temporal_config requires model.channels when model.kernel_sizes is absent")
    if "up_kernel_sizes" in model:
        model["up_kernel_sizes"] = _framewise_kernel_specs(cast(object, model["up_kernel_sizes"]))
    elif level_count is not None:
        model["up_kernel_sizes"] = [[1, 3, 3] for _ in range(max(1, int(level_count) - 1))]
    else:
        raise ValueError("framewise_temporal_config requires model.channels when model.up_kernel_sizes is absent")
    if "transpose_post_conv_kernel_size" in model:
        model["transpose_post_conv_kernel_size"] = _framewise_kernel_spec(model["transpose_post_conv_kernel_size"])
    return config


def _framewise_kernel_specs(value: object) -> object:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and value and _is_kernel_spec(value):
        return _framewise_kernel_spec(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_framewise_kernel_spec(item) for item in value]
    return _framewise_kernel_spec(value)


def _framewise_kernel_spec(value: object) -> object:
    if isinstance(value, int):
        return [1, int(value), int(value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = [int(item) for item in value]
        if len(items) == 2:
            return [1, items[0], items[1]]
        if len(items) == 3:
            return [1, items[1], items[2]]
    return value


def _resolve_head_class(value: str | type[TaskHead]) -> type[TaskHead]:
    if isinstance(value, type):
        return value
    if "." not in value and ":" not in value:
        raise ValueError(
            f"Unknown task kind: {value!r}; expected one of {sorted(_HEAD_REGISTRY)} or a dotted TaskHead path"
        )
    module_name, _, attr = value.replace(":", ".").rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"Invalid TaskHead path: {value!r}")
    candidate = getattr(import_module(module_name), attr)
    if not isinstance(candidate, type) or not issubclass(candidate, TaskHead):
        raise TypeError(f"{value!r} must resolve to a TaskHead subclass")
    return candidate


def as_2tuple(value: int | Sequence[int], *, name: str) -> tuple[int, int]:
    if isinstance(value, int):
        return (int(value), int(value))
    result = tuple(int(item) for item in value)
    if len(result) != 2 or min(result) <= 0:
        raise ValueError(f"{name} must have two positive values, got {value}")
    return (result[0], result[1])


def as_3tuple(value: Spec3, *, name: str) -> Int3:
    if isinstance(value, int):
        return (int(value), int(value), int(value))
    result = tuple(int(item) for item in value)
    if len(result) != 3 or min(result) <= 0:
        raise ValueError(f"{name} must have three positive values, got {value}")
    return cast(Int3, result)


def _unet_strides(values: Sequence[Spec3] | None, *, level_count: int) -> tuple[Int3, ...]:
    if values is None:
        return tuple((1, 1, 1) for _ in range(int(level_count)))
    specs = tuple(as_3tuple(value, name="stride") for value in values)
    if len(specs) != int(level_count):
        raise ValueError(f"Expected {level_count} stride specs, got {len(specs)}")
    return specs


def _normalize_kernel_specs(
    specs: Sequence[KernelSpec] | KernelSpec | None,
    *,
    count: int,
    default: KernelSpec,
) -> tuple[Int3, ...]:
    if specs is None:
        return tuple(_as_3d_kernel_size(default) for _ in range(count))
    if _is_kernel_spec(specs):
        kernel_size = _as_3d_kernel_size(cast(KernelSpec, specs))
        return tuple(kernel_size for _ in range(count))
    sequence_specs = cast(Sequence[KernelSpec], specs)
    if len(sequence_specs) != count:
        raise ValueError(f"Expected {count} kernel specs, got {len(sequence_specs)}")
    return tuple(_as_3d_kernel_size(spec) for spec in sequence_specs)


def _is_kernel_spec(value: object) -> bool:
    if isinstance(value, int):
        return True
    if not isinstance(value, Sequence):
        return False
    if len(value) != 3:
        return False
    return all(isinstance(item, int) for item in value)


def _as_3d_kernel_size(kernel_size: KernelSpec) -> Int3:
    return as_3tuple(cast(Spec3, kernel_size), name="kernel_size")


def _same_padding_3d(kernel_size: Spec3) -> Int3:
    kernel = as_3tuple(kernel_size, name="kernel_size")
    if any(value % 2 == 0 for value in kernel):
        raise ValueError(f"Replication same padding requires odd kernel sizes, got {kernel}")
    return cast(Int3, tuple(value // 2 for value in kernel))


def _replicate_pad_3d(tensor: Tensor, padding: Int3) -> Tensor:
    pad_t, pad_h, pad_w = padding
    if pad_t == 0 and pad_h == 0 and pad_w == 0:
        return tensor
    return F.pad(tensor, (pad_w, pad_w, pad_h, pad_h, pad_t, pad_t), mode="replicate")


def _init_depthwise_identity(conv: nn.Conv3d) -> None:
    with torch.no_grad():
        conv.weight.zero_()
        center = tuple(int(size) // 2 for size in conv.kernel_size)
        conv.weight[:, 0, center[0], center[1], center[2]] = 1.0
        if conv.bias is not None:
            conv.bias.zero_()


def _pad_to_stride_multiple(tensor: Tensor, *, stride_multiple: Int3) -> tuple[Tensor, Int3]:
    pad_t = (-int(tensor.shape[-3])) % max(1, int(stride_multiple[0]))
    pad_h = (-int(tensor.shape[-2])) % max(1, int(stride_multiple[1]))
    pad_w = (-int(tensor.shape[-1])) % max(1, int(stride_multiple[2]))
    if pad_t == 0 and pad_h == 0 and pad_w == 0:
        return tensor, (0, 0, 0)
    return F.pad(tensor, (0, pad_w, 0, pad_h, 0, pad_t), mode="replicate"), (pad_t, pad_h, pad_w)


def _condition_video_frames(
    frames: Tensor,
    *,
    conditioning: Tensor | None,
    projection: nn.Linear | None,
    require_conditioning: bool,
) -> Tensor:
    if projection is None:
        return frames
    if conditioning is None:
        if require_conditioning:
            raise ValueError("conditioning is required by this model configuration")
        return frames
    projected = projection(conditioning.reshape(frames.shape[0], -1).to(device=frames.device, dtype=frames.dtype))
    if projected.shape[0] != frames.shape[0] or projected.shape[1] != frames.shape[1]:
        raise ValueError("conditioning projection must match batch and input channel dimensions")
    return cast(Tensor, frames + projected.view(frames.shape[0], frames.shape[1], 1, 1, 1))


def _stride_multiple(module: nn.Module) -> Int3:
    strides = getattr(module, "strides", ())
    multiple = [1, 1, 1]
    for stride in strides:
        stride_tuple = as_3tuple(cast(Spec3, stride), name="stride")
        for idx, value in enumerate(stride_tuple):
            multiple[idx] *= int(value)
    return cast(Int3, (max(1, multiple[0]), max(1, multiple[1]), max(1, multiple[2])))


def output_spatial_shape(
    *,
    input_shape: tuple[int, int],
    output_spatial_stride: int | Sequence[int],
) -> tuple[int, int]:
    stride_h, stride_w = as_2tuple(output_spatial_stride, name="output_spatial_stride")
    return _ceil_div(input_shape[0], stride_h), _ceil_div(input_shape[1], stride_w)


def _ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def _top_output_projection_stride(
    *, output_spatial_stride: Int3 | tuple[int, int], top_down_stride: tuple[int, int]
) -> tuple[int, int]:
    stride_h, stride_w = int(output_spatial_stride[0]), int(output_spatial_stride[1])
    top_h = max(1, int(top_down_stride[0]))
    top_w = max(1, int(top_down_stride[1]))
    if stride_h < top_h or stride_w < top_w or stride_h % top_h != 0 or stride_w % top_w != 0:
        raise ValueError(
            "output_spatial_stride must be an integer multiple of the top encoder stride "
            f"{(top_h, top_w)}, got {(stride_h, stride_w)}"
        )
    return stride_h // top_h, stride_w // top_w


def norm_groups(channels: int) -> int:
    channel_count = max(1, int(channels))
    for group_count in range(min(8, channel_count), 0, -1):
        if channel_count % group_count == 0:
            return group_count
    return 1


__all__ = [
    "ColorDopplerHead",
    "SegmentationHead",
    "TaskHead",
    "TaskModel",
    "TissueDopplerHead",
    "as_2tuple",
    "build_model",
    "norm_groups",
]
