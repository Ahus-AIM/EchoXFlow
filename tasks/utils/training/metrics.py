from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tasks.utils.models.unet import TaskModel
from tasks.utils.training.loss import masked_color_doppler_loss_terms, masked_velocity_band_l1_loss


def segmentation_val_metrics(model: TaskModel, sample: object, config: object) -> dict[str, Tensor]:
    typed_sample = cast(Any, sample)
    target = typed_sample.cartesian_metric_target_masks
    valid_mask = typed_sample.cartesian_metric_valid_mask
    if target is None or valid_mask is None:
        return {}
    logits = cast(Tensor, model(typed_sample.frames).float())
    target = target.to(device=logits.device, dtype=logits.dtype)
    valid_mask = valid_mask.to(device=logits.device, dtype=logits.dtype)
    valid_foreground = None
    if tuple(valid_mask.shape[-2:]) == tuple(logits.shape[-2:]):
        valid_foreground = _segmentation_foreground_valid_mask(valid_mask, target=target).to(
            dtype=target.dtype, device=target.device
        )
    probs = _segmentation_foreground_probabilities(
        logits,
        target_channels=int(target.shape[1]),
        valid_foreground_mask=valid_foreground,
    )
    probs = _prediction_to_cartesian_metric_grid(probs, typed_sample, target=target)
    if probs.shape != target.shape:
        raise ValueError(
            f"cartesian metric target shape {tuple(target.shape)} must match probabilities {tuple(probs.shape)}"
        )
    dice, channel_valid = _cartesian_soft_dice_by_channel(
        probs=probs,
        target=target,
        valid_mask=valid_mask,
        eps=_dice_eps(config),
    )
    if not bool(torch.any(channel_valid)):
        zero = logits.sum() * 0.0
        return {"cartesian_dice_mean": zero}
    terms: dict[str, Tensor] = {}
    for channel_index in range(int(target.shape[1])):
        if bool(channel_valid[channel_index]):
            terms[f"cartesian_dice_ch{channel_index}"] = dice[channel_index]
    terms["cartesian_dice_mean"] = dice[channel_valid].mean()
    return terms


def color_doppler_cartesian_val_metrics(model: TaskModel, sample: object, config: object) -> dict[str, Tensor]:
    del config
    typed_sample = cast(Any, sample)
    target = _required_metric_tensor(typed_sample, "cartesian_metric_doppler_target")
    valid_mask = _required_metric_tensor(typed_sample, "cartesian_metric_valid_mask")
    velocity_loss_mask = _required_metric_tensor(typed_sample, "cartesian_metric_velocity_loss_mask")
    color_box_target = _required_metric_tensor(typed_sample, "cartesian_metric_color_box_target")
    prediction = cast(Tensor, model(typed_sample.frames, conditioning=typed_sample.conditioning).float())
    prediction = _prediction_to_cartesian_metric_grid(prediction, typed_sample, target=target)
    terms = masked_color_doppler_loss_terms(
        prediction,
        target.to(device=prediction.device, dtype=prediction.dtype),
        valid_mask=valid_mask.to(device=prediction.device, dtype=prediction.dtype),
        velocity_loss_mask=velocity_loss_mask.to(device=prediction.device, dtype=prediction.dtype),
        color_box_target=color_box_target.to(device=prediction.device, dtype=prediction.dtype),
        outside_box_weight=0.0,
    )
    return {
        "cartesian_velocity": terms["velocity"],
        "cartesian_power": terms["power"],
        "cartesian_std": terms["std"],
    }


def tissue_doppler_cartesian_val_metrics(model: TaskModel, sample: object, config: object) -> dict[str, Tensor]:
    del config
    typed_sample = cast(Any, sample)
    target = _required_metric_tensor(typed_sample, "cartesian_metric_doppler_target")
    prediction = cast(
        Tensor,
        model(
            typed_sample.frames,
            velocity_scale_mps_per_px_frame=typed_sample.velocity_scale_mps_per_px_frame,
        ).float(),
    )
    prediction = _prediction_to_cartesian_metric_grid(prediction, typed_sample, target=target)
    loss = masked_velocity_band_l1_loss(
        prediction,
        target.to(device=prediction.device, dtype=prediction.dtype),
        typed_sample.sector_velocity_limit_mps,
    )
    return {"cartesian_velocity": loss}


def _compute_validation_metric_terms(module: nn.Module, sample: object) -> Mapping[str, Tensor]:
    if not hasattr(module, "validation_metric_terms"):
        return {}
    raw_terms = cast(Any, module).validation_metric_terms(sample)
    if raw_terms is None:
        return {}
    if not isinstance(raw_terms, Mapping):
        raise TypeError("validation_metric_terms must return a mapping of metric names to scalar tensors")
    return {str(key): _as_scalar_metric_tensor(value, label=str(key)) for key, value in raw_terms.items()}


def _prediction_to_cartesian_metric_grid(prediction: Tensor, sample: Any, *, target: Tensor) -> Tensor:
    sample_grid = getattr(sample, "cartesian_metric_sample_grid", None)
    if not isinstance(sample_grid, Tensor):
        if tuple(prediction.shape[-2:]) == tuple(target.shape[-2:]):
            return prediction
        raise ValueError("Cartesian benchmark metric requires cartesian_metric_sample_grid when shapes differ")
    if tuple(prediction.shape[-2:]) == tuple(target.shape[-2:]) and sample_grid.numel() == 0:
        return prediction
    if sample_grid.ndim != 4 or sample_grid.shape[0] != prediction.shape[0] or sample_grid.shape[-1] != 2:
        raise ValueError(
            "cartesian_metric_sample_grid must be [batch,height,width,2], "
            f"got {tuple(sample_grid.shape)} for prediction {tuple(prediction.shape)}"
        )
    batch_size, channel_count, frame_count, _height, _width = prediction.shape
    out_h, out_w = int(target.shape[-2]), int(target.shape[-1])
    converted: list[Tensor] = []
    for batch_index in range(int(batch_size)):
        grid = sample_grid[batch_index : batch_index + 1].to(device=prediction.device, dtype=prediction.dtype)
        grid = grid.expand(int(channel_count) * int(frame_count), out_h, out_w, 2)
        values = (
            prediction[batch_index]
            .permute(1, 0, 2, 3)
            .reshape(int(channel_count) * int(frame_count), 1, prediction.shape[-2], prediction.shape[-1])
        )
        sampled = F.grid_sample(values, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        converted.append(sampled.reshape(int(frame_count), int(channel_count), out_h, out_w).permute(1, 0, 2, 3))
    return torch.stack(converted, dim=0)


def _required_metric_tensor(sample: Any, name: str) -> Tensor:
    value = getattr(sample, name, None)
    if not isinstance(value, Tensor):
        raise ValueError(f"Cartesian benchmark metric requires sample.{name}")
    return value


def _dice_eps(config: object) -> float:
    metric_section = config.metrics if hasattr(config, "metrics") else {}
    if isinstance(metric_section, dict) and "dice_eps" in metric_section:
        return float(metric_section["dice_eps"])
    loss_section = config.loss if hasattr(config, "loss") else {}
    value = loss_section.get("dice_eps", 1e-6) if isinstance(loss_section, dict) else 1e-6
    return float(value)


def _segmentation_foreground_probabilities(
    logits: Tensor,
    *,
    target_channels: int,
    valid_foreground_mask: Tensor | None = None,
) -> Tensor:
    expected_channels = int(target_channels) + 1
    if logits.shape[1] != expected_channels:
        raise ValueError(
            f"segmentation logits must have {expected_channels} channels "
            f"(background + {target_channels} targets), got {logits.shape[1]}"
        )
    if valid_foreground_mask is not None:
        if valid_foreground_mask.shape != logits[:, 1:].shape:
            raise ValueError(
                f"valid foreground mask shape {tuple(valid_foreground_mask.shape)} "
                f"must match foreground logits {tuple(logits[:, 1:].shape)}"
            )
        logits = logits.clone()
        logits[:, 1:] = logits[:, 1:].masked_fill(valid_foreground_mask <= 0.0, torch.finfo(logits.dtype).min)
    return torch.softmax(logits, dim=1)[:, 1:]


def _cartesian_soft_dice_by_channel(
    *,
    probs: Tensor,
    target: Tensor,
    valid_mask: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor]:
    valid = _segmentation_foreground_valid_mask(valid_mask, target=target).to(dtype=target.dtype, device=target.device)
    prediction = probs * valid
    target = target * valid
    reduce_dims = (0, 2, 3, 4)
    intersection = torch.sum(prediction * target, dim=reduce_dims)
    dice_denominator = torch.sum(prediction + target, dim=reduce_dims)
    valid_pixels = torch.sum(valid, dim=reduce_dims)
    dice = (2.0 * intersection + eps) / (dice_denominator + eps)
    return dice, valid_pixels > 0.0


def _segmentation_foreground_valid_mask(valid_mask: Tensor, *, target: Tensor) -> Tensor:
    if valid_mask.shape[1] == 1:
        return valid_mask.expand(-1, target.shape[1], -1, target.shape[-2], target.shape[-1])
    return valid_mask.expand(-1, -1, -1, target.shape[-2], target.shape[-1])


def _as_scalar_metric_tensor(value: object, *, label: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"metric term {label!r} must be a tensor")
    _validate_metric(value)
    return value


def _validate_metric(metric: Tensor) -> None:
    if metric.ndim != 0:
        raise ValueError(f"validation metric must be a scalar tensor, got shape {tuple(metric.shape)}")
    if not bool(torch.isfinite(metric.detach()).all()):
        raise FloatingPointError(f"non-finite validation metric: {float(metric.detach().cpu())}")
