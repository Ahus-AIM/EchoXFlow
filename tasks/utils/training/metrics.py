from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor

from tasks.utils.training.loss import (
    _color_doppler_prediction,
    _nan_where_invalid,
    _segmentation_annotated_frame_dice_by_item,
    _segmentation_annotated_frame_dice_by_item_channel,
    _segmentation_dice_reduction,
    _segmentation_prediction,
    _tissue_doppler_prediction,
    masked_color_doppler_loss_terms,
    masked_velocity_band_l1_loss,
    masked_wrapped_velocity_l1,
)

if TYPE_CHECKING:
    from tasks.utils.models.unet import TaskModel


def segmentation_val_metrics(model: TaskModel, sample: object, config: object) -> dict[str, Tensor]:
    typed_sample = cast(Any, sample)
    target = typed_sample.cartesian_metric_target_masks
    valid_mask = typed_sample.cartesian_metric_valid_mask
    if target is None or valid_mask is None:
        return {}
    logits = _segmentation_prediction(model, typed_sample.frames, config=config).float()
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
    eps = _dice_eps(config)
    reduction = _segmentation_dice_reduction(config, prefer_metrics=True)
    # Headline Dice is thresholded: each metric-grid pixel takes the argmax over background and the structures
    # annotated in this clip. The soft Dice of the probabilities is kept as a secondary metric.
    hard = _segmentation_hard_foreground(probs, valid_mask=valid_mask, target=target)
    terms = {
        **_segmentation_dice_terms(hard, target=target, valid_mask=valid_mask, eps=eps, reduction=reduction, prefix=""),
        **_segmentation_dice_terms(
            probs, target=target, valid_mask=valid_mask, eps=eps, reduction=reduction, prefix="soft_"
        ),
    }
    terms["n_valid_frames"] = _valid_frame_count(valid_mask)
    return terms


def _segmentation_hard_foreground(probs: Tensor, *, valid_mask: Tensor, target: Tensor) -> Tensor:
    """One-hot foreground of the argmax over background and the valid structures on the metric grid."""
    valid = _segmentation_foreground_valid_mask(valid_mask, target=target).to(dtype=probs.dtype, device=probs.device)
    foreground = probs * (valid > 0.0).to(dtype=probs.dtype)
    background = (1.0 - foreground.sum(dim=1, keepdim=True)).clamp_min(0.0)
    labels = torch.argmax(torch.cat([background, foreground], dim=1), dim=1)
    channels = torch.arange(1, int(foreground.shape[1]) + 1, device=labels.device).view(1, -1, 1, 1, 1)
    return (labels.unsqueeze(1) == channels).to(dtype=probs.dtype)


def _segmentation_dice_terms(
    prediction: Tensor,
    *,
    target: Tensor,
    valid_mask: Tensor,
    eps: float,
    reduction: str,
    prefix: str,
) -> dict[str, Tensor]:
    """Per-clip Dice per channel and averaged over the channels present in the clip (NaN when none is present)."""
    if reduction == "annotated_frame":
        frame_dice, class_frame_valid = _cartesian_soft_dice_by_item_channel_frame(
            probs=prediction,
            target=target,
            valid_mask=valid_mask,
            eps=eps,
        )
        dice, channel_valid = _segmentation_annotated_frame_dice_by_item_channel(
            dice=frame_dice,
            class_frame_valid=class_frame_valid,
        )
        dice_mean, item_valid = _segmentation_annotated_frame_dice_by_item(
            dice=frame_dice,
            class_frame_valid=class_frame_valid,
        )
    else:
        dice, channel_valid = _cartesian_soft_dice_by_item_channel(
            probs=prediction,
            target=target,
            valid_mask=valid_mask,
            eps=eps,
        )
        present = channel_valid.to(dtype=dice.dtype)
        dice_mean = torch.sum(dice * present, dim=1) / present.sum(dim=1).clamp_min(1.0)
        item_valid = present.sum(dim=1) > 0.0
    terms = {
        f"cartesian_{prefix}dice_ch{channel_index}": _nan_where_invalid(
            dice[:, channel_index], channel_valid[:, channel_index]
        )
        for channel_index in range(int(target.shape[1]))
    }
    terms[f"cartesian_{prefix}dice_mean"] = _nan_where_invalid(dice_mean, item_valid)
    return terms


def color_doppler_cartesian_val_metrics(model: TaskModel, sample: object, config: object) -> dict[str, Tensor]:
    typed_sample = cast(Any, sample)
    target = _required_metric_tensor(typed_sample, "cartesian_metric_doppler_target")
    valid_mask = _required_metric_tensor(typed_sample, "cartesian_metric_valid_mask")
    velocity_loss_mask = _required_metric_tensor(typed_sample, "cartesian_metric_velocity_loss_mask")
    color_box_target = _required_metric_tensor(typed_sample, "cartesian_metric_color_box_target")
    prediction = _color_doppler_prediction(model, typed_sample, config=config).float()
    prediction = _prediction_to_cartesian_metric_grid(prediction, typed_sample, target=target)
    terms = masked_color_doppler_loss_terms(
        prediction,
        target.to(device=prediction.device, dtype=prediction.dtype),
        valid_mask=valid_mask.to(device=prediction.device, dtype=prediction.dtype),
        velocity_loss_mask=velocity_loss_mask.to(device=prediction.device, dtype=prediction.dtype),
        color_box_target=color_box_target.to(device=prediction.device, dtype=prediction.dtype),
        outside_box_weight=0.0,
        per_sample=True,
    )
    return {
        "cartesian_velocity": terms["velocity"],
        "cartesian_power": terms["power"],
        "cartesian_std": terms["std"],
        "n_valid_frames": _valid_frame_count(valid_mask),
    }


def tissue_doppler_cartesian_val_metrics(model: TaskModel, sample: object, config: object) -> dict[str, Tensor]:
    typed_sample = cast(Any, sample)
    target = _required_metric_tensor(typed_sample, "cartesian_metric_doppler_target")
    prediction = _tissue_doppler_prediction(model, typed_sample, config=config).float()
    prediction = _prediction_to_cartesian_metric_grid(prediction, typed_sample, target=target)
    target = target.to(device=prediction.device, dtype=prediction.dtype)
    valid_mask = _tissue_doppler_metric_valid_mask(typed_sample, target=target)
    limit = typed_sample.sector_velocity_limit_mps
    return {
        # Headline: wrapped absolute error in m/s. The Nyquist-normalized error (the training loss) is secondary.
        "cartesian_velocity": masked_wrapped_velocity_l1(prediction, target, limit, valid_mask, per_sample=True),
        "cartesian_velocity_nyquist": masked_velocity_band_l1_loss(
            prediction, target, limit, valid_mask, per_sample=True
        ),
        "n_valid_frames": _valid_frame_count(torch.isfinite(target) & (valid_mask > 0.0)),
    }


def _tissue_doppler_metric_valid_mask(sample: Any, *, target: Tensor) -> Tensor:
    """Scan-sector mask on the Cartesian metric grid, broadcast to ``target`` (``[B,C,T,H,W]``).

    Both domains are scored on this grid, so the same mask applies to beamspace and Cartesian models. Samples
    without sector geometry have no mask and are scored on every finite target pixel.
    """
    mask = getattr(sample, "cartesian_metric_valid_mask", None)
    if not isinstance(mask, Tensor):
        return torch.ones_like(target)
    mask = mask.to(device=target.device, dtype=target.dtype)
    if tuple(mask.shape[-2:]) != tuple(target.shape[-2:]) or mask.shape[0] != target.shape[0]:
        raise ValueError(
            f"cartesian_metric_valid_mask shape {tuple(mask.shape)} does not match metric target {tuple(target.shape)}"
        )
    return torch.broadcast_to(mask, target.shape)


def _valid_frame_count(mask: Tensor) -> Tensor:
    """Number of frames per batch item with at least one valid metric pixel (``mask`` is ``[B,C,T,H,W]``)."""
    frame_valid = torch.sum(mask.float(), dim=(1, 3, 4)) > 0.0
    return frame_valid.sum(dim=1).to(dtype=torch.float32)


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


def _cartesian_soft_dice_by_item_channel(
    *,
    probs: Tensor,
    target: Tensor,
    valid_mask: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """Soft Dice per batch item and channel over all frames and pixels; validity is ``any valid pixel``."""
    valid = _segmentation_foreground_valid_mask(valid_mask, target=target).to(dtype=target.dtype, device=target.device)
    prediction = probs * valid
    target = target * valid
    reduce_dims = (2, 3, 4)
    intersection = torch.sum(prediction * target, dim=reduce_dims)
    dice_denominator = torch.sum(prediction + target, dim=reduce_dims)
    valid_pixels = torch.sum(valid, dim=reduce_dims)
    dice = (2.0 * intersection + eps) / (dice_denominator + eps)
    return dice, valid_pixels > 0.0


def _cartesian_soft_dice_by_item_channel_frame(
    *,
    probs: Tensor,
    target: Tensor,
    valid_mask: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor]:
    valid = _segmentation_foreground_valid_mask(valid_mask, target=target).to(dtype=target.dtype, device=target.device)
    prediction = probs * valid
    target = target * valid
    intersection = torch.sum(prediction * target, dim=(-2, -1))
    dice_denominator = torch.sum(prediction + target, dim=(-2, -1))
    valid_pixels = torch.sum(valid, dim=(-2, -1))
    dice = (2.0 * intersection + eps) / (dice_denominator + eps)
    return dice, valid_pixels > 0.0


def _segmentation_foreground_valid_mask(valid_mask: Tensor, *, target: Tensor) -> Tensor:
    if valid_mask.shape[1] == 1:
        return valid_mask.expand(-1, target.shape[1], -1, target.shape[-2], target.shape[-1])
    return valid_mask.expand(-1, -1, -1, target.shape[-2], target.shape[-1])
