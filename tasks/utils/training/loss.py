from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from tasks.utils.models.unet import TaskModel


def segmentation_loss_terms(model: TaskModel, sample: object, config: object) -> dict[str, Tensor]:
    typed_sample = cast(Any, sample)
    logits = cast(Tensor, model(typed_sample.frames).float())
    target = typed_sample.target_masks.to(device=logits.device, dtype=logits.dtype)
    valid_mask = _segmentation_sample_valid_mask(typed_sample, device=logits.device, dtype=logits.dtype)
    pixel_weight = (
        _segmentation_beamspace_pixel_size_weight(target)
        if _segmentation_use_beamspace_pixel_size_weighting(model=model, sample=typed_sample, config=config)
        else None
    )
    dice = _segmentation_dice_by_item_channel(
        logits=logits,
        target=target,
        valid_mask=valid_mask,
        eps=_dice_eps(config),
        pixel_weight=pixel_weight,
    )
    dice_loss = _segmentation_foreground_dice_loss(dice=dice, target=target, valid_mask=valid_mask)
    return {"loss": dice_loss, "dice_loss": dice_loss}


def tissue_doppler_loss_terms(model: TaskModel, sample: object, config: object) -> dict[str, Tensor]:
    del config
    typed_sample = cast(Any, sample)
    prediction = cast(
        Tensor,
        model(
            typed_sample.frames,
            velocity_scale_mps_per_px_frame=typed_sample.velocity_scale_mps_per_px_frame,
        ).float(),
    )
    loss = masked_velocity_band_l1_loss(
        prediction,
        typed_sample.doppler_target,
        typed_sample.sector_velocity_limit_mps,
    )
    return {"loss": loss, "velocity": loss}


def masked_velocity_band_l1_loss(
    prediction: Tensor,
    target: Tensor,
    sector_velocity_limit_mps: Tensor,
    valid_mask: Tensor | None = None,
) -> Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"prediction shape {tuple(prediction.shape)} must match target {tuple(target.shape)}")
    limit = sector_velocity_limit_mps.reshape(-1, 1, 1, 1, 1).to(device=prediction.device, dtype=prediction.dtype)
    period = torch.clamp(2.0 * limit, min=torch.finfo(prediction.dtype).eps)
    raw_error = prediction - target.to(device=prediction.device, dtype=prediction.dtype)
    wrapped = torch.remainder(raw_error + limit, period) - limit
    normalized = torch.abs(wrapped) / torch.clamp(limit, min=torch.finfo(prediction.dtype).eps)
    finite = torch.isfinite(target).to(device=prediction.device, dtype=prediction.dtype)
    mask = finite if valid_mask is None else finite * valid_mask.to(device=prediction.device, dtype=prediction.dtype)
    return torch.sum(normalized * mask) / torch.clamp(mask.sum(), min=1.0)


def color_doppler_loss_terms(model: TaskModel, sample: object, config: object) -> dict[str, Tensor]:
    typed_sample = cast(Any, sample)
    prediction = cast(Tensor, model(typed_sample.frames, conditioning=typed_sample.conditioning).float())
    return masked_color_doppler_loss_terms(
        prediction,
        typed_sample.doppler_target,
        valid_mask=typed_sample.valid_mask,
        velocity_loss_mask=typed_sample.velocity_loss_mask,
        color_box_target=typed_sample.color_box_target,
        outside_box_weight=_loss_float(config, "outside_box_weight", 0.01),
    )


def masked_color_doppler_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    valid_mask: Tensor,
    velocity_loss_mask: Tensor,
    color_box_target: Tensor | None = None,
    velocity_limit_mps: Tensor | None = None,
    outside_box_weight: float = 0.01,
) -> Tensor:
    return masked_color_doppler_loss_terms(
        prediction,
        target,
        valid_mask=valid_mask,
        velocity_loss_mask=velocity_loss_mask,
        color_box_target=color_box_target,
        velocity_limit_mps=velocity_limit_mps,
        outside_box_weight=outside_box_weight,
    )["loss"]


def masked_color_doppler_loss_terms(
    prediction: Tensor,
    target: Tensor,
    *,
    valid_mask: Tensor,
    velocity_loss_mask: Tensor,
    color_box_target: Tensor | None = None,
    velocity_limit_mps: Tensor | None = None,
    outside_box_weight: float = 0.01,
) -> dict[str, Tensor]:
    del velocity_limit_mps
    if prediction.shape[0] != target.shape[0] or prediction.shape[2:] != target.shape[2:]:
        raise ValueError("prediction and target batch/time/spatial dimensions must match for Color Doppler loss")
    if prediction.shape[1] != 3:
        raise ValueError(f"Color Doppler expects three output channels, got {prediction.shape[1]}")
    if target.shape[1] != 3:
        raise ValueError(f"Color Doppler target expects three channels, got {target.shape[1]}")
    if valid_mask.shape[0] != prediction.shape[0] or valid_mask.shape[2:] != prediction.shape[2:]:
        raise ValueError("valid_mask shape must match prediction batch/time/spatial dimensions")
    if velocity_loss_mask.shape != valid_mask.shape:
        raise ValueError("velocity_loss_mask shape must match valid_mask")
    mask = valid_mask.to(device=prediction.device, dtype=prediction.dtype).clamp(0.0, 1.0)
    finite = torch.isfinite(target).all(dim=1, keepdim=True).to(dtype=mask.dtype)
    mask = mask * finite

    if color_box_target is None:
        inside_mask = mask
        outside_mask = torch.zeros_like(mask)
    else:
        if color_box_target.shape != valid_mask.shape:
            raise ValueError("color_box_target shape must match valid_mask")
        color_box_mask = color_box_target.to(device=prediction.device, dtype=prediction.dtype).clamp(0.0, 1.0)
        color_box_finite = torch.isfinite(color_box_target.to(device=prediction.device)).to(dtype=prediction.dtype)
        inside_mask = mask * color_box_mask * color_box_finite
        outside_mask = mask * (1.0 - color_box_mask) * color_box_finite

    velocity_inside_mask = velocity_loss_mask.to(device=prediction.device, dtype=prediction.dtype).clamp(0.0, 1.0)
    velocity_mask = (velocity_inside_mask * inside_mask) + (float(outside_box_weight) * outside_mask)
    power_mask = inside_mask + (float(outside_box_weight) * outside_mask)
    velocity_count = velocity_mask.sum().clamp_min(1.0)
    power_count = power_mask.sum().clamp_min(1.0)
    std_count = (velocity_inside_mask * inside_mask).sum().clamp_min(1.0)

    velocity_target = torch.where(
        (inside_mask > 0.0) & torch.isfinite(target[:, 0:1]), target[:, 0:1], torch.zeros_like(target[:, 0:1])
    )
    power_target = torch.where(
        (inside_mask[:, 0] > 0.0) & torch.isfinite(target[:, 1]), target[:, 1], torch.zeros_like(target[:, 1])
    )
    std_target = torch.where(torch.isfinite(target[:, 2]), target[:, 2], prediction[:, 2].detach())

    velocity_loss = (torch.abs(prediction[:, 0:1] - velocity_target) * velocity_mask).sum() / velocity_count
    power_loss = (torch.abs(prediction[:, 1] - power_target) * power_mask[:, 0]).sum() / power_count
    std_loss = (
        torch.abs(prediction[:, 2] - std_target) * velocity_inside_mask[:, 0] * inside_mask[:, 0]
    ).sum() / std_count
    return {
        "loss": velocity_loss + power_loss + std_loss,
        "velocity": velocity_loss,
        "power": power_loss,
        "std": std_loss,
    }


def _dice_eps(config: object) -> float:
    loss_section = config.loss if hasattr(config, "loss") else {}
    value = loss_section.get("dice_eps", 1e-6) if isinstance(loss_section, dict) else 1e-6
    return float(value)


def _segmentation_sample_valid_mask(sample: Any, *, device: torch.device, dtype: torch.dtype) -> Tensor | None:
    raw = sample.target_mask_valid if sample.target_mask_valid is not None else sample.valid_mask
    return None if raw is None else raw.to(device=device, dtype=dtype)


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


def _segmentation_foreground_dice_loss(*, dice: Tensor, target: Tensor, valid_mask: Tensor | None) -> Tensor:
    target_for_presence = target
    if valid_mask is not None:
        target_for_presence = target * _segmentation_broadcast_valid_mask(valid_mask, target=target)
    reduce_dims = tuple(range(2, target.ndim))
    foreground = torch.sum(target_for_presence, dim=reduce_dims) > 0.0
    if dice.ndim == 1:
        foreground = torch.any(foreground, dim=0)
    elif dice.shape != foreground.shape:
        raise ValueError(f"dice shape {tuple(dice.shape)} is incompatible with target {tuple(target.shape)}")
    if not bool(torch.any(foreground)):
        return dice.sum() * 0.0
    return 1.0 - dice[foreground].mean()


def _segmentation_dice_by_item_channel(
    *,
    logits: Tensor,
    target: Tensor,
    valid_mask: Tensor | None,
    eps: float,
    pixel_weight: Tensor | None = None,
) -> Tensor:
    if valid_mask is not None:
        valid = _segmentation_foreground_valid_mask(valid_mask, target=target).to(
            dtype=target.dtype, device=target.device
        )
        probs = _segmentation_foreground_probabilities(
            logits,
            target_channels=int(target.shape[1]),
            valid_foreground_mask=valid,
        )
        probs = probs * valid
        target = target * valid
    else:
        probs = _segmentation_foreground_probabilities(logits, target_channels=int(target.shape[1]))
    weight = None if pixel_weight is None else _segmentation_broadcast_pixel_weight(pixel_weight, target=target)
    reduce_dims = tuple(range(2, target.ndim))
    intersection_terms = probs * target
    denominator_terms = probs + target
    if weight is not None:
        intersection_terms = intersection_terms * weight
        denominator_terms = denominator_terms * weight
    intersection = torch.sum(intersection_terms, dim=reduce_dims)
    denominator = torch.sum(denominator_terms, dim=reduce_dims)
    return (2.0 * intersection + eps) / (denominator + eps)


def _segmentation_broadcast_valid_mask(valid_mask: Tensor, *, target: Tensor) -> Tensor:
    return torch.broadcast_to(valid_mask, target.shape).to(dtype=target.dtype, device=target.device)


def _segmentation_broadcast_pixel_weight(pixel_weight: Tensor, *, target: Tensor) -> Tensor:
    try:
        return torch.broadcast_to(pixel_weight.to(dtype=target.dtype, device=target.device), target.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"pixel_weight shape {tuple(pixel_weight.shape)} is incompatible with target {tuple(target.shape)}"
        ) from exc


def _segmentation_beamspace_pixel_size_weight(target: Tensor) -> Tensor:
    height = int(target.shape[-2])
    if height <= 1:
        rows = torch.ones((height,), dtype=target.dtype, device=target.device)
    else:
        rows = torch.linspace(0.001, 1.999, steps=height, dtype=target.dtype, device=target.device)
    return rows.reshape(1, 1, 1, height, 1)


def _segmentation_use_beamspace_pixel_size_weighting(*, model: object, sample: Any, config: object) -> bool:
    if not _loss_bool(config, "beamspace_pixel_size_weighting", False):
        return False
    if not bool(getattr(model, "training", False)):
        return False
    return str(getattr(sample, "coordinate_space", "")).strip().lower() == "beamspace"


def _segmentation_foreground_valid_mask(valid_mask: Tensor, *, target: Tensor) -> Tensor:
    if valid_mask.shape[1] == 1:
        return valid_mask.expand(-1, target.shape[1], -1, target.shape[-2], target.shape[-1])
    return valid_mask.expand(-1, -1, -1, target.shape[-2], target.shape[-1])


def _loss_bool(config: object, key: str, default: bool) -> bool:
    loss_section = config.loss if hasattr(config, "loss") else {}
    value = loss_section.get(key, default) if isinstance(loss_section, dict) else default
    return bool(value)


def _loss_float(config: object, key: str, default: float) -> float:
    loss_section = config.loss if hasattr(config, "loss") else {}
    value = loss_section.get(key, default) if isinstance(loss_section, dict) else default
    return float(value)


def _compute_loss_and_terms(module: nn.Module, sample: object) -> tuple[Tensor, Mapping[str, Tensor]]:
    if hasattr(module, "compute_loss_terms"):
        raw_terms = cast(Any, module).compute_loss_terms(sample)
        if not isinstance(raw_terms, Mapping):
            raise TypeError("compute_loss_terms must return a mapping of loss term names to scalar tensors")
        terms = {str(key): _as_scalar_tensor(value, label=str(key)) for key, value in raw_terms.items()}
        if not terms:
            raise ValueError("compute_loss_terms must return at least one loss term")
        loss: Tensor | None = terms.get("loss")
        if loss is None:
            loss = sum(terms.values(), torch.zeros((), device=next(iter(terms.values())).device))
            terms["loss"] = loss
        return loss, terms
    loss = _as_scalar_tensor(cast(Any, module).compute_loss(sample), label="loss")
    return loss, {"loss": loss}


def _as_scalar_tensor(value: object, *, label: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"loss term {label!r} must be a tensor")
    _validate_loss(value)
    return value


def _loss_terms_to_floats(loss_terms: Mapping[str, Tensor]) -> dict[str, float]:
    return {name: float(value.detach().cpu()) for name, value in loss_terms.items()}


def _accumulate_loss_terms(total_terms: dict[str, float], loss_terms: Mapping[str, float]) -> None:
    for name, value in loss_terms.items():
        total_terms[name] = total_terms.get(name, 0.0) + float(value)


def _mean_loss_terms(total_terms: Mapping[str, float], steps: int) -> dict[str, float]:
    divisor = max(int(steps), 1)
    return {name: float(value) / divisor for name, value in total_terms.items()}


def _round_terms(loss_terms: Mapping[str, float]) -> dict[str, float]:
    return {name: round(float(value), 4) for name, value in sorted(loss_terms.items())}


def _validate_loss(loss: Tensor) -> None:
    if loss.ndim != 0:
        raise ValueError(f"compute_loss must return a scalar tensor, got shape {tuple(loss.shape)}")
    if not bool(torch.isfinite(loss.detach()).all()):
        raise FloatingPointError(f"non-finite loss: {float(loss.detach().cpu())}")
