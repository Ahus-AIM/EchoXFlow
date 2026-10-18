from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from tasks.utils.models.unet import TaskModel


def segmentation_loss_terms(
    model: TaskModel,
    sample: object,
    config: object,
    *,
    per_sample: bool = False,
) -> dict[str, Tensor]:
    typed_sample = cast(Any, sample)
    logits = _segmentation_prediction(model, typed_sample.frames, config=config).float()
    target = typed_sample.target_masks.to(device=logits.device, dtype=logits.dtype)
    valid_mask = _segmentation_sample_valid_mask(typed_sample, device=logits.device, dtype=logits.dtype)
    pixel_weight = (
        _segmentation_beamspace_pixel_size_weight(target)
        if _segmentation_use_beamspace_pixel_size_weighting(model=model, sample=typed_sample, config=config)
        else None
    )
    if _segmentation_dice_reduction(config) == "annotated_frame":
        dice, class_frame_valid = _segmentation_dice_by_item_channel_frame(
            logits=logits,
            target=target,
            valid_mask=valid_mask,
            eps=_dice_eps(config),
            pixel_weight=pixel_weight,
        )
        if per_sample:
            item_dice, valid_items = _segmentation_annotated_frame_dice_by_item(
                dice=dice,
                class_frame_valid=class_frame_valid,
            )
            dice_loss = _nan_where_invalid(1.0 - item_dice, valid_items)
            return {"loss": dice_loss, "dice_loss": dice_loss}
        dice_score, valid_items = _segmentation_annotated_frame_dice_score(
            dice=dice,
            class_frame_valid=class_frame_valid,
        )
        dice_loss = dice.sum() * 0.0 if not bool(torch.any(valid_items)) else 1.0 - dice_score
    else:
        dice = _segmentation_dice_by_item_channel(
            logits=logits,
            target=target,
            valid_mask=valid_mask,
            eps=_dice_eps(config),
            pixel_weight=pixel_weight,
        )
        dice_loss = (
            _segmentation_foreground_dice_loss_by_item(dice=dice, target=target, valid_mask=valid_mask)
            if per_sample
            else _segmentation_foreground_dice_loss(dice=dice, target=target, valid_mask=valid_mask)
        )
    return {"loss": dice_loss, "dice_loss": dice_loss}


_SHARED_EVAL_PREDICTIONS: ContextVar[dict[tuple[object, ...], tuple[tuple[object, ...], Tensor]] | None] = ContextVar(
    "_SHARED_EVAL_PREDICTIONS", default=None
)


@contextmanager
def shared_eval_predictions() -> Iterator[None]:
    """Share evaluation predictions between the loss and metric functions called inside this context.

    In eval mode without autograd, the model prediction for a given model and input tensors is computed once and
    reused, instead of once for the loss and again for each metric function. Training-mode or gradient-enabled
    calls are never shared.
    """
    token = _SHARED_EVAL_PREDICTIONS.set({})
    try:
        yield
    finally:
        _SHARED_EVAL_PREDICTIONS.reset(token)


def _shared_eval_prediction(
    kind: str, model: nn.Module, inputs: tuple[object, ...], compute: Callable[[], Tensor]
) -> Tensor:
    cache = _SHARED_EVAL_PREDICTIONS.get()
    if cache is None or model.training or torch.is_grad_enabled():
        return compute()
    key = (kind, id(model), *(_prediction_input_key(value) for value in inputs))
    cached = cache.get(key)
    if cached is not None:
        return cached[1]
    prediction = compute()
    # The inputs are kept alive with the entry, so their ids cannot be reused by other tensors within the context.
    cache[key] = (inputs, prediction)
    return prediction


def _prediction_input_key(value: object) -> object:
    if isinstance(value, Tensor):
        return (id(value), value._version)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return id(value)


def _segmentation_prediction(model: TaskModel, frames: Tensor, *, config: object) -> Tensor:
    return _shared_eval_prediction(
        "segmentation",
        model,
        (frames, config),
        lambda: windowed_eval_prediction(model, frames, config=config, predict=model, interval_outputs=False),
    )


EVAL_WINDOW_KEYS = ("eval_window_frames", "eval_window_overlap_frames")
_REMOVED_EVAL_KEYS = ("eval_temporal_chunk_frames", "eval_temporal_chunk_overlap_frames")


def eval_window_frames(config: object, *, interval_outputs: bool) -> tuple[int, int] | None:
    """Return the ``(window, overlap)`` input-frame counts of evaluation windowing, or ``None`` when disabled.

    The window defaults to the training clip's input length: ``data.clip_length`` frames for frame-aligned
    outputs (segmentation) and ``data.clip_length + 1`` for interval outputs (Doppler tasks predict one value per
    pair of consecutive input frames). The overlap defaults to half a window. ``data.eval_window_frames: 0``
    disables windowing.
    """
    data_config = getattr(config, "data", {})
    if not isinstance(data_config, Mapping):
        return None
    removed = [key for key in _REMOVED_EVAL_KEYS if data_config.get(key) is not None]
    if removed:
        raise ValueError(
            f"data.{removed[0]} was replaced by uniform evaluation windowing; use data.eval_window_frames and "
            "data.eval_window_overlap_frames"
        )
    configured = data_config.get("eval_window_frames")
    if configured is None:
        clip_length = data_config.get("clip_length")
        if clip_length is None:
            return None
        window = int(cast(Any, clip_length)) + (1 if interval_outputs else 0)
    else:
        window = int(cast(Any, configured))
        if window <= 0:
            return None
    minimum = 3 if interval_outputs else 2
    if window < minimum:
        raise ValueError(f"evaluation window must be at least {minimum} frames, got {window}")
    raw_overlap = data_config.get("eval_window_overlap_frames")
    overlap = window // 2 if raw_overlap is None else int(cast(Any, raw_overlap))
    # Interval outputs need one shared input frame between windows, or the interval between them is never predicted.
    minimum_overlap = 1 if interval_outputs else 0
    if not minimum_overlap <= overlap < window:
        raise ValueError(f"data.eval_window_overlap_frames must be in [{minimum_overlap}, {window}), got {overlap}")
    return window, overlap


def eval_window_starts(time_steps: int, *, window: int, overlap: int) -> list[int]:
    """Window starts covering ``time_steps`` frames: stride ``window - overlap``, last window shifted back."""
    stride = window - overlap
    starts = list(range(0, time_steps - window + 1, stride))
    final_start = time_steps - window
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    return starts


def windowed_eval_prediction(
    model: nn.Module,
    frames: Tensor,
    *,
    config: object,
    predict: Callable[[Tensor], Tensor],
    interval_outputs: bool,
    output_frames_per_step: int = 1,
) -> Tensor:
    """Evaluate every model with the same temporal context: sliding windows of the training clip length.

    In training mode, or when the clip is not longer than one window, ``predict`` runs once on the whole clip.
    Otherwise windows of ``window`` input frames start every ``window - overlap`` frames, the last window is shifted
    back so it has full length, and overlapping predictions are averaged. Interval outputs (Doppler tasks) have
    ``(frames - 1) * output_frames_per_step`` output frames and frame outputs (segmentation) one per input frame.
    """
    settings = None if model.training else eval_window_frames(config, interval_outputs=interval_outputs)
    time_steps = int(frames.shape[2])
    if settings is None or time_steps <= settings[0]:
        return predict(frames)
    window, overlap = settings
    offset = 1 if interval_outputs else 0
    factor = int(output_frames_per_step)
    window_outputs = (window - offset) * factor
    total_outputs = (time_steps - offset) * factor
    prediction_sum: Tensor | None = None
    prediction_count: Tensor | None = None
    for start in eval_window_starts(time_steps, window=window, overlap=overlap):
        prediction = predict(frames[:, :, start : start + window])
        if int(prediction.shape[2]) != window_outputs:
            raise ValueError(
                f"windowed prediction of {window} input frames produced {prediction.shape[2]} output frames, "
                f"expected {window_outputs}"
            )
        if prediction_sum is None or prediction_count is None:
            prediction_sum = prediction.new_zeros(
                (prediction.shape[0], prediction.shape[1], total_outputs, *prediction.shape[3:])
            )
            prediction_count = prediction.new_zeros((1, 1, total_outputs, *(1 for _ in prediction.shape[3:])))
        output_start = start * factor
        prediction_sum[:, :, output_start : output_start + window_outputs].add_(prediction)
        prediction_count[:, :, output_start : output_start + window_outputs].add_(1)
    if prediction_sum is None or prediction_count is None or bool(torch.any(prediction_count == 0)):
        raise RuntimeError("windowed prediction did not cover every output frame")
    return prediction_sum / prediction_count


def tissue_doppler_loss_terms(
    model: TaskModel,
    sample: object,
    config: object,
    *,
    per_sample: bool = False,
) -> dict[str, Tensor]:
    typed_sample = cast(Any, sample)
    prediction = _tissue_doppler_prediction(model, typed_sample, config=config).float()
    # Cartesian samples carry a scan-sector mask: the zero fill outside the sector is not data (see the metric).
    loss = masked_velocity_band_l1_loss(
        prediction,
        typed_sample.doppler_target,
        typed_sample.sector_velocity_limit_mps,
        getattr(typed_sample, "loss_valid_mask", None),
        per_sample=per_sample,
    )
    return {"loss": loss, "velocity": loss}


def _tissue_doppler_prediction(model: TaskModel, sample: Any, *, config: object) -> Tensor:
    scale = cast(Tensor, sample.velocity_scale_mps_per_px_frame)
    frames = cast(Tensor, sample.frames)
    return _shared_eval_prediction(
        "tissue_doppler",
        model,
        (frames, scale, config),
        lambda: windowed_eval_prediction(
            model,
            frames,
            config=config,
            predict=lambda window: cast(Tensor, model(window, velocity_scale_mps_per_px_frame=scale)),
            interval_outputs=True,
        ),
    )


def _color_doppler_prediction(model: TaskModel, sample: Any, *, config: object) -> Tensor:
    conditioning = sample.conditioning
    frames = cast(Tensor, sample.frames)
    factor = _color_doppler_temporal_upsample_factor(model, config)
    return _shared_eval_prediction(
        "color_doppler",
        model,
        (frames, conditioning, config, factor),
        lambda: windowed_eval_prediction(
            model,
            frames,
            config=config,
            predict=lambda window: cast(Tensor, model(window, conditioning=conditioning)),
            interval_outputs=True,
            output_frames_per_step=factor,
        ),
    )


def _color_doppler_temporal_upsample_factor(model: object, config: object) -> int:
    head_factor = getattr(getattr(model, "head", None), "temporal_upsample_factor", None)
    if head_factor is not None:
        return int(head_factor)
    model_config = getattr(config, "model", {})
    return int(model_config.get("temporal_upsample_factor", 2)) if isinstance(model_config, Mapping) else 2


def masked_velocity_band_l1_loss(
    prediction: Tensor,
    target: Tensor,
    sector_velocity_limit_mps: Tensor,
    valid_mask: Tensor | None = None,
    *,
    per_sample: bool = False,
) -> Tensor:
    """Nyquist-normalized periodic L1 error over valid pixels (the wrapped error divided by the velocity limit).

    Pools all pixels of the batch by default; with ``per_sample=True`` returns one value per batch item (NaN for
    items without valid pixels).
    """
    return _masked_wrapped_velocity_l1(
        prediction,
        target,
        sector_velocity_limit_mps,
        valid_mask,
        normalize=True,
        per_sample=per_sample,
    )


def masked_wrapped_velocity_l1(
    prediction: Tensor,
    target: Tensor,
    sector_velocity_limit_mps: Tensor,
    valid_mask: Tensor | None = None,
    *,
    per_sample: bool = False,
) -> Tensor:
    """Absolute periodic L1 error in m/s over valid pixels.

    The error is wrapped into ``[-limit, limit)`` like :func:`masked_velocity_band_l1_loss`, but not divided by the
    per-recording velocity limit, so recordings with different Nyquist limits are compared in the same unit.
    """
    return _masked_wrapped_velocity_l1(
        prediction,
        target,
        sector_velocity_limit_mps,
        valid_mask,
        normalize=False,
        per_sample=per_sample,
    )


def _masked_wrapped_velocity_l1(
    prediction: Tensor,
    target: Tensor,
    sector_velocity_limit_mps: Tensor,
    valid_mask: Tensor | None,
    *,
    normalize: bool,
    per_sample: bool,
) -> Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"prediction shape {tuple(prediction.shape)} must match target {tuple(target.shape)}")
    limit = sector_velocity_limit_mps.reshape(-1, 1, 1, 1, 1).to(device=prediction.device, dtype=prediction.dtype)
    period = torch.clamp(2.0 * limit, min=torch.finfo(prediction.dtype).eps)
    raw_error = prediction - target.to(device=prediction.device, dtype=prediction.dtype)
    wrapped = torch.remainder(raw_error + limit, period) - limit
    error = torch.abs(wrapped)
    if normalize:
        error = error / torch.clamp(limit, min=torch.finfo(prediction.dtype).eps)
    finite = torch.isfinite(target).to(device=prediction.device, dtype=prediction.dtype)
    mask = finite if valid_mask is None else finite * valid_mask.to(device=prediction.device, dtype=prediction.dtype)
    mask = torch.broadcast_to(mask, error.shape)
    weighted = torch.where(mask > 0.0, error, torch.zeros_like(error)) * mask
    if per_sample:
        return _masked_mean_by_item(weighted, mask)
    return torch.sum(weighted) / torch.clamp(mask.sum(), min=1.0)


def color_doppler_loss_terms(
    model: TaskModel,
    sample: object,
    config: object,
    *,
    per_sample: bool = False,
) -> dict[str, Tensor]:
    typed_sample = cast(Any, sample)
    prediction = _color_doppler_prediction(model, typed_sample, config=config).float()
    return masked_color_doppler_loss_terms(
        prediction,
        typed_sample.doppler_target,
        valid_mask=typed_sample.valid_mask,
        velocity_loss_mask=typed_sample.velocity_loss_mask,
        color_box_target=typed_sample.color_box_target,
        outside_box_weight=_loss_float(config, "outside_box_weight", 0.01),
        per_sample=per_sample,
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
    per_sample: bool = False,
) -> dict[str, Tensor]:
    """Masked L1 terms for velocity, power, and local velocity variation.

    Pools all pixels of the batch by default; with ``per_sample=True`` every term has one value per batch item
    (NaN for items without valid pixels for that term).
    """
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

    velocity_error = torch.abs(prediction[:, 0:1] - velocity_target) * velocity_mask
    power_error = torch.abs(prediction[:, 1] - power_target) * power_mask[:, 0]
    std_error = torch.abs(prediction[:, 2] - std_target) * velocity_inside_mask[:, 0] * inside_mask[:, 0]
    if per_sample:
        terms = {
            "velocity": _masked_mean_by_item(velocity_error, velocity_mask),
            "power": _masked_mean_by_item(power_error, power_mask[:, 0]),
            "std": _masked_mean_by_item(std_error, velocity_inside_mask[:, 0] * inside_mask[:, 0]),
        }
        stacked = torch.stack(tuple(terms.values()), dim=0)
        any_valid = torch.isfinite(stacked).any(dim=0)
        loss = _nan_where_invalid(torch.nan_to_num(stacked, nan=0.0).sum(dim=0), any_valid)
        return {"loss": loss, **terms}
    velocity_loss = velocity_error.sum() / velocity_count
    power_loss = power_error.sum() / power_count
    std_loss = std_error.sum() / std_count
    return {
        "loss": velocity_loss + power_loss + std_loss,
        "velocity": velocity_loss,
        "power": power_loss,
        "std": std_loss,
    }


def _masked_mean_by_item(weighted_values: Tensor, mask: Tensor) -> Tensor:
    """Return ``sum(weighted_values) / sum(mask)`` per batch item, NaN for items whose mask is empty."""
    mask = torch.broadcast_to(mask, weighted_values.shape)
    dims = tuple(range(1, weighted_values.ndim))
    count = mask.sum(dim=dims)
    return _nan_where_invalid(weighted_values.sum(dim=dims) / count.clamp_min(1e-12), count > 0.0)


def _nan_where_invalid(values: Tensor, valid: Tensor) -> Tensor:
    return torch.where(valid, values, torch.full_like(values, float("nan")))


def _dice_eps(config: object) -> float:
    loss_section = config.loss if hasattr(config, "loss") else {}
    value = loss_section.get("dice_eps", 1e-6) if isinstance(loss_section, dict) else 1e-6
    return float(value)


def _segmentation_dice_reduction(config: object, *, prefer_metrics: bool = False) -> str:
    sections: list[object] = []
    if prefer_metrics:
        sections.append(config.metrics if hasattr(config, "metrics") else {})
    sections.append(config.loss if hasattr(config, "loss") else {})
    value: object = "spatiotemporal"
    for section in sections:
        if isinstance(section, Mapping) and "dice_reduction" in section:
            value = section["dice_reduction"]
            break
    reduction = str(value).strip().lower()
    if reduction not in {"spatiotemporal", "annotated_frame"}:
        raise ValueError(f"segmentation dice_reduction must be 'spatiotemporal' or 'annotated_frame', got {value!r}")
    return reduction


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


def _segmentation_foreground_dice_loss_by_item(
    *,
    dice: Tensor,
    target: Tensor,
    valid_mask: Tensor | None,
) -> Tensor:
    """Per-item ``1 - mean Dice`` over the channels present in that item; NaN when none is present."""
    target_for_presence = target
    if valid_mask is not None:
        target_for_presence = target * _segmentation_broadcast_valid_mask(valid_mask, target=target)
    foreground = (torch.sum(target_for_presence, dim=tuple(range(2, target.ndim))) > 0.0).to(dtype=dice.dtype)
    if dice.shape != foreground.shape:
        raise ValueError(f"dice shape {tuple(dice.shape)} is incompatible with target {tuple(target.shape)}")
    present = foreground.sum(dim=1)
    mean_dice = torch.sum(dice * foreground, dim=1) / present.clamp_min(1.0)
    return _nan_where_invalid(1.0 - mean_dice, present > 0.0)


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


def _segmentation_dice_by_item_channel_frame(
    *,
    logits: Tensor,
    target: Tensor,
    valid_mask: Tensor | None,
    eps: float,
    pixel_weight: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
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
        class_frame_valid = torch.sum(valid, dim=(-2, -1)) > 0.0
    else:
        probs = _segmentation_foreground_probabilities(logits, target_channels=int(target.shape[1]))
        class_frame_valid = torch.sum(target, dim=(-2, -1)) > 0.0
    weight = None if pixel_weight is None else _segmentation_broadcast_pixel_weight(pixel_weight, target=target)
    intersection_terms = probs * target
    denominator_terms = probs + target
    if weight is not None:
        intersection_terms = intersection_terms * weight
        denominator_terms = denominator_terms * weight
    intersection = torch.sum(intersection_terms, dim=(-2, -1))
    denominator = torch.sum(denominator_terms, dim=(-2, -1))
    return (2.0 * intersection + eps) / (denominator + eps), class_frame_valid


def _segmentation_annotated_frame_dice_score(
    *,
    dice: Tensor,
    class_frame_valid: Tensor,
) -> tuple[Tensor, Tensor]:
    if dice.ndim != 3:
        raise ValueError(f"annotated-frame Dice must have shape [batch,channel,time], got {tuple(dice.shape)}")
    if class_frame_valid.shape != dice.shape:
        raise ValueError(
            f"class-frame validity shape {tuple(class_frame_valid.shape)} must match Dice {tuple(dice.shape)}"
        )
    valid = class_frame_valid.to(device=dice.device, dtype=dice.dtype)
    valid_classes_per_frame = valid.sum(dim=1)
    valid_frames = valid_classes_per_frame > 0.0
    frame_dice = torch.sum(dice * valid, dim=1) / valid_classes_per_frame.clamp_min(1.0)
    valid_frame_weights = valid_frames.to(dtype=dice.dtype)
    valid_frames_per_item = valid_frame_weights.sum(dim=1)
    valid_items = valid_frames_per_item > 0.0
    item_dice = torch.sum(frame_dice * valid_frame_weights, dim=1) / valid_frames_per_item.clamp_min(1.0)
    if not bool(torch.any(valid_items)):
        return dice.sum() * 0.0, valid_items
    return item_dice[valid_items].mean(), valid_items


def _segmentation_annotated_frame_dice_by_item(
    *,
    dice: Tensor,
    class_frame_valid: Tensor,
) -> tuple[Tensor, Tensor]:
    """Per-item annotated-frame Dice: mean over valid frames of the mean over valid classes."""
    valid = class_frame_valid.to(device=dice.device, dtype=dice.dtype)
    valid_classes_per_frame = valid.sum(dim=1)
    frame_dice = torch.sum(dice * valid, dim=1) / valid_classes_per_frame.clamp_min(1.0)
    valid_frames = (valid_classes_per_frame > 0.0).to(dtype=dice.dtype)
    frames_per_item = valid_frames.sum(dim=1)
    item_dice = torch.sum(frame_dice * valid_frames, dim=1) / frames_per_item.clamp_min(1.0)
    return item_dice, frames_per_item > 0.0


def _segmentation_annotated_frame_dice_by_item_channel(
    *,
    dice: Tensor,
    class_frame_valid: Tensor,
) -> tuple[Tensor, Tensor]:
    """Per-item, per-channel annotated-frame Dice: mean over the frames where that channel is valid."""
    valid = class_frame_valid.to(device=dice.device, dtype=dice.dtype)
    frames = valid.sum(dim=2)
    return torch.sum(dice * valid, dim=2) / frames.clamp_min(1.0), frames > 0.0


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


@dataclass(frozen=True)
class ClipLossSum:
    """Training loss of one micro-batch as a sum over its clips.

    ``loss_sum`` is the sum of the per-clip losses of the ``n_clips`` clips that have a defined loss; dividing the
    gradients accumulated over several micro-batches by their total ``n_clips`` gives the mean over clips, whatever
    the batch size and accumulation layout. ``term_sums``/``term_counts`` are detached per-term sums for logging.
    """

    loss_sum: Tensor
    n_clips: int
    term_sums: dict[str, float]
    term_counts: dict[str, int]
    #: ``loss_sum`` as a Python float, read in the same device-to-host transfer as the other statistics.
    loss_sum_value: float | None = None

    @property
    def loss_value(self) -> float:
        return float(self.loss_sum.detach().cpu()) if self.loss_sum_value is None else self.loss_sum_value

    @property
    def mean_loss(self) -> float:
        return self.loss_value / max(self.n_clips, 1)


def per_clip_loss_sum(module: nn.Module, sample: object, *, batch_size: int) -> ClipLossSum:
    """Compute the training loss per clip and sum it over the clips of the micro-batch.

    Loss functions that support ``per_sample`` give one value per clip (NaN for clips without valid pixels, which
    are left out). Loss functions without per-clip support fall back to their batch-level scalar, weighted by
    ``batch_size`` examples.
    """
    raw_terms = cast(Any, module).per_sample_loss_terms(sample) if hasattr(module, "per_sample_loss_terms") else None
    if isinstance(raw_terms, Mapping) and raw_terms:
        terms = {str(name): value for name, value in raw_terms.items()}
        if "loss" not in terms:
            terms["loss"] = sum(terms.values(), torch.zeros((), device=next(iter(terms.values())).device))
        loss = terms["loss"]
        if isinstance(loss, Tensor) and loss.ndim == 1 and int(loss.shape[0]) == int(batch_size):
            valid = torch.isfinite(loss.detach())
            loss_sum = torch.where(valid, loss, torch.zeros_like(loss)).sum()
            names = list(terms)
            values = [terms[name].detach().float().reshape(-1) for name in names]
            finite = [torch.isfinite(value) for value in values]
            # All per-step statistics in one device-to-host transfer instead of a synchronization per value. Sums of
            # finite float32 values are exact in float64; a term with some non-finite clips is summed again below
            # exactly as before (compacted), which only happens for clips without a defined loss.
            stats = torch.stack(
                [
                    torch.isinf(loss.detach()).any().double(),
                    valid.sum().double(),
                    loss_sum.detach().double(),
                    *(mask.sum().double() for mask in finite),
                    *(
                        torch.where(mask, value, torch.zeros_like(value)).sum().double()
                        for value, mask in zip(values, finite, strict=True)
                    ),
                ]
            ).tolist()
            if stats[0]:
                raise FloatingPointError("non-finite per-clip training loss")
            counts = [int(count) for count in stats[3 : 3 + len(names)]]
            sums = stats[3 + len(names) :]
            term_sums: dict[str, float] = {}
            term_counts: dict[str, int] = {}
            for name, value, mask, count, total in zip(names, values, finite, counts, sums, strict=True):
                if count == 0:
                    continue
                term_sums[name] = float(total) if count == int(value.numel()) else float(value[mask].sum().cpu())
                term_counts[name] = count
            return ClipLossSum(
                loss_sum=loss_sum,
                n_clips=int(stats[1]),
                term_sums=term_sums,
                term_counts=term_counts,
                loss_sum_value=float(stats[2]),
            )
    loss, scalar_terms = _compute_loss_and_terms(module, sample)
    _validate_loss(loss)
    count = int(batch_size)
    return ClipLossSum(
        loss_sum=loss * count,
        n_clips=count,
        term_sums={name: value * count for name, value in _loss_terms_to_floats(scalar_terms).items()},
        term_counts={name: count for name in scalar_terms},
    )


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


def _round_terms(loss_terms: Mapping[str, float]) -> dict[str, float]:
    return {name: round(float(value), 4) for name, value in sorted(loss_terms.items())}


def _validate_loss(loss: Tensor) -> None:
    if loss.ndim != 0:
        raise ValueError(f"compute_loss must return a scalar tensor, got shape {tuple(loss.shape)}")
    if not bool(torch.isfinite(loss.detach()).all()):
        raise FloatingPointError(f"non-finite loss: {float(loss.detach().cpu())}")
