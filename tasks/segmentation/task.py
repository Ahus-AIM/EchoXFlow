from __future__ import annotations

from tasks.registry import BaselineSpec, PreviewSpec, TaskSpec
from tasks.utils.training.loss import segmentation_loss_terms
from tasks.utils.training.metrics import segmentation_val_metrics
from tasks.utils.training.preview import (
    _write_segmentation_recording_previews,
    segmentation_preview_prediction,
)

TASK_SPEC = TaskSpec(
    name="segmentation",
    loss_fn=segmentation_loss_terms,
    metric_keys={"cartesian_dice_mean": "dice_mean"},
    val_metrics_fn=segmentation_val_metrics,
    preview=PreviewSpec(
        prediction_fn=segmentation_preview_prediction,
        writer=_write_segmentation_recording_previews,
    ),
    baseline=BaselineSpec(),
)
