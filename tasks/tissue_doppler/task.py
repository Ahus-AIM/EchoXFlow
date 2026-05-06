from __future__ import annotations

from tasks.registry import BaselineSpec, PreviewSpec, TaskSpec
from tasks.utils.training.loss import tissue_doppler_loss_terms
from tasks.utils.training.metrics import tissue_doppler_cartesian_val_metrics
from tasks.utils.training.preview import _write_tissue_recording_previews, tissue_doppler_preview_prediction
from tasks.utils.training.runners import doppler_result

TASK_SPEC = TaskSpec(
    name="tissue_doppler",
    loss_fn=tissue_doppler_loss_terms,
    metric_keys={"cartesian_velocity": "velocity_l1"},
    val_metrics_fn=tissue_doppler_cartesian_val_metrics,
    result_factory=doppler_result,
    preview=PreviewSpec(
        prediction_fn=tissue_doppler_preview_prediction,
        writer=_write_tissue_recording_previews,
    ),
    baseline=BaselineSpec(),
)
