from __future__ import annotations

from tasks.registry import BaselineSpec, PreviewSpec, TaskSpec
from tasks.utils.training.loss import color_doppler_loss_terms
from tasks.utils.training.metrics import color_doppler_cartesian_val_metrics
from tasks.utils.training.preview import _write_color_recording_previews, color_doppler_preview_prediction
from tasks.utils.training.runners import doppler_result

TASK_SPEC = TaskSpec(
    name="color_doppler",
    loss_fn=color_doppler_loss_terms,
    metric_keys={
        "cartesian_velocity": "velocity_l1",
        "cartesian_power": "power_l1",
        "cartesian_std": "variation_l1",
    },
    val_metrics_fn=color_doppler_cartesian_val_metrics,
    result_factory=doppler_result,
    preview=PreviewSpec(
        prediction_fn=color_doppler_preview_prediction,
        writer=_write_color_recording_previews,
    ),
    baseline=BaselineSpec(),
)
