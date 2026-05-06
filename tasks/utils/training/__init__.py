from __future__ import annotations

from tasks.utils.training.config import make_load_config
from tasks.utils.training.core import fit
from tasks.utils.training.device import (
    _autocast_context,
    _first_cuda_device_index,
    _resolve_amp_config,
    _resolve_cuda_device,
    _resolve_device,
    move_to_device,
)
from tasks.utils.training.io import (
    _color_full_region_preview_attrs,
    _common_preview_arrays,
    _encode_tissue_doppler_for_source,
    _find_sector,
    _finite_array,
    _full_region_doppler_sector,
    _native_tdi_extra_arrays,
    _preview_attrs,
    _preview_output_timestamps,
    _resize_channel_video,
    _resize_rgb_video,
    _resize_video,
    _resolve_segmentation_bmode,
    _safe_recording_id,
    _sample_scalar,
    _sample_timestamps,
    _sector_semantic_id,
    _source_bmode_array,
    _source_ecg_arrays,
    _symmetric_limit,
    _tensor_video,
    _write_preview_pair,
    _write_preview_recording_video,
    _zero_ecg_array,
)
from tasks.utils.training.logs import (
    _batch_loss_plot_figure,
    _cartesian_domain_loss_plot_figure,
    _default_run_name,
    _loss_plot_figure,
    _plottable_loss_terms,
    _prepare_batch_log,
    _prepare_epoch_log,
    _read_score_rows,
    _score_epoch,
    _score_float,
    _style_loss_axes,
    _write_batch_loss_plot,
    _write_batch_metrics,
    _write_cartesian_domain_loss_plot,
    _write_epoch_metrics,
    _write_loss_plot,
)
from tasks.utils.training.loss import (
    _accumulate_loss_terms,
    _as_scalar_tensor,
    _compute_loss_and_terms,
    _loss_terms_to_floats,
    _mean_loss_terms,
    _round_terms,
    _validate_loss,
    color_doppler_loss_terms,
    masked_color_doppler_loss,
    masked_color_doppler_loss_terms,
    masked_velocity_band_l1_loss,
    segmentation_loss_terms,
    tissue_doppler_loss_terms,
)
from tasks.utils.training.metrics import (
    color_doppler_cartesian_val_metrics,
    segmentation_val_metrics,
    tissue_doppler_cartesian_val_metrics,
)
from tasks.utils.training.module import LossFn, TrainingModule, ValMetricsFn
from tasks.utils.training.preview import (
    _color_power,
    _color_velocity,
    _random_preview_sample,
    _segmentation_foreground_video,
    _stable_preview_sample,
    _write_color_recording_previews,
    _write_epoch_previews,
    _write_parseable_epoch_previews,
    _write_segmentation_recording_previews,
    _write_tissue_recording_previews,
)
from tasks.utils.training.progress import _progress_bar, _progress_total, _should_update_progress, _update_logged_loss
from tasks.utils.training.runners import (
    doppler_result,
    evaluate_module,
    run_task_evaluation,
    run_task_training,
    run_task_training_step,
)
from tasks.utils.training.runtime import (
    _COLOR_POWER_FLOOR,
    _LOG_LOSS_EMA_GAMMA,
    _PREVIEW_FPS,
    _PREVIEW_MAX_FPS,
    _PREVIEW_STYLE,
    _PREVIEW_VIEW_MODE,
    AmpConfig,
    BatchMetrics,
    EpochMetrics,
    FitResult,
    build_optimizer,
    configure_training_logging,
)

# flake8: noqa: F401


__all__ = [name for name in globals() if not name.startswith("__")]
