from __future__ import annotations

_UNET_EXPORTS = {
    "ColorDopplerHead",
    "SegmentationHead",
    "TaskHead",
    "TaskModel",
    "TissueDopplerHead",
    "as_2tuple",
    "build_model",
    "norm_groups",
}
_TEMPORAL_MEAN_EXPORTS = {"TemporalMeanModel", "TemporalMeanModule", "fit_temporal_mean"}


def __getattr__(name: str) -> object:
    if name in _UNET_EXPORTS:
        from tasks.utils.models import unet

        return getattr(unet, name)
    if name in _TEMPORAL_MEAN_EXPORTS:
        from tasks.utils.models import temporal_mean

        return getattr(temporal_mean, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ColorDopplerHead",
    "SegmentationHead",
    "TaskHead",
    "TaskModel",
    "TemporalMeanModel",
    "TemporalMeanModule",
    "TissueDopplerHead",
    "as_2tuple",
    "build_model",
    "fit_temporal_mean",
    "norm_groups",
]
