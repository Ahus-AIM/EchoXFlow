"""Shared helpers for EchoXFlow example tasks."""

from __future__ import annotations

_MODEL_EXPORTS = {
    "ColorDopplerHead",
    "SegmentationHead",
    "TaskHead",
    "TaskModel",
    "TissueDopplerHead",
    "as_2tuple",
    "build_model",
    "norm_groups",
}


def __getattr__(name: str) -> object:
    if name in _MODEL_EXPORTS:
        from tasks.utils import models

        return getattr(models, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
