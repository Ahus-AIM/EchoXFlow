"""Samples whose preprocessing finishes on the training device.

With ``data.device_preprocessing`` on, a task dataset's ``__getitem__`` returns a :class:`DeferredSample`: the
cheaply loaded native arrays plus the task's finisher. Collation keeps them as a :class:`DeferredBatch`, and
:func:`tasks.utils.training.device.move_to_device` finishes each item on the target device and collates the result,
so every training, evaluation and baseline loop that already moves batches to a device needs no change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch


class SampleFinisher(Protocol):
    def finish(self, raw: Any, device: torch.device) -> Any: ...


@dataclass(frozen=True)
class DeferredSample:
    raw: Any
    finisher: SampleFinisher

    def finish(self, device: torch.device) -> Any:
        return self.finisher.finish(self.raw, device)


@dataclass(frozen=True)
class DeferredBatch:
    items: tuple[DeferredSample, ...]


def finish_on(value: DeferredSample | DeferredBatch, device: torch.device) -> Any:
    """Finish deferred preprocessing on ``device`` and return a ready (collated) sample."""
    from tasks.utils.dataset import collate_samples

    if isinstance(value, DeferredSample):
        return value.finish(device)
    return collate_samples([item.finish(device) for item in value.items])
