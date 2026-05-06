from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

from tqdm.auto import tqdm

from tasks.utils.training.runtime import _LOG_LOSS_EMA_GAMMA

SampleT = TypeVar("SampleT")


def _progress_bar(*, dataloader: Iterable[SampleT], max_steps: int | None, description: str) -> tqdm | None:
    total = _progress_total(dataloader, max_steps)
    return tqdm(total=total, desc=description, leave=False, dynamic_ncols=True)


def _progress_total(dataloader: Iterable[SampleT], max_steps: int | None) -> int | None:
    try:
        dataloader_length = len(dataloader)  # type: ignore[arg-type]
    except TypeError:
        dataloader_length = None
    if max_steps is None:
        return dataloader_length
    if dataloader_length is None:
        return int(max_steps)
    return min(int(max_steps), int(dataloader_length))


def _should_update_progress(step: int, log_every_steps: int) -> bool:
    interval = max(1, int(log_every_steps))
    return step == 1 or step % interval == 0


def _update_logged_loss(previous: float | None, current: float) -> float:
    if previous is None:
        return current
    return (_LOG_LOSS_EMA_GAMMA * previous) + ((1.0 - _LOG_LOSS_EMA_GAMMA) * current)
