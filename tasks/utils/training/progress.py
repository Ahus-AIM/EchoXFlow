from __future__ import annotations

import logging
import os
import sys
import time
from collections.abc import Iterable
from typing import TypeVar

from tqdm.auto import tqdm

from tasks.utils.training.runtime import _LOG_LOSS_EMA_GAMMA

SampleT = TypeVar("SampleT")
#: Seconds between progress lines when stderr is not a terminal (``ECHOXFLOW_PROGRESS_SECONDS`` overrides).
DEFAULT_PROGRESS_LOG_SECONDS = 300.0


def interactive_output() -> bool:
    """True when progress bars render on a terminal; redirected output (log files) gets periodic lines instead.

    ``ECHOXFLOW_PROGRESS_BARS=1`` or ``0`` forces progress bars on or off.
    """
    forced = os.environ.get("ECHOXFLOW_PROGRESS_BARS", "").strip().lower()
    if forced in {"1", "true", "yes", "on"}:
        return True
    if forced in {"0", "false", "no", "off"}:
        return False
    return bool(getattr(sys.stderr, "isatty", lambda: False)())


def _progress_bar(*, dataloader: Iterable[SampleT], max_steps: int | None, description: str) -> tqdm | None:
    if not interactive_output():
        return None
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


class ProgressReporter:
    """Log one progress line every ``interval_seconds`` instead of a carriage-return progress bar."""

    def __init__(
        self,
        log: logging.Logger,
        *,
        dataloader: Iterable[object],
        max_steps: int | None,
        description: str,
        interval_seconds: float | None = None,
    ) -> None:
        self.log = log
        self.total = _progress_total(dataloader, max_steps)
        self.description = description
        configured = os.environ.get("ECHOXFLOW_PROGRESS_SECONDS")
        self.interval = float(configured) if configured else float(interval_seconds or DEFAULT_PROGRESS_LOG_SECONDS)
        self.started = time.perf_counter()
        self.last = self.started

    def update(self, step: int, loss: float | None) -> None:
        now = time.perf_counter()
        if now - self.last < self.interval:
            return
        self.last = now
        elapsed = now - self.started
        rate = step / elapsed if elapsed > 0 else 0.0
        remaining = "" if self.total is None or rate <= 0 else f" eta={(self.total - step) / rate / 60.0:.1f}m"
        total = "?" if self.total is None else str(self.total)
        loss_text = "n/a" if loss is None else f"{loss:.4g}"
        self.log.info("%s step %d/%s loss~%s %.2f it/s%s", self.description, step, total, loss_text, rate, remaining)
