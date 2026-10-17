from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import lru_cache
from typing import Any, cast

from torch import Tensor, nn

LossFn = Callable[[Any, object, object], dict[str, Tensor]]
ValMetricsFn = Callable[[Any, object, object], dict[str, Tensor]]


def call_per_sample_loss_fn(loss_fn: LossFn, model: object, sample: object, config: object) -> dict[str, Tensor]:
    """Call ``loss_fn`` for evaluation, requesting one value per batch item when it supports ``per_sample``.

    Loss functions without a ``per_sample`` keyword return batch-level scalars, which the evaluator then attributes
    to every item of the batch.
    """
    if _accepts_per_sample(loss_fn):
        return cast(dict[str, Tensor], cast(Any, loss_fn)(model, sample, config, per_sample=True))
    return loss_fn(model, sample, config)


@lru_cache(maxsize=None)
def _accepts_per_sample(function: Callable[..., object]) -> bool:
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return False
    return "per_sample" in parameters or any(item.kind is item.VAR_KEYWORD for item in parameters.values())


class TrainingModule(nn.Module):
    def __init__(
        self,
        *,
        config: object,
        loss_fn: LossFn,
        val_metrics_fn: ValMetricsFn | None = None,
    ) -> None:
        super().__init__()
        from tasks.utils.models.unet import build_model

        self.config = config
        self.model = build_model(**config.model)  # type: ignore[attr-defined]
        self._loss_fn = loss_fn
        self._val_metrics_fn = val_metrics_fn

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        if len(args) == 2 and not kwargs:
            frames, extra = args
            extra_kwarg = getattr(getattr(self.model, "head", None), "extra_kwarg", None)
            if extra_kwarg:
                return cast(Tensor, self.model(frames, **{str(extra_kwarg): extra}).float())
            return cast(Tensor, self.model(frames).float())
        return cast(Tensor, self.model(*args, **kwargs).float())

    def compute_loss(self, sample: object) -> Tensor:
        return self.compute_loss_terms(sample)["loss"]

    def compute_loss_terms(self, sample: object) -> dict[str, Tensor]:
        return self._loss_fn(self.model, sample, self.config)

    def per_sample_loss_terms(self, sample: object) -> dict[str, Tensor]:
        return call_per_sample_loss_fn(self._loss_fn, self.model, sample, self.config)

    def validation_metric_terms(self, sample: object) -> dict[str, Tensor]:
        if self._val_metrics_fn is None:
            return {}
        return self._val_metrics_fn(self.model, sample, self.config)
