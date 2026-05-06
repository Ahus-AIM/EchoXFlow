from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from torch import Tensor, nn

from tasks.utils.models.unet import build_model

LossFn = Callable[[Any, object, object], dict[str, Tensor]]
ValMetricsFn = Callable[[Any, object, object], dict[str, Tensor]]


class TrainingModule(nn.Module):
    def __init__(
        self,
        *,
        config: object,
        loss_fn: LossFn,
        val_metrics_fn: ValMetricsFn | None = None,
    ) -> None:
        super().__init__()
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

    def validation_metric_terms(self, sample: object) -> dict[str, Tensor]:
        if self._val_metrics_fn is None:
            return {}
        return self._val_metrics_fn(self.model, sample, self.config)
