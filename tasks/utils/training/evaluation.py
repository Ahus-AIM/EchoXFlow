"""Batch-size-invariant evaluation: every reported metric is defined per validation clip.

Loss and metric functions return one value per batch item (NaN where a clip has no valid pixels for that metric).
:class:`ClipMetricAccumulator` keeps one row per clip, so results do not depend on the evaluation batch size or on
which clips are batched together. Batches that were zero-padded during collation are split back into their unpadded
items before the model runs, so padded frames never reach a prediction or a metric.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn

from tasks.utils.atomic import atomic_open
from tasks.utils.batching import batch_items, split_padded_batch
from tasks.utils.training.loss import shared_eval_predictions

#: Terms with this prefix are per-clip counts (for example ``n_valid_frames``): written per clip, never averaged.
COUNT_TERM_PREFIX = "n_"


@dataclass(frozen=True)
class ClipMetrics:
    """Metric values of one evaluated clip."""

    index: int
    sample_id: str
    exam_id: str
    recording_id: str
    values: dict[str, float]


@dataclass
class ClipMetricAccumulator:
    """Collects per-clip metric values and aggregates them over clips or over examinations."""

    clips: list[ClipMetrics] = field(default_factory=list)

    @property
    def n_clips(self) -> int:
        return len(self.clips)

    @property
    def n_exams(self) -> int:
        return len({clip.exam_id for clip in self.clips})

    def add(self, sample: object, terms: Mapping[str, Tensor]) -> int:
        """Add one batch of per-item ``terms`` (each shaped ``[B]``); return the number of clips added."""
        items = batch_items(sample)
        size = items.size
        values = {name: per_item_values(value, size=size, name=name) for name, value in terms.items()}
        for item_index in range(size):
            sample_id = items.sample_ids[item_index]
            record = items.records[item_index]
            self.clips.append(
                ClipMetrics(
                    index=len(self.clips),
                    sample_id=sample_id,
                    exam_id=str(getattr(record, "exam_id", None) or sample_id.split("/", 1)[0]),
                    recording_id=str(getattr(record, "recording_id", None) or sample_id.split(":", 1)[0]),
                    values={name: float(item_values[item_index]) for name, item_values in values.items()},
                )
            )
        return size

    def clip_means(self) -> dict[str, float]:
        """Mean over clips of each metric, skipping clips where the metric is undefined (NaN)."""
        return _finite_means(
            {name: [clip.values.get(name, math.nan) for clip in self.clips] for name in self.metric_names}
        )

    def exam_means(self) -> dict[str, float]:
        """Examination-equal means: average clips within each exam, then average over exams."""
        by_exam: dict[str, list[ClipMetrics]] = {}
        for clip in self.clips:
            by_exam.setdefault(clip.exam_id, []).append(clip)
        per_exam = {
            name: [
                _finite_mean([clip.values.get(name, math.nan) for clip in exam_clips])
                for exam_clips in by_exam.values()
            ]
            for name in self.metric_names
        }
        return _finite_means(per_exam)

    @property
    def names(self) -> list[str]:
        return list(dict.fromkeys(name for clip in self.clips for name in clip.values))

    @property
    def metric_names(self) -> list[str]:
        return [name for name in self.names if not name.startswith(COUNT_TERM_PREFIX)]

    def write_csv(self, path: str | Path, *, rename: Mapping[str, str] | None = None) -> Path:
        """Write one row per clip (atomically): identity columns followed by one column per metric."""
        mapping = dict(rename or {})
        names = self.names
        output = Path(path)
        with atomic_open(output, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["clip_index", "exam_id", "recording_id", "sample_id", *(mapping.get(n, n) for n in names)])
            for clip in self.clips:
                writer.writerow(
                    [
                        clip.index,
                        clip.exam_id,
                        clip.recording_id,
                        clip.sample_id,
                        *(_csv_value(clip.values.get(name, math.nan)) for name in names),
                    ]
                )
        return output


@dataclass(frozen=True)
class EvaluationResult:
    clip_means: dict[str, float]
    exam_means: dict[str, float]
    n_clips: int
    n_exams: int
    accumulator: ClipMetricAccumulator


def per_item_terms(module: nn.Module, sample: object, *, include_metrics: bool = True) -> dict[str, Tensor]:
    """Return evaluation loss and metric terms with one value per batch item.

    Padded batches are evaluated item by item on their unpadded tensors.
    """
    parts = split_padded_batch(sample)
    collected: list[tuple[int, dict[str, Tensor]]] = []
    for part in parts:
        size = batch_items(part).size
        with shared_eval_predictions():
            terms = _loss_terms(module, part)
            if include_metrics and hasattr(module, "validation_metric_terms"):
                terms.update(dict(cast(Any, module).validation_metric_terms(part) or {}))
        collected.append((size, _per_item_values_on_host(terms, size=size)))
    names = list(dict.fromkeys(name for _size, terms in collected for name in terms))
    combined: dict[str, Tensor] = {}
    for name in names:
        pieces = []
        for size, terms in collected:
            value = terms.get(name)
            pieces.append(torch.full((size,), math.nan) if value is None else value)
        combined[name] = torch.cat(pieces)
    return combined


def _per_item_values_on_host(terms: Mapping[str, object], *, size: int) -> dict[str, Tensor]:
    """``per_item_values`` of every term, copied to the host in one transfer instead of one per term."""
    if not all(isinstance(value, Tensor) for value in terms.values()):
        return {name: per_item_values(value, size=size, name=name) for name, value in terms.items()}
    tensors = {name: cast(Tensor, value).detach().float() for name, value in terms.items()}
    if not tensors:
        return {}
    host = torch.cat([value.reshape(-1) for value in tensors.values()]).cpu()
    pieces = host.split([int(value.numel()) for value in tensors.values()])
    return {
        name: per_item_values(piece.reshape(value.shape), size=size, name=name)
        for (name, value), piece in zip(tensors.items(), pieces, strict=True)
    }


def per_item_values(value: object, *, size: int, name: str) -> Tensor:
    """Return a CPU float tensor of shape ``[size]``; batch-level scalars apply to every item."""
    if not isinstance(value, Tensor):
        raise TypeError(f"evaluation term {name!r} must be a tensor")
    values = value.detach().float().cpu()
    if values.ndim == 0:
        values = values.expand(int(size)).clone()
    if tuple(values.shape) != (int(size),):
        raise ValueError(f"evaluation term {name!r} must be a scalar or have shape [{size}], got {tuple(value.shape)}")
    if bool(torch.isinf(values).any()):
        raise FloatingPointError(f"infinite evaluation term {name!r}")
    return values


def _loss_terms(module: nn.Module, sample: object) -> dict[str, Tensor]:
    if hasattr(module, "per_sample_loss_terms"):
        terms = dict(cast(Any, module).per_sample_loss_terms(sample))
    elif hasattr(module, "compute_loss_terms"):
        terms = dict(cast(Any, module).compute_loss_terms(sample))
    else:
        terms = {"loss": cast(Any, module).compute_loss(sample)}
    if not terms:
        raise ValueError("loss function must return at least one term")
    if "loss" not in terms:
        terms["loss"] = sum(terms.values(), torch.zeros(()))
    return terms


def _finite_mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return math.fsum(finite) / len(finite) if finite else math.nan


def _finite_means(values_by_name: Mapping[str, Sequence[float]]) -> dict[str, float]:
    means = {name: _finite_mean(values) for name, values in values_by_name.items()}
    return {name: value for name, value in means.items() if math.isfinite(value)}


def _csv_value(value: float) -> str:
    return "" if not math.isfinite(value) else repr(float(value))


__all__ = [
    "COUNT_TERM_PREFIX",
    "ClipMetricAccumulator",
    "ClipMetrics",
    "EvaluationResult",
    "per_item_terms",
    "per_item_values",
]
