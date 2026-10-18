"""Per-item bookkeeping for collated task batches.

`collate_samples` concatenates dataclass samples along the batch dimension and zero-pads tensors whose shapes differ
(for example variable-length validation clips). It attaches a :class:`BatchItems` record so evaluation can recover
each item's identity and its unpadded tensors: padded frames never reach a model or a metric.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any, TypeVar, cast

import torch

SampleT = TypeVar("SampleT")

BATCH_ITEMS_ATTR = "_batch_items"


@dataclass(frozen=True)
class BatchItems:
    """Identity and pre-padding tensor shapes of every item in a collated batch."""

    sample_ids: tuple[str, ...]
    records: tuple[object, ...]
    tensor_shapes: dict[str, tuple[tuple[int, ...], ...]]

    @property
    def size(self) -> int:
        return len(self.sample_ids)

    @property
    def padded(self) -> bool:
        return any(len({shape[1:] for shape in shapes}) > 1 for shapes in self.tensor_shapes.values())


def attach_batch_items(batch: object, items: BatchItems) -> None:
    object.__setattr__(batch, BATCH_ITEMS_ATTR, items)


def batch_items(sample: object) -> BatchItems:
    """Return per-item identity for a (possibly uncollated) sample."""
    attached = getattr(sample, BATCH_ITEMS_ATTR, None)
    if isinstance(attached, BatchItems):
        return attached
    size = _leading_size(sample)
    raw_id = str(getattr(sample, "sample_id", "sample"))
    ids = raw_id.split("|") if len(raw_id.split("|")) == size else [raw_id] * size
    record = getattr(sample, "record", None)
    return BatchItems(sample_ids=tuple(ids), records=tuple(record for _ in range(size)), tensor_shapes={})


def carry_batch_items(source: object, target: SampleT) -> SampleT:
    """Copy attached batch-item bookkeeping onto a derived sample (for example after a device move)."""
    attached = getattr(source, BATCH_ITEMS_ATTR, None)
    if attached is not None and target is not source:
        object.__setattr__(target, BATCH_ITEMS_ATTR, attached)
    return target


def split_padded_batch(sample: SampleT) -> list[SampleT]:
    """Return ``[sample]`` unless it was padded during collation; then return each item unpadded."""
    items = getattr(sample, BATCH_ITEMS_ATTR, None)
    if not isinstance(items, BatchItems) or not items.padded or not is_dataclass(sample):
        return [sample]
    parts: list[SampleT] = []
    for index in range(items.size):
        updates: dict[str, object] = {"sample_id": items.sample_ids[index], "record": items.records[index]}
        for field in fields(cast(Any, sample)):
            value = getattr(sample, field.name)
            if isinstance(value, torch.Tensor) and field.name in items.tensor_shapes:
                shapes = items.tensor_shapes[field.name]
                start = sum(shape[0] for shape in shapes[:index])
                shape = shapes[index]
                updates[field.name] = value[(slice(start, start + shape[0]), *(slice(0, size) for size in shape[1:]))]
        names = {field.name for field in fields(cast(Any, sample))}
        part = replace(cast(Any, sample), **{key: value for key, value in updates.items() if key in names})
        attach_batch_items(
            part,
            BatchItems(
                sample_ids=(items.sample_ids[index],),
                records=(items.records[index],),
                tensor_shapes={name: (shapes[index],) for name, shapes in items.tensor_shapes.items()},
            ),
        )
        parts.append(cast(SampleT, part))
    return parts


def collated_items(samples: Sequence[object]) -> BatchItems:
    first = samples[0]
    shapes: dict[str, tuple[tuple[int, ...], ...]] = {}
    for field in fields(cast(Any, first)):
        values = [getattr(sample, field.name) for sample in samples]
        if isinstance(values[0], torch.Tensor):
            shapes[field.name] = tuple(tuple(int(dim) for dim in value.shape) for value in values)
    return BatchItems(
        sample_ids=tuple(str(getattr(sample, "sample_id", "sample")) for sample in samples),
        records=tuple(getattr(sample, "record", None) for sample in samples),
        tensor_shapes=shapes,
    )


def _leading_size(sample: object) -> int:
    if is_dataclass(sample):
        for field in fields(cast(Any, sample)):
            value = getattr(sample, field.name)
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                return int(value.shape[0])
    return 1


__all__ = [
    "BATCH_ITEMS_ATTR",
    "BatchItems",
    "attach_batch_items",
    "batch_items",
    "carry_batch_items",
    "collated_items",
    "split_padded_batch",
]
