"""Fixed-length clips from catalog records as a PyTorch dataset. Requires torch, unlike the rest of echoxflow."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from echoxflow.croissant import RecordingRecord
from echoxflow.loading import open_recording
from echoxflow.scan.conversion import sector_stack_to_cartesian_array
from echoxflow.scan.geometry import sector_geometry_from_mapping


class ClipDataset(Dataset):
    """Sliding windows of ``clip_length`` frames, every ``stride`` frames, over one stream of each record.

    Windows come from the catalog frame counts, so building the dataset opens no store. ``__getitem__`` reads one
    window and returns ``(frames, timestamps, record)``: float32 frames ``[T, H, W]`` in the stream's native value
    range (optionally scan-converted to Cartesian and/or resized), float64 timestamps ``[T]`` (NaN when the stream has
    none) and the source :class:`RecordingRecord`. Use :func:`collate_clips` as ``collate_fn`` of a ``DataLoader``.
    Records without the stream, or shorter than one clip, are skipped; ``skipped`` counts them.
    """

    def __init__(
        self,
        records: Sequence[RecordingRecord],
        stream: str = "2d_brightness_mode",
        *,
        clip_length: int = 32,
        stride: int = 16,
        resize: tuple[int, int] | None = None,
        cartesian: bool = False,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        root: str | Path | None = None,
    ) -> None:
        if clip_length < 1 or stride < 1:
            raise ValueError("clip_length and stride must be positive")
        self.stream, self.clip_length, self.resize = stream, int(clip_length), resize
        self.cartesian, self.transform, self.root = cartesian, transform, root
        self.windows: list[tuple[RecordingRecord, int]] = []
        self.skipped = 0
        for record in records:
            frames = record.frame_count(stream)
            if frames is None or frames < clip_length:
                self.skipped += 1
                continue
            self.windows.extend((record, start) for start in range(0, frames - clip_length + 1, stride))

    def __len__(self) -> int:
        return len(self.windows)

    def __repr__(self) -> str:
        return (
            f"ClipDataset({len(self)} clips of {self.clip_length} {self.stream} frames, {self.skipped} records skipped)"
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, RecordingRecord]:
        record, start = self.windows[index]
        with open_recording(record, root=self.root) as store:
            stream = store.load_stream_slice(self.stream, start, start + self.clip_length)
        frames = np.asarray(stream.data, dtype=np.float32)
        if self.cartesian:
            geometry = stream.metadata.geometry or sector_geometry_from_mapping(stream.metadata.raw or {})
            frames = np.asarray(sector_stack_to_cartesian_array(frames, geometry).data, dtype=np.float32)
        clip = torch.from_numpy(np.ascontiguousarray(frames))
        if self.resize is not None:
            clip = F.interpolate(clip[:, None], size=self.resize, mode="bilinear", align_corners=False)[:, 0]
        if self.transform is not None:
            clip = self.transform(clip)
        times = np.full(len(frames), np.nan) if stream.timestamps is None else np.asarray(stream.timestamps, float)
        return clip, torch.from_numpy(times), record


def collate_clips(batch: Sequence[tuple[torch.Tensor, torch.Tensor, RecordingRecord]]) -> tuple[Any, Any, list]:
    """Stack frames and timestamps of a batch; the records stay a list."""
    frames, timestamps, records = zip(*batch, strict=True)
    return torch.stack(frames), torch.stack(timestamps), list(records)
