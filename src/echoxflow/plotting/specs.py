"""Small plotting data contracts used across renderer modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from echoxflow.loading import LoadedArray

PanelKind = Literal["image", "matrix", "line"]
PlotViewMode = Literal["pre_converted", "clinical", "both"]
PanelView = Literal["pre_converted", "clinical"]


@dataclass(frozen=True)
class FrameRequest:
    time_s: float | None = None
    frame_index: int | None = None


@dataclass(frozen=True)
class PanelSpec:
    loaded: LoadedArray
    kind: PanelKind
    label: str
    view: PanelView = "pre_converted"


@dataclass(frozen=True)
class TraceSpec:
    signal: np.ndarray
    timestamps: np.ndarray
    label: str = "ECG"


@dataclass(frozen=True)
class RenderTimeline:
    timestamps: np.ndarray
    fps: float


@dataclass(frozen=True)
class RenderedFrame:
    image: np.ndarray
    time_s: float


@dataclass(frozen=True)
class VideoRequest:
    output: Path
    max_fps: float = 60.0
    dpi: int = 150
