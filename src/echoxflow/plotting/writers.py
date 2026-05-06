"""Image and video writer helpers for rendered Matplotlib output."""

from __future__ import annotations

import warnings
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def figure_to_rgb(figure: Figure) -> np.ndarray:
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
    return np.ascontiguousarray(rgba[..., :3])


def save_figure(figure: Figure, output: str | Path, *, dpi: int) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=int(dpi), facecolor=figure.get_facecolor(), bbox_inches="tight")
    return path


def write_video(output: str | Path, frames: np.ndarray, *, fps: float, source: str | Path | None = None) -> Path:
    path = Path(output)
    if path.suffix.lower() != ".mp4":
        raise ValueError(f"Unsupported video output extension `{path.suffix}`; only .mp4 is currently supported")
    av = import_module("av")
    arr = np.asarray(frames, dtype=np.uint8)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB frame stack [T,H,W,3], got shape {arr.shape}")
    if int(arr.shape[0]) == 1:
        source_text = "" if source is None else f" from {Path(source)}"
        warnings.warn(
            f"Writing single-frame MP4 {path}{source_text}; prefer an image output for static sources",
            UserWarning,
            stacklevel=2,
        )
    arr = np.stack([_pad_even(frame) for frame in arr], axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for codec in ("libx264", "h264", "libopenh264", "mpeg4"):
        try:
            _encode_with_codec(av, path, arr, fps=float(fps), codec=codec)
            return path
        except Exception as exc:
            last_error = exc
            if path.exists():
                path.unlink()
    if last_error is not None:
        raise last_error
    raise RuntimeError("Failed to encode MP4")


def _encode_with_codec(av: Any, path: Path, frames: np.ndarray, *, fps: float, codec: str) -> None:
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream(codec, rate=max(1, int(round(float(fps)))))
        stream.width = int(frames.shape[2])
        stream.height = int(frames.shape[1])
        stream.pix_fmt = "yuv444p" if codec == "libx264" else "yuv420p"
        for frame in frames:
            video_frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _pad_even(frame: np.ndarray) -> np.ndarray:
    pad_h = int(frame.shape[0]) % 2
    pad_w = int(frame.shape[1]) % 2
    if pad_h == 0 and pad_w == 0:
        return frame
    return np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
