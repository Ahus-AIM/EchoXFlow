"""Image and video writer helpers for rendered Matplotlib output."""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Iterable, Iterator
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
    figure.savefig(path, dpi=int(dpi), facecolor=figure.get_facecolor())
    return path


def write_video(
    output: str | Path,
    frames: np.ndarray | Iterable[np.ndarray],
    *,
    fps: float,
    source: str | Path | None = None,
) -> Path:
    """Encode RGB frames to an MP4 (H.264) or WebM (VP9) video, or an animated WebP.

    ``frames`` is either an array [T,H,W,3] or an iterable of [H,W,3] frames.
    Iterables are encoded as they arrive and never held in memory in full, except for WebP, which is written in one
    go (GitHub plays animated WebP in an ``<img>`` tag but not relative ``<video>`` sources).
    """
    path = Path(output)
    if path.suffix.lower() == ".webp":
        return _write_webp(path, frames, fps=fps)
    codecs = _CODECS.get(path.suffix.lower())
    if codecs is None:
        raise ValueError(f"Unsupported video output extension `{path.suffix}`; use .mp4, .webm or .webp")
    av = import_module("av")
    if isinstance(frames, np.ndarray) and (frames.ndim != 4 or frames.shape[-1] != 3):
        raise ValueError(f"Expected RGB frame stack [T,H,W,3], got shape {frames.shape}")
    iterator = iter(frames)
    first = next(iterator, None)
    if first is None:
        raise ValueError("Expected at least one RGB frame, got none")
    first = _pad_even(_checked_frame(first))
    second = next(iterator, None)
    if second is None:
        source_text = "" if source is None else f" from {Path(source)}"
        warnings.warn(
            f"Writing single-frame MP4 {path}{source_text}; prefer an image output for static sources",
            UserWarning,
            stacklevel=2,
        )
    rest = iterator if second is None else itertools.chain((second,), iterator)
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for codec in codecs:
        try:
            _encode_with_codec(av, path, first, rest, fps=float(fps), codec=codec)
            return path
        except _CodecSetupError as exc:
            last_error = exc.__cause__ if isinstance(exc.__cause__, Exception) else exc
            if path.exists():
                path.unlink()
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to encode {path.suffix}")


_CODECS = {".mp4": ("libx264", "h264", "libopenh264", "mpeg4"), ".webm": ("libvpx-vp9", "libvpx")}
_CODEC_OPTIONS = {
    "libx264": {"preset": "veryfast", "crf": "20"},
    # Constant quality: libvpx uses a target bitrate unless it is zero.
    "libvpx-vp9": {"crf": "30", "b:v": "0", "row-mt": "1", "cpu-used": "2"},
    "libvpx": {"crf": "10", "b:v": "0"},
}


def _write_webp(path: Path, frames: np.ndarray | Iterable[np.ndarray], *, fps: float) -> Path:
    from PIL import Image

    images = [Image.fromarray(_checked_frame(frame)) for frame in frames]
    if not images:
        raise ValueError("Expected at least one RGB frame, got none")
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=max(1, int(round(1000.0 / float(fps)))),
        loop=0,
        quality=65,  # README-sized clips: a third smaller than the default 80 with no visible change
        method=6,  # slowest, smallest encoding
    )
    return path


class _CodecSetupError(Exception):
    """The codec could not be opened or could not encode the first frame."""


def _checked_frame(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB frame [H,W,3], got shape {arr.shape}")
    return arr


def _encode_with_codec(
    av: Any, path: Path, first: np.ndarray, rest: Iterator[np.ndarray], *, fps: float, codec: str
) -> None:
    # Only the first frame is retried with another codec: later frames come from a
    # one-shot iterator, so errors after the codec has accepted a frame propagate.
    height, width = int(first.shape[0]), int(first.shape[1])
    with av.open(str(path), mode="w") as container:
        try:
            stream = container.add_stream(codec, rate=max(1, int(round(float(fps)))))
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            if codec in _CODEC_OPTIONS:
                stream.options = dict(_CODEC_OPTIONS[codec])
            _encode_frame(av, container, stream, first)
        except Exception as exc:
            raise _CodecSetupError(str(exc)) from exc
        for frame in rest:
            padded = _pad_even(_checked_frame(frame))
            if padded.shape[:2] != (height, width):
                raise ValueError(f"Frame size {padded.shape[:2]} differs from first frame size {(height, width)}")
            _encode_frame(av, container, stream, padded)
        for packet in stream.encode():
            container.mux(packet)


def _encode_frame(av: Any, container: Any, stream: Any, frame: np.ndarray) -> None:
    video_frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")
    for packet in stream.encode(video_frame):
        container.mux(packet)


def _pad_even(frame: np.ndarray) -> np.ndarray:
    pad_h = int(frame.shape[0]) % 2
    pad_w = int(frame.shape[1]) % 2
    if pad_h == 0 and pad_w == 0:
        return frame
    return np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
