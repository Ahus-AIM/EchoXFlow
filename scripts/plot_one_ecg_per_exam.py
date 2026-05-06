#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from numcodecs import get_codec

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class SelectedEcg:
    exam_id: str
    recording_id: str
    zarr_path: str
    sample_count: int


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot one raw ECG signal per exam from an EchoXFlow Croissant export.")
    parser.add_argument("croissant", type=Path, help="Path to croissant.json.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ecg_per_exam_pngs"))
    parser.add_argument("--max-points", type=int, default=2200)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = select_one_ecg_per_exam(args.croissant)
    write_plots(selected, root=args.croissant.expanduser().parent, output_dir=output_dir, max_points=args.max_points)
    print(output_dir)
    print(f"wrote {len(selected)} ECG PNGs")
    return 0


def select_one_ecg_per_exam(croissant_path: Path) -> list[SelectedEcg]:
    metadata = json.loads(croissant_path.expanduser().read_text(encoding="utf-8"))
    recordings = _record_set_data(metadata, "recordings")
    ecg_counts = _ecg_sample_counts(_record_set_data(metadata, "arrays"))
    by_exam: dict[str, SelectedEcg] = {}
    for row in recordings:
        array_paths = _string_list(row.get("recordings/array_paths"))
        if "data/ecg" not in array_paths:
            continue
        exam_id = _clean(row.get("recordings/exam_id"))
        recording_id = _clean(row.get("recordings/recording_id"))
        zarr_path = _clean(row.get("recordings/zarr_path"))
        if not exam_id or not recording_id or not zarr_path:
            continue
        sample_count = int(ecg_counts.get(recording_id, 0))
        candidate = SelectedEcg(
            exam_id=exam_id,
            recording_id=recording_id,
            zarr_path=zarr_path,
            sample_count=sample_count,
        )
        current = by_exam.get(exam_id)
        if current is None or _selection_key(candidate) > _selection_key(current):
            by_exam[exam_id] = candidate
    return sorted(by_exam.values(), key=lambda item: item.exam_id)


def write_plots(selected: list[SelectedEcg], *, root: Path, output_dir: Path, max_points: int) -> None:
    index_path = output_dir / "index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("filename", "exam_id", "recording_id", "zarr_path", "sample_count", "duration_s"),
        )
        writer.writeheader()
        for index, item in enumerate(selected, start=1):
            output = output_dir / f"{index:04d}_{_slug(item.exam_id)}.png"
            recording_path = root / item.zarr_path
            signal, timestamps = _load_ecg(recording_path)
            _write_plot(
                signal=signal,
                timestamps=timestamps,
                output=output,
                title=f"{item.exam_id}  {item.recording_id}",
                max_points=max_points,
            )
            duration_s = _duration_s(timestamps)
            writer.writerow(
                {
                    "filename": output.name,
                    "exam_id": item.exam_id,
                    "recording_id": item.recording_id,
                    "zarr_path": item.zarr_path,
                    "sample_count": int(signal.size),
                    "duration_s": "" if duration_s is None else f"{duration_s:.6g}",
                }
            )


def _record_set_data(metadata: dict[str, Any], name: str) -> list[dict[str, Any]]:
    record_sets = metadata.get("recordSet")
    if not isinstance(record_sets, list):
        return []
    for record_set in record_sets:
        if not isinstance(record_set, dict):
            continue
        if str(record_set.get("@id", "")).strip() == name or str(record_set.get("name", "")).strip() == name:
            data = record_set.get("data")
            return [row for row in data if isinstance(row, dict)] if isinstance(data, list) else []
    return []


def _ecg_sample_counts(array_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in array_rows:
        if _clean(row.get("arrays/array_path")) != "data/ecg":
            continue
        shape = row.get("arrays/shape")
        if not isinstance(shape, list) or not shape:
            continue
        recording_id = _clean(row.get("arrays/recording_id"))
        if recording_id:
            counts[recording_id] = int(shape[0])
    return counts


def _selection_key(item: SelectedEcg) -> tuple[int, str]:
    return (int(item.sample_count), item.recording_id)


def _load_ecg(recording_path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    signal = _load_zarr_1d(recording_path / "data" / "ecg").astype(np.float32, copy=False)
    timestamp_path = recording_path / "timestamps" / "ecg"
    timestamps = _load_zarr_1d(timestamp_path).astype(np.float64, copy=False) if timestamp_path.exists() else None
    if timestamps is not None and int(timestamps.size) != int(signal.size):
        timestamps = None
    return signal, timestamps


def _load_zarr_1d(array_path: Path) -> np.ndarray:
    metadata = json.loads((array_path / ".zarray").read_text(encoding="utf-8"))
    shape = tuple(int(size) for size in metadata["shape"])
    if len(shape) != 1:
        raise ValueError(f"Expected one-dimensional ECG array at {array_path}, got shape {shape}")
    dtype = np.dtype(str(metadata["dtype"]))
    count = shape[0]
    output = np.empty(count, dtype=dtype)
    chunks = tuple(int(size) for size in metadata["chunks"])
    chunk_size = chunks[0]
    compressor = metadata.get("compressor")
    codec = None if compressor is None else get_codec(compressor)
    for chunk_index in range(int(math.ceil(count / chunk_size))):
        chunk_file = array_path / str(chunk_index)
        raw = chunk_file.read_bytes()
        decoded = raw if codec is None else codec.decode(raw)
        chunk = np.frombuffer(decoded, dtype=dtype)
        start = chunk_index * chunk_size
        stop = min(count, start + chunk_size)
        output[start:stop] = chunk[: stop - start]
    return output.reshape(-1)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_clean(item) for item in value if _clean(item)]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _write_plot(
    *,
    signal: np.ndarray,
    timestamps: np.ndarray | None,
    output: Path,
    title: str,
    max_points: int,
) -> None:
    x_values = np.arange(int(signal.size), dtype=np.float64) if timestamps is None else timestamps
    x_values, signal = _downsample(x_values, signal, max_points=max_points)
    figure, ax = plt.subplots(figsize=(4.8, 1.35), dpi=120)
    ax.plot(x_values, signal, color="#14883B", linewidth=0.7)
    ax.set_title(title, fontsize=6)
    ax.set_xlabel("sample" if timestamps is None else "time (s)", fontsize=6)
    ax.set_ylabel("ECG", fontsize=6)
    ax.tick_params(axis="both", labelsize=5, length=2)
    ax.grid(True, color="#D9D9D9", linewidth=0.4)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    figure.tight_layout(pad=0.4)
    figure.savefig(output)
    plt.close(figure)


def _downsample(x_values: np.ndarray, signal: np.ndarray, *, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or int(signal.size) <= max_points:
        return x_values, signal
    indices = np.linspace(0, int(signal.size) - 1, int(max_points)).astype(np.int64)
    return x_values[indices], signal[indices]


def _duration_s(timestamps: np.ndarray | None) -> float | None:
    if timestamps is None or int(timestamps.size) < 2:
        return None
    return float(timestamps[-1] - timestamps[0])


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "exam"


if __name__ == "__main__":
    raise SystemExit(main())
