"""`RecordingStore.load_stream_header` reads what a full stream load knows, without the samples."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from echoxflow import BrightnessModeStream, ColorDopplerPowerStream, open_recording
from echoxflow.streams import validated_stream_timestamps

zarr = pytest.importorskip("zarr")


def _store(tmp_path: Path, arrays: dict[str, np.ndarray]) -> Path:
    path = tmp_path / "rec.zarr"
    group = zarr.open_group(path, mode="w", zarr_format=2)
    for name, values in arrays.items():
        group.create_array(name, data=values, chunks=values.shape if values.ndim else None)
    group.attrs["schema_version"] = "EchoXFlow-2"
    zarr.consolidate_metadata(path)
    return path


def _errors(call: object) -> tuple[str, str] | None:
    try:
        call()  # type: ignore[operator]
    except Exception as exc:  # noqa: BLE001 - the error type and message are compared
        return type(exc).__name__, str(exc)
    return None


def test_header_matches_full_brightness_mode_load(tmp_path: Path) -> None:
    frames = np.arange(6 * 4 * 3, dtype=np.uint8).reshape(6, 4, 3)
    timestamps = np.linspace(0.0, 0.5, 6).astype(np.float64)
    store = open_recording(
        _store(tmp_path, {"data/2d_brightness_mode": frames, "timestamps/2d_brightness_mode": timestamps})
    )

    full = store.load_stream("2d_brightness_mode")
    header = store.load_stream_header("2d_brightness_mode")

    assert isinstance(header.stream, BrightnessModeStream)
    assert type(header.stream) is type(full)
    assert header.shape == (6, 4, 3)
    assert header.frame_count == 6
    assert header.stream.metadata == full.metadata
    assert header.stream.data.shape == (1, 4, 3)
    np.testing.assert_array_equal(header.stream.data, full.data[:1])
    assert header.timestamps is not None and full.timestamps is not None
    assert header.timestamps.dtype == full.timestamps.dtype
    np.testing.assert_array_equal(header.timestamps, full.timestamps)


def test_header_resolves_short_names(tmp_path: Path) -> None:
    power = np.linspace(0.0, 1.0, 3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    store = open_recording(_store(tmp_path, {"derived/2d_color_doppler_power": power}))

    full = store.load_stream("data/2d_color_doppler_power")
    header = store.load_stream_header("data/2d_color_doppler_power")

    assert isinstance(header.stream, ColorDopplerPowerStream)
    assert header.stream.data_path == full.data_path
    assert header.stream.metadata == full.metadata
    assert header.shape == (3, 4, 5)
    assert header.timestamps is None


def test_header_raises_the_full_load_errors(tmp_path: Path) -> None:
    store = open_recording(
        _store(
            tmp_path,
            {
                "data/2d_brightness_mode": np.zeros((5, 2, 2), dtype=np.uint8),
                "timestamps/2d_brightness_mode": np.arange(4, dtype=np.float32),
                "data/tissue_doppler": np.zeros((5, 2, 2), dtype=np.float32),
            },
        )
    )

    cases = [
        ("2d_brightness_mode", False),  # timestamps do not match the stored length
        ("tissue_doppler", False),  # no velocity limit metadata
        ("tissue_doppler", True),
        ("2d_color_doppler_velocity", False),  # missing array
    ]
    for name, display in cases:
        full_error = _errors(
            lambda name=name, display=display: (
                store.load_modality(name).stream if display else store.load_stream(name)
            )
        )
        header_error = _errors(lambda name=name, display=display: store.load_stream_header(name, display=display))
        assert full_error is not None
        assert header_error == full_error, name


def test_header_of_an_empty_stream(tmp_path: Path) -> None:
    store = open_recording(
        _store(
            tmp_path,
            {
                "data/2d_brightness_mode": np.zeros((0, 2, 2), dtype=np.uint8),
                "timestamps/2d_brightness_mode": np.zeros((0,), dtype=np.float32),
            },
        )
    )

    header = store.load_stream_header("2d_brightness_mode")

    assert header.frame_count == 0
    assert header.shape == (0, 2, 2)
    assert header.stream.data.shape == store.load_stream("2d_brightness_mode").data.shape


def test_validated_stream_timestamps_checks_length_and_finiteness() -> None:
    values = validated_stream_timestamps("2d_brightness_mode", (3, 2, 2), np.arange(3, dtype=np.float64))
    assert values is not None and values.dtype == np.float32
    assert validated_stream_timestamps("2d_brightness_mode", (3, 2, 2), None) is None
    with pytest.raises(ValueError, match="data/2d_brightness_mode has 3 temporal samples but 2 timestamps"):
        validated_stream_timestamps("2d_brightness_mode", (3, 2, 2), np.arange(2, dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        validated_stream_timestamps("2d_brightness_mode", (2,), np.asarray([0.0, np.nan]))
