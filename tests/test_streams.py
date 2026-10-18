import numpy as np
import pytest

from echoxflow.streams import (
    ColorDopplerVelocityStream,
    StreamMetadata,
    stream_from_arrays,
)


def test_color_doppler_velocity_requires_nyquist_limit() -> None:
    data = np.zeros((2, 4, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="velocity_limit_mps"):
        stream_from_arrays(
            data_path="data/2d_color_doppler_velocity",
            data=data,
            timestamps_path="timestamps/2d_color_doppler",
            timestamps=np.asarray([0.0, 0.1], dtype=np.float32),
            sample_rate_hz=10.0,
            metadata=StreamMetadata(data_path="data/2d_color_doppler_velocity"),
        )


def test_color_doppler_velocity_stream_uses_metadata_range() -> None:
    stream = stream_from_arrays(
        data_path="data/2d_color_doppler_velocity",
        data=np.zeros((2, 4, 4), dtype=np.float32),
        timestamps_path="timestamps/2d_color_doppler",
        timestamps=np.asarray([0.0, 0.1], dtype=np.float32),
        sample_rate_hz=10.0,
        metadata=StreamMetadata(data_path="data/2d_color_doppler_velocity", velocity_limit_mps=0.7),
    )

    assert isinstance(stream, ColorDopplerVelocityStream)
    assert stream.metadata.value_range == (-0.7, 0.7)


def test_timestamps_must_match_stream_temporal_length() -> None:
    with pytest.raises(ValueError, match="temporal samples"):
        stream_from_arrays(
            data_path="data/2d_brightness_mode",
            data=np.zeros((2, 4, 4), dtype=np.uint8),
            timestamps_path="timestamps/2d_brightness_mode",
            timestamps=np.asarray([0.0], dtype=np.float32),
            sample_rate_hz=None,
        )


def test_spectral_doppler_timestamps_use_time_first_axis() -> None:
    stream = stream_from_arrays(
        data_path="data/1d_pulsed_wave_doppler",
        data=np.zeros((5, 16), dtype=np.float32),
        timestamps_path="timestamps/1d_pulsed_wave_doppler",
        timestamps=np.linspace(0.0, 0.4, 5, dtype=np.float32),
        sample_rate_hz=10.0,
    )

    assert stream.data.shape == (5, 16)
