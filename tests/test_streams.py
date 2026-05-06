import numpy as np
import pytest

from echoxflow.streams import (
    ColorDopplerVelocityStream,
    StreamMetadata,
    TissueDopplerFloatStream,
    TissueDopplerRawStream,
    encode_tdi_linear_uint8_codes,
    encode_tdi_native_codes,
    render_tdi_linear_uint8_codes,
    render_tdi_native_codes,
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


def test_native_tissue_doppler_stream_casts_to_float_with_fenc_table() -> None:
    raw_codes = np.asarray([[[0, 1]]], dtype=np.uint16)
    fenc_table = np.zeros((256 * 256, 4), dtype=np.uint8)
    stream = stream_from_arrays(
        data_path="data/tissue_doppler",
        data=raw_codes,
        timestamps_path="timestamps/tissue_doppler",
        timestamps=np.asarray([0.0], dtype=np.float32),
        sample_rate_hz=30.0,
        metadata=StreamMetadata(
            data_path="data/tissue_doppler",
            velocity_limit_mps=0.2,
            storage_encoding="ge_tdi_raw_u16",
            fenc_table_path="data/tissue_doppler_fenc_table",
        ),
    )

    assert isinstance(stream, TissueDopplerRawStream)
    converted = stream.to_float(fenc_table)
    assert isinstance(converted, TissueDopplerFloatStream)
    assert converted.data.dtype == np.float32
    assert converted.metadata.value_range == (-0.2, 0.2)


def test_tissue_doppler_velocity_encoders_round_trip() -> None:
    fenc_table = np.zeros((256 * 256, 4), dtype=np.uint8)
    fenc_table[:, 0] = np.arange(256 * 256, dtype=np.uint32).astype(np.uint8)
    values = np.asarray([[[-0.2, 0.0, 0.2]]], dtype=np.float32)

    native = encode_tdi_native_codes(values, fenc_table=fenc_table, velocity_scale_mps=0.2)
    linear = encode_tdi_linear_uint8_codes(values, velocity_scale_mps=0.2)

    assert native.dtype == np.uint16
    assert linear.dtype == np.uint8
    assert render_tdi_native_codes(native, fenc_table=fenc_table, velocity_scale_mps=0.2) == pytest.approx(
        values, abs=0.002
    )
    assert render_tdi_linear_uint8_codes(linear, velocity_scale_mps=0.2) == pytest.approx(values, abs=0.002)


def test_linear_uint8_tissue_doppler_stream_decodes_directly_to_velocity() -> None:
    stream = stream_from_arrays(
        data_path="data/tissue_doppler",
        data=np.asarray([[[0, 255]]], dtype=np.uint8),
        timestamps_path="timestamps/tissue_doppler",
        timestamps=np.asarray([0.0], dtype=np.float32),
        sample_rate_hz=30.0,
        metadata=StreamMetadata(
            data_path="data/tissue_doppler",
            velocity_limit_mps=0.2,
            storage_encoding="linear_velocity_uint8_mps_v1",
        ),
    )

    assert isinstance(stream, TissueDopplerFloatStream)
    assert stream.data.dtype == np.float32
    assert stream.metadata.value_range == (-0.2, 0.2)
    assert stream.data[0, 0, 0] == pytest.approx(-0.2)
    assert stream.data[0, 0, 1] == pytest.approx(0.2)


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
