"""Reader contracts for the October 2026 export: source-resolved derived products, native TDI encodings,
per-frame Doppler calibration, scoped lookup tables, stream summaries, clock resets and spectral calibration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from echoxflow import ColorDopplerVelocityStream, TissueDopplerFloatStream, TissueDopplerRawStream, open_recording
from echoxflow.croissant import (
    CroissantArrayRecord,
    RecordingRecord,
    load_croissant,
    stream_summaries_from_attrs,
    stream_summary_mismatches,
)
from echoxflow.loading import lookup_table_path_at
from echoxflow.schema import TDI_FREQ_BW_STORAGE_ENCODING, TDI_PACKED_STORAGE_ENCODING, tdi_raw_to_velocity
from echoxflow.spectral import SpectralCalibration, column_velocity_axes
from echoxflow.streams import ColorFlowRawStream
from echoxflow.timing import interpolate_on_runs, strictly_increasing_runs

zarr = pytest.importorskip("zarr")

FRAMES = 4
FENC = "data/lookup_tables/fenc_65536x4"
FLOW = "data/lookup_tables/flow_256x256x3"
FLOW_2 = "data/lookup_tables/flow2_256x256x3"
COLOR_SCALE = np.asarray([0.6, 0.6, 0.45, 0.45])
TDI_SCALE = np.asarray([0.2, 0.2, 0.12, np.nan])


def _tilt(geometry: Any) -> float:
    return 0.5 * (geometry.angle_start_rad + geometry.angle_end_rad)


def _ref(path: str) -> dict[str, str]:
    return {"zarr_path": path, "format": "zarr_array"}


def _geometry(tilt: float) -> dict[str, Any]:
    return {
        "sample_type": "RxRyPower(8,8,8,8)",
        "depth_start_m": 0.01,
        "depth_end_m": 0.1,
        "tilt_rad": tilt,
        "width_rad": 0.8,
        "grid_size": [3, 5],
    }


def _tables(*entries: tuple[str, str]) -> list[dict[str, Any]]:
    return [{"occurrence_index": i, "role": role, "data": _ref(path)} for i, (role, path) in enumerate(entries)]


def _color_sector(suffix: str, timestamps: str, tilt: float, flow_table: str) -> dict[str, Any]:
    return {
        "semantic_id": f"2d_color_doppler_raw{suffix}",
        "frames": _ref(f"data/2d_color_doppler_raw{suffix}"),
        "timestamps": _ref(timestamps),
        "geometry": _geometry(tilt),
        "storage_encoding": "ge_color_flow_rx_ry_power_aux_u8x4",
        "velocity_scale_per_frame": _ref(f"derived/2d_color_doppler_raw{suffix}_velocity_scale_per_frame"),
        "validity_mask": _ref(f"derived/2d_color_doppler_validity{suffix}"),
        "lookup_tables": _tables(("flow_color_rgb_table", flow_table), ("fenc_table", FENC)),
        "wrapped_radial_velocity": _ref(f"derived/2d_color_doppler_velocity{suffix}"),
        "segment_frame_offsets": _ref(f"coordinates/2d_color_doppler_raw{suffix}_segment_frame_offsets"),
        "stream_controls": {"velocity_limit_mps": 0.6, "packet_size": 12},
    }


def _tdi_sector(name: str, encoding: str, sample_type: str) -> dict[str, Any]:
    return {
        "semantic_id": "tissue_doppler",
        "frames": _ref(f"data/{name}"),
        "timestamps": _ref("timestamps/2d_tissue_doppler"),
        "geometry": {**_geometry(0.0), "sample_type": sample_type},
        "storage_encoding": encoding,
        "velocity_scale_per_frame": _ref(f"derived/{name}_velocity_scale_per_frame"),
        "lookup_tables": _tables(("fenc_table", FENC), ("flow_color_rgb_table", FLOW)),
        "display_velocity": _ref("derived/2d_tissue_doppler_display_velocity"),
    }


def _array_entry(name: str, values: np.ndarray, **extra: Any) -> dict[str, Any]:
    return {"name": name, "dtype": str(values.dtype), "shape": list(values.shape), **extra}


def _write(path: Path, arrays: dict[str, np.ndarray], attrs: dict[str, Any]) -> Path:
    group = zarr.open_group(path, mode="w", zarr_format=2)
    for name, values in arrays.items():
        group.create_array(name, data=values, chunks=values.shape)
    group.attrs.update(json.loads(json.dumps(attrs)))
    zarr.consolidate_metadata(path)
    return path


def _color_channels(rng: np.random.Generator, count: int) -> np.ndarray:
    channels = rng.integers(0, 256, size=(count, 3, 5, 4), dtype=np.uint8)
    channels[0, 0, 0, :2] = 0  # no flow estimate
    channels[1, 0, 1, 2] = 0  # power code 0 but a phase
    return channels


def _expected_color_velocity(raw: np.ndarray, scale: np.ndarray) -> np.ndarray:
    real = raw[..., 0].view(np.int8).astype(np.float32)
    imag = raw[..., 1].view(np.int8).astype(np.float32)
    # The exporter divides the float64 scale by pi before casting to float32.
    velocity = np.arctan2(imag, real).astype(np.float32) * (scale / np.pi).astype(np.float32)[:, None, None]
    velocity[(real == 0) & (imag == 0)] = np.nan
    return velocity


def _fenc_table() -> np.ndarray:
    table = np.zeros((65536, 4), dtype=np.uint8)
    codes = np.arange(65536, dtype=np.uint32)
    table[:, 0] = ((codes * 37 + 11) % 256).astype(np.uint8)
    return table


def _write_recording(tmp_path: Path, *, tdi_encoding: str = TDI_PACKED_STORAGE_ENCODING) -> tuple[Path, dict]:
    rng = np.random.default_rng(7)
    raw = _color_channels(rng, FRAMES)
    raw_2 = _color_channels(rng, FRAMES)
    tdi_raw = rng.integers(0, 256, size=(FRAMES, 3, 5, 2), dtype=np.uint8)
    fenc = _fenc_table()
    sample_type = "RxRyPower(5,5,6)" if tdi_encoding == TDI_PACKED_STORAGE_ENCODING else "FreqBw(8,8)"
    tdi_velocity = tdi_raw_to_velocity(
        tdi_raw, storage_encoding=tdi_encoding, velocity_scale_mps=TDI_SCALE, fenc_table=fenc
    )
    arrays = {
        "data/2d_brightness_mode": rng.integers(0, 256, size=(FRAMES, 6, 8), dtype=np.uint8),
        "timestamps/2d_brightness_mode": np.arange(FRAMES, dtype=np.float64) * 0.1,
        "data/2d_color_doppler_raw": raw,
        "derived/2d_color_doppler_velocity": _expected_color_velocity(raw, COLOR_SCALE),
        "derived/2d_color_doppler_power": raw[..., 2].astype(np.float32) / 255.0,
        "derived/2d_color_doppler_validity": (raw[..., 2] > 0) & ((raw[..., 0] != 0) | (raw[..., 1] != 0)),
        "derived/2d_color_doppler_raw_velocity_scale_per_frame": COLOR_SCALE,
        "coordinates/2d_color_doppler_raw_segment_frame_offsets": np.asarray([0, 2, FRAMES], dtype=np.int64),
        "timestamps/2d_color_doppler": np.arange(FRAMES, dtype=np.float64) * 0.1 + 0.05,
        "data/2d_color_doppler_raw_2": raw_2,
        "derived/2d_color_doppler_velocity_2": _expected_color_velocity(raw_2, np.full(FRAMES, 0.3)),
        "derived/2d_color_doppler_validity_2": np.ones((FRAMES, 3, 5), dtype=bool),
        "derived/2d_color_doppler_raw_2_velocity_scale_per_frame": np.full(FRAMES, 0.3),
        "coordinates/2d_color_doppler_raw_2_segment_frame_offsets": np.asarray([0, FRAMES], dtype=np.int64),
        "timestamps/2d_color_doppler_velocity_2": np.arange(FRAMES, dtype=np.float64) * 0.1 + 0.07,
        "data/2d_tissue_doppler_raw": tdi_raw,
        "derived/2d_tissue_doppler_raw_velocity_scale_per_frame": TDI_SCALE,
        "derived/2d_tissue_doppler_display_velocity": tdi_velocity,
        "timestamps/2d_tissue_doppler": np.arange(FRAMES, dtype=np.float64) * 0.1,
        FENC: fenc,
        FLOW: np.zeros((256, 256, 3), dtype=np.uint8),
        FLOW_2: np.ones((256, 256, 3), dtype=np.uint8),
        "data/ecg": np.sin(np.arange(12, dtype=np.float32)),
        "timestamps/ecg": np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]),
        "data/1d_pulsed_wave_doppler": rng.random((6, 4), dtype=np.float32),
        "timestamps/1d_pulsed_wave_doppler": np.arange(6, dtype=np.float64) * 0.01,
        "coordinates/pw_doppler_baseline_fraction": np.asarray([0.5, 0.5, 0.75, 0.75, np.nan, 0.5]),
        "coordinates/pw_doppler_nyquist_mps": np.asarray([1.0, 1.0, 1.0, 2.0, 1.0, np.nan]),
        "coordinates/pw_doppler_angle_correction_rad": np.zeros(6),
        "coordinates/pw_doppler_segment_sample_offsets": np.asarray([0, 3, 6], dtype=np.int64),
    }
    derivations = {
        "derived/2d_color_doppler_velocity": [
            "data/2d_color_doppler_raw",
            "derived/2d_color_doppler_raw_velocity_scale_per_frame",
        ],
        "derived/2d_color_doppler_power": ["data/2d_color_doppler_raw"],
        "derived/2d_color_doppler_validity": ["data/2d_color_doppler_raw"],
        "derived/2d_color_doppler_velocity_2": [
            "data/2d_color_doppler_raw_2",
            "derived/2d_color_doppler_raw_2_velocity_scale_per_frame",
        ],
        "derived/2d_tissue_doppler_display_velocity": [
            "data/2d_tissue_doppler_raw",
            FENC,
            "derived/2d_tissue_doppler_raw_velocity_scale_per_frame",
        ],
    }
    entries = []
    for name, values in arrays.items():
        extra: dict[str, Any] = {}
        if name.endswith("_velocity_scale_per_frame"):
            extra = {"role": "calibration", "units": "m/s", "dimensions": ["time"], "representation": "derived"}
        elif name.startswith("data/lookup_tables/"):
            extra = {"role": "lookup_table", "dimensions": ["code", "component"]}
        elif name.startswith("timestamps/"):
            extra = {"role": "timestamps", "units": "seconds", "dimensions": ["time"]}
        elif values.ndim == 3 and name.startswith(("data/", "derived/")):
            extra = {"role": "frames", "units": "m/s", "dimensions": ["time", "radial", "beam"]}
        if name in derivations:
            extra["derivation"] = {"name": "test", "version": 1, "inputs": derivations[name]}
        entries.append(_array_entry(name, values, **extra))
    spectral = {
        key: _ref(f"coordinates/pw_doppler_{key}")
        for key in ("baseline_fraction", "nyquist_mps", "angle_correction_rad")
    }
    attrs = {
        "schema_version": "EchoXFlow-2",
        "recording_id": "recording_test",
        "arrays": entries,
        "recording_manifest": {
            "manifest_type": "2d",
            "metadata": {"heart_rate_bpm": 71.0},
            "sectors": [
                {
                    "semantic_id": "2d_brightness_mode",
                    "frames": _ref("data/2d_brightness_mode"),
                    "timestamps": _ref("timestamps/2d_brightness_mode"),
                    "geometry": {**_geometry(0.0), "sample_type": "Tissue(8)"},
                },
                _color_sector("", "timestamps/2d_color_doppler", 0.1, FLOW),
                {
                    "semantic_id": "2d_color_doppler_power",
                    "frames": _ref("derived/2d_color_doppler_power"),
                    "timestamps": _ref("timestamps/2d_color_doppler"),
                    "geometry": _geometry(0.1),
                    "lookup_tables": _tables(("flow_color_rgb_table", FLOW)),
                },
                _color_sector("_2", "timestamps/2d_color_doppler_velocity_2", -0.2, FLOW_2),
                _tdi_sector("2d_tissue_doppler_raw", tdi_encoding, sample_type),
            ],
            "tracks": [
                {
                    "semantic_id": "pw_doppler",
                    "data": _ref("data/1d_pulsed_wave_doppler"),
                    "timestamps": _ref("timestamps/1d_pulsed_wave_doppler"),
                    "segment_sample_offsets": _ref("coordinates/pw_doppler_segment_sample_offsets"),
                    "baseline_frac": 0.5,
                    "spectral_calibration": spectral,
                }
            ],
        },
        "stream_summaries": [
            {
                "array_path": "data/2d_color_doppler_raw",
                "timestamp_path": "timestamps/2d_color_doppler",
                "sample_count": FRAMES,
                "clock_status": "aligned",
                "reset_count": 0,
                "velocity_scale_status": "varying",
                "velocity_scale_min_mps": 0.45,
                "velocity_scale_max_mps": 0.6,
                "velocity_scale_change_count": 1,
                "calibration_status": "varying",
            },
            {
                "array_path": "data/ecg",
                "timestamp_path": "timestamps/ecg",
                "sample_count": 12,
                "clock_status": "reset",
                "reset_count": 1,
                "velocity_scale_status": "missing",
            },
        ],
    }
    return _write(tmp_path / "recording_test.zarr", arrays, attrs), arrays


# --- R1: derived colour velocity inherits its source sector --------------------------------------------------------


def test_derived_color_velocity_inherits_its_raw_sector(tmp_path: Path) -> None:
    path, arrays = _write_recording(tmp_path)
    store = open_recording(path)

    stream = store.load_stream("data/2d_color_doppler_velocity")

    assert isinstance(stream, ColorDopplerVelocityStream)
    assert stream.data_path == "derived/2d_color_doppler_velocity"
    assert stream.metadata.source_data_path == "data/2d_color_doppler_raw"
    assert stream.timestamps_path == "timestamps/2d_color_doppler"
    assert stream.metadata.geometry is not None and _tilt(stream.metadata.geometry) == pytest.approx(0.1)
    assert stream.metadata.validity_mask_path == "derived/2d_color_doppler_validity"
    np.testing.assert_allclose(stream.metadata.velocity_scale_mps_per_frame, COLOR_SCALE)
    # The scale changes, so no single limit is claimed (the stream_controls snapshot says 0.6).
    assert stream.metadata.velocity_limit_mps is None
    assert stream.metadata.velocity_scale_status == "varying"
    assert stream.metadata.acquisition_controls["packet_size"] == 12
    np.testing.assert_array_equal(np.isnan(stream.data), np.isnan(arrays["derived/2d_color_doppler_velocity"]))


def test_secondary_color_stream_uses_its_own_sector_clock_and_table(tmp_path: Path) -> None:
    store = open_recording(_write_recording(tmp_path)[0])

    velocity = store.load_stream("derived/2d_color_doppler_velocity_2")
    raw = store.load_stream("data/2d_color_doppler_raw_2")

    assert isinstance(velocity, ColorDopplerVelocityStream)
    assert isinstance(raw, ColorFlowRawStream)
    assert velocity.timestamps_path == "timestamps/2d_color_doppler_velocity_2"
    assert velocity.metadata.geometry is not None and _tilt(velocity.metadata.geometry) == pytest.approx(-0.2)
    assert velocity.metadata.colormap_path == FLOW_2
    assert velocity.metadata.velocity_limit_mps == pytest.approx(0.3)
    np.testing.assert_allclose(raw.to_velocity().data, velocity.data, equal_nan=True)


def test_derived_product_claimed_by_two_sectors_is_rejected(tmp_path: Path) -> None:
    path, _arrays = _write_recording(tmp_path)
    group = zarr.open_group(path, mode="r+")
    attrs = dict(group.attrs)
    attrs["recording_manifest"]["sectors"][3]["wrapped_radial_velocity"] = _ref("derived/2d_color_doppler_velocity")
    group.attrs.put(attrs)

    with pytest.raises(ValueError, match="several manifest sectors"):
        open_recording(path).load_stream("data/2d_color_doppler_velocity")


# --- R2/R3: native TDI encodings and per-frame calibration ---------------------------------------------------------


@pytest.mark.parametrize("encoding", [TDI_PACKED_STORAGE_ENCODING, TDI_FREQ_BW_STORAGE_ENCODING])
def test_native_tdi_decodes_to_the_exported_display_velocity(tmp_path: Path, encoding: str) -> None:
    path, arrays = _write_recording(tmp_path, tdi_encoding=encoding)
    store = open_recording(path)

    raw = store.load_stream("data/2d_tissue_doppler_raw")
    decoded = store.load_modality("data/2d_tissue_doppler_raw").data
    alias = store.load_modality("data/tissue_doppler").stream

    assert isinstance(raw, TissueDopplerRawStream)
    assert raw.data.shape == (FRAMES, 3, 5, 2)
    assert raw.metadata.storage_encoding == encoding
    assert isinstance(alias, TissueDopplerFloatStream)
    np.testing.assert_array_equal(decoded, arrays["derived/2d_tissue_doppler_display_velocity"])
    np.testing.assert_array_equal(alias.data, arrays["derived/2d_tissue_doppler_display_velocity"])
    # The uncalibrated last frame stays unknown rather than borrowing another frame's scale.
    assert np.all(np.isnan(decoded[3]))


def test_packed_tdi_keeps_the_declared_sign_and_scale() -> None:
    raw = np.zeros((2, 1, 2, 2), dtype=np.uint8)
    raw[:, 0, 1, 0] = 1
    fenc = np.zeros((65536, 4), dtype=np.uint8)
    fenc[0, 0] = np.uint8(64)  # +64 -> half the positive range
    fenc[1, 0] = np.int8(-128).view(np.uint8)  # most negative frequency

    velocity = tdi_raw_to_velocity(
        raw, storage_encoding=TDI_PACKED_STORAGE_ENCODING, velocity_scale_mps=np.asarray([0.2, 0.1]), fenc_table=fenc
    )

    np.testing.assert_allclose(velocity[:, 0, 0], [0.1, 0.05])
    np.testing.assert_allclose(velocity[:, 0, 1], [-0.2, -0.1])


def test_freq_bw_tdi_reads_frequency_directly_without_a_fenc_table() -> None:
    raw = np.zeros((1, 1, 1, 2), dtype=np.uint8)
    raw[..., 0] = np.int8(-32).view(np.uint8)
    raw[..., 1] = 200  # bandwidth must not leak into the velocity

    velocity = tdi_raw_to_velocity(raw, storage_encoding=TDI_FREQ_BW_STORAGE_ENCODING, velocity_scale_mps=0.4)

    assert velocity[0, 0, 0] == pytest.approx(-0.1)


def test_frame_selections_carry_their_own_calibration(tmp_path: Path) -> None:
    path, arrays = _write_recording(tmp_path)
    store = open_recording(path)

    first = store.load_stream_slice("data/2d_color_doppler_velocity", 0, 2)
    later = store.load_stream_slice("data/2d_color_doppler_raw", 2, 4)
    header = store.load_stream_header("data/2d_color_doppler_velocity")

    assert first.metadata.velocity_limit_mps == pytest.approx(0.6)
    np.testing.assert_allclose(later.metadata.velocity_scale_mps_per_frame, [0.45, 0.45])
    assert isinstance(later, ColorFlowRawStream)
    np.testing.assert_allclose(
        later.to_velocity().data, arrays["derived/2d_color_doppler_velocity"][2:4], equal_nan=True
    )
    assert later.to_velocity().metadata.value_range == pytest.approx((-0.45, 0.45))
    np.testing.assert_allclose(header.velocity_scale_mps_per_frame, COLOR_SCALE)
    np.testing.assert_allclose(header.stream.metadata.velocity_scale_mps_per_frame, COLOR_SCALE[:1])
    np.testing.assert_array_equal(
        store.load_validity_mask("data/2d_color_doppler_velocity", 1, 3),
        arrays["derived/2d_color_doppler_validity"][1:3],
    )


def test_invalid_per_frame_scales_are_rejected(tmp_path: Path) -> None:
    path, _arrays = _write_recording(tmp_path)
    group = zarr.open_group(path, mode="r+")
    group["derived/2d_color_doppler_raw_velocity_scale_per_frame"][:] = np.asarray([0.6, -0.6, 0.6, 0.6])

    with pytest.raises(ValueError, match="positive where calibrated"):
        open_recording(path).load_stream("data/2d_color_doppler_velocity")


# --- R7: scoped lookup tables --------------------------------------------------------------------------------------


def test_lookup_tables_come_from_the_streams_own_sector(tmp_path: Path) -> None:
    path, _arrays = _write_recording(tmp_path)
    store = open_recording(path)

    # Two flow tables share the store; shape alone cannot pick one, the sector reference can.
    assert lookup_table_path_at(path, "flow_color_rgb_table") is None
    assert store.load_stream("data/2d_color_doppler_raw").metadata.colormap_path == FLOW
    assert store.load_stream("data/2d_color_doppler_raw_2").metadata.colormap_path == FLOW_2
    assert store.load_stream("data/2d_color_doppler_raw").metadata.fenc_table_path is None
    assert store.load_stream("data/tissue_doppler").metadata.fenc_table_path == FENC


# --- R8: interpretation metadata --------------------------------------------------------------------------------------


def test_store_inventory_and_array_declarations_survive_a_reduced_catalog(tmp_path: Path) -> None:
    path, _arrays = _write_recording(tmp_path)
    record = RecordingRecord(
        exam_id="exam",
        recording_id="recording_test",
        zarr_path=path.name,
        modes=("2d",),
        content_types=("2d_color_doppler",),
        frame_counts_by_content_type={},
        median_delta_time_by_content_type={},
        array_paths=("data/2d_color_doppler_raw",),
    )
    store = open_recording(record, root=tmp_path)

    assert FENC in store.array_paths
    assert "derived/2d_color_doppler_raw_velocity_scale_per_frame" in store.array_paths
    scale = store.array_info("derived/2d_color_doppler_raw_velocity_scale_per_frame")
    velocity = store.array_info("data/2d_color_doppler_velocity")
    assert scale is not None and scale.units == "m/s" and not scale.is_frame_array
    assert velocity is not None and velocity.is_frame_array
    assert velocity.derivation_inputs[0] == "data/2d_color_doppler_raw"
    assert store.heart_rate_bpm() == pytest.approx(71.0)
    np.testing.assert_array_equal(store.segment_offsets("data/2d_color_doppler_velocity"), [0, 2, FRAMES])


def test_catalog_arrays_distinguish_frames_from_calibration() -> None:
    def row(path: str, role: str, shape: tuple[int, ...]) -> CroissantArrayRecord:
        return CroissantArrayRecord(recording_id="r", array_path=path, content_types=(), role=role, shape=shape)

    assert row("derived/2d_color_doppler_velocity", "frames", (4, 3, 5)).is_frame_array
    assert not row("derived/2d_color_doppler_raw_velocity_scale_per_frame", "calibration", (4,)).is_frame_array
    assert not row(FENC, "lookup_table", (65536, 4)).is_frame_array


# --- R4: stream summaries ---------------------------------------------------------------------------------------------


def test_stream_summaries_parse_from_store_and_catalog(tmp_path: Path) -> None:
    path, _arrays = _write_recording(tmp_path)
    store = open_recording(path)
    catalog_path = tmp_path / "croissant.json"
    catalog_path.write_text(
        json.dumps(
            {
                "recordSet": [
                    {"@id": "recordings", "data": []},
                    {
                        "@id": "streams",
                        "data": [
                            {
                                "streams/array_id": "recording_test/data/2d_color_doppler_raw",
                                "streams/recording_id": "recording_test",
                                "streams/timestamp_array_id": "recording_test/timestamps/2d_color_doppler",
                                "streams/sample_count": FRAMES,
                                "streams/clock_status": "aligned",
                                "streams/reset_count": 0,
                                "streams/velocity_scale_status": "constant",
                                "streams/velocity_scale_min_mps": 0.6,
                                "streams/velocity_scale_max_mps": 0.6,
                            }
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    catalog_summary = load_croissant(catalog_path).stream_summary("recording_test", "data/2d_color_doppler_raw")
    store_summary = store.stream_summary("data/2d_color_doppler_velocity")
    ecg = store.stream_summary("data/ecg")

    assert catalog_summary is not None and store_summary is not None and ecg is not None
    assert store_summary.array_path == "data/2d_color_doppler_raw"
    assert store_summary.velocity_scale_may_intersect((0.60, 0.61)) is True
    assert store_summary.velocity_scale_may_intersect((0.1, 0.2)) is False
    assert stream_summary_mismatches(catalog_summary, store_summary) == ("velocity_scale_status",)
    assert ecg.has_clock_reset and "clock_reset" in ecg.issues()
    assert stream_summaries_from_attrs({}) == ()


def test_catalog_without_stream_summaries_still_loads(tmp_path: Path) -> None:
    catalog_path = tmp_path / "croissant.json"
    catalog_path.write_text(json.dumps({"recordSet": [{"@id": "recordings", "data": []}]}), encoding="utf-8")

    catalog = load_croissant(catalog_path)

    assert catalog.stream_summaries == ()
    assert catalog.stream_summary("recording_test", "data/ecg") is None


# --- R5: clock resets -------------------------------------------------------------------------------------------------


def test_reset_clock_is_split_into_runs_without_reordering() -> None:
    times = np.asarray([0.0, 0.1, 0.2, 0.15, 0.25, 0.25, 0.3])

    runs = strictly_increasing_runs(times)
    with_segments = strictly_increasing_runs(
        np.arange(6.0), segment_offsets=np.asarray([0, 3, 6]), require_continuity_across_segments=True
    )

    assert [(run.start, run.stop) for run in runs] == [(0, 3), (3, 5), (5, 7)]
    assert [(run.start, run.stop) for run in with_segments] == [(0, 3), (3, 6)]


def test_interpolation_masks_the_ambiguous_overlap_of_a_reset() -> None:
    times = np.asarray([0.0, 0.1, 0.2, 0.15, 0.25, 0.35])
    values = np.asarray([0.0, 1.0, 2.0, 10.0, 20.0, 30.0])

    result, valid = interpolate_on_runs(times, values, np.asarray([0.05, 0.17, 0.3, 0.5]))

    assert valid.tolist() == [True, False, True, False]
    np.testing.assert_allclose(result, [0.5, 0.0, 25.0, 0.0])


# --- R6: spectral calibration -----------------------------------------------------------------------------------------


def test_spectral_columns_use_their_own_calibration(tmp_path: Path) -> None:
    store = open_recording(_write_recording(tmp_path)[0])

    full = store.spectral_metadata("data/1d_pulsed_wave_doppler")
    constant = store.spectral_metadata("data/1d_pulsed_wave_doppler", 0, 2)

    assert full.calibration is not None and full.calibration.status == "partial"
    assert full.row_velocity_mps is None  # no single axis fits every column
    axes = full.column_velocity_mps
    assert axes is not None and axes.shape == (6, 4)
    np.testing.assert_allclose(axes[0], [0.75, 0.25, -0.25, -0.75])
    np.testing.assert_allclose(axes[3], [2.5, 1.5, 0.5, -0.5])  # baseline 0.75, Nyquist 2
    assert np.all(np.isnan(axes[4])) and np.all(np.isnan(axes[5]))
    assert constant.row_velocity_mps is not None
    np.testing.assert_allclose(constant.row_velocity_mps, [0.75, 0.25, -0.25, -0.75])
    assert constant.nyquist_limit_mps == pytest.approx(1.0)
    np.testing.assert_array_equal(store.segment_offsets("data/1d_pulsed_wave_doppler"), [0, 3, 6])


def test_column_axes_span_twice_the_nyquist_limit() -> None:
    calibration = SpectralCalibration(baseline_fraction=np.asarray([0.25]), nyquist_mps=np.asarray([1.0]))

    axis = column_velocity_axes(calibration, row_count=2)[0]

    np.testing.assert_allclose(axis, [0.0, -1.0])
