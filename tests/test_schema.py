import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from echoxflow.loading import _with_spectral_colormap, lookup_table_path_at, lookup_table_paths_at, open_recording
from echoxflow.schema import (
    annotation_names,
    canonical_stream_name,
    color_flow_raw_to_power,
    color_flow_raw_to_velocity,
    color_flow_velocity_codes,
    current_array_candidates,
    flow_color_first_axis_is_brightness,
    flow_color_velocity_ramp,
    is_lookup_table_path,
    lookup_table_role,
    normalize_array_path,
    render_flow_color_table,
    resolve_array_path,
    semantic_name,
    tdi_raw_channels_to_native_codes,
)
from echoxflow.streams import StreamMetadata


def test_normalize_array_path_defaults_bare_names_into_data() -> None:
    assert normalize_array_path("2d_brightness_mode") == "data/2d_brightness_mode"
    assert normalize_array_path("/derived/2d_color_doppler_power/") == "derived/2d_color_doppler_power"
    assert normalize_array_path("annotations/measurement_000_points") == "annotations/measurement_000_points"


def test_semantic_name_strips_any_group_prefix() -> None:
    assert semantic_name("data/2d_color_doppler_velocity") == "2d_color_doppler_velocity"
    assert semantic_name("derived/2d_color_doppler_velocity") == "2d_color_doppler_velocity"
    assert semantic_name("coordinates/mmode_segment_sample_offsets") == "mmode_segment_sample_offsets"


def test_canonical_stream_name_maps_display_products_onto_stable_names() -> None:
    assert canonical_stream_name("derived/2d_tissue_doppler_display_velocity") == "tissue_doppler"
    # The raw estimator array keeps its own name so the two never collapse into one stream.
    assert canonical_stream_name("data/2d_tissue_doppler_raw") == "2d_tissue_doppler_raw"
    assert canonical_stream_name("data/2d_brightness_mode") == "2d_brightness_mode"


def test_current_array_candidates_offers_stored_replacements() -> None:
    assert current_array_candidates("data/2d_color_doppler_velocity") == (
        "data/2d_color_doppler_velocity",
        "derived/2d_color_doppler_velocity",
    )
    assert current_array_candidates("data/tissue_doppler") == (
        "data/tissue_doppler",
        "derived/2d_tissue_doppler_display_velocity",
        "data/2d_tissue_doppler_raw",
    )
    assert current_array_candidates("data/2d_brightness_mode") == ("data/2d_brightness_mode",)


def test_resolve_array_path_prefers_the_requested_path() -> None:
    available = {"data/2d_color_doppler_velocity", "derived/2d_color_doppler_velocity"}
    assert resolve_array_path("data/2d_color_doppler_velocity", available) == "data/2d_color_doppler_velocity"
    assert resolve_array_path("data/2d_color_doppler_velocity", {"derived/2d_color_doppler_velocity"}) == (
        "derived/2d_color_doppler_velocity"
    )
    assert resolve_array_path("data/2d_color_doppler_velocity", set()) is None


def test_lookup_table_role_classifies_the_reshaped_tables() -> None:
    assert lookup_table_role((256, 256, 3)) == "flow_color_rgb_table"
    assert lookup_table_role((65536, 4)) == "fenc_table"
    assert lookup_table_role((256, 3)) == "color_rgb_table"
    assert lookup_table_role((4096,)) == "unknown_lookup_table"


def test_is_lookup_table_path_matches_content_addressed_tables() -> None:
    assert is_lookup_table_path("data/lookup_tables/abc123_256x3")
    assert not is_lookup_table_path("data/2d_brightness_mode")


def test_color_flow_raw_decodes_to_wrapped_velocity_and_power() -> None:
    raw = np.zeros((1, 1, 3, 4), dtype=np.uint8)
    # rx and ry are signed; a positive real axis is zero velocity, the negative axis wraps.
    raw[0, 0, 0] = (100, 0, 255, 0)
    raw[0, 0, 1] = (0, 0, 0, 0)
    raw[0, 0, 2] = (np.uint8(256 - 100), 0, 128, 0)

    velocity = color_flow_raw_to_velocity(raw, velocity_limit_mps=0.6)
    power = color_flow_raw_to_power(raw)

    assert velocity[0, 0, 0] == pytest.approx(0.0)
    assert np.isnan(velocity[0, 0, 1])
    assert velocity[0, 0, 2] == pytest.approx(0.6, abs=1e-6)
    assert power[0, 0, 0] == pytest.approx(1.0)
    assert power[0, 0, 1] == pytest.approx(0.0)


def test_color_flow_raw_rejects_a_wrong_channel_count() -> None:
    with pytest.raises(ValueError, match="time, radial, beam, 4"):
        color_flow_raw_to_velocity(np.zeros((1, 1, 1, 2), dtype=np.uint8), velocity_limit_mps=0.6)


def test_color_flow_velocity_codes_follow_the_tables_phase_axis() -> None:
    # Zero flow and no-estimate both sit at code 0, the table's black entry. The Nyquist
    # extremes meet at the 127/128 wrap, which is where a scanner shows aliasing.
    velocities = np.asarray([0.0, 0.6, -0.6, 0.3, -0.3, np.nan], dtype=np.float32)

    codes = color_flow_velocity_codes(velocities, velocity_limit_mps=0.6)

    assert codes.tolist() == [0, 127, 128, 64, 192, 0]


def test_color_flow_velocity_codes_clamp_beyond_the_nyquist_limit() -> None:
    codes = color_flow_velocity_codes(np.asarray([1.2, -1.2], dtype=np.float32), velocity_limit_mps=0.6)

    assert codes.tolist() == [127, 128]


def test_color_flow_velocity_codes_require_a_positive_limit() -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        color_flow_velocity_codes(np.zeros(2, dtype=np.float32), velocity_limit_mps=0.0)


def test_render_flow_color_table_indexes_secondary_then_velocity() -> None:
    table = np.zeros((256, 256, 3), dtype=np.uint8)
    table[10, 20] = (1, 2, 3)

    rgb = render_flow_color_table(
        velocity_codes=np.asarray([[20]], dtype=np.uint8),
        secondary_codes=np.asarray([[10]], dtype=np.uint8),
        flow_color_table=table,
    )

    assert rgb.shape == (1, 1, 3)
    assert rgb[0, 0].tolist() == [1, 2, 3]


def _brightness_style_table() -> np.ndarray:
    """One hue ramp scaled by the first index, as a `flow_variance: 0` stream stores it."""
    hues = np.zeros((256, 3), dtype=np.float32)
    hues[:, 0] = np.linspace(255.0, 0.0, 256)
    hues[:, 2] = np.linspace(0.0, 255.0, 256)
    scale = (np.arange(256, dtype=np.float32) / 255.0)[:, None, None]
    return np.rint(scale * hues[None, :, :]).astype(np.uint8)


def _variance_style_table() -> np.ndarray:
    """A hue ramp that blends toward green with the first index, as turbulence maps do."""
    table = _brightness_style_table()
    table[:] = _brightness_style_table()[255][None, :, :]
    green = (np.arange(256, dtype=np.float32) / 255.0)[:, None]
    table[..., 1] = np.rint(green * 255.0).astype(np.uint8)
    return table


def test_flow_color_first_axis_is_brightness_separates_the_two_table_styles() -> None:
    assert flow_color_first_axis_is_brightness(_brightness_style_table())
    assert not flow_color_first_axis_is_brightness(_variance_style_table())


def test_flow_color_velocity_ramp_picks_the_row_the_first_axis_does_not_alter() -> None:
    brightness = _brightness_style_table()
    variance = _variance_style_table()

    # Full power for a brightness table, zero turbulence for a variance table.
    assert flow_color_velocity_ramp(brightness).tolist() == brightness[255].tolist()
    assert flow_color_velocity_ramp(variance).tolist() == variance[0].tolist()


def test_flow_color_velocity_ramp_returns_a_full_length_ramp() -> None:
    assert flow_color_velocity_ramp(_brightness_style_table()).shape == (256, 3)


def _phase_ordered_table() -> np.ndarray:
    """A colour-flow table: black zero flow at code 0, yellow at +limit, cyan at -limit."""
    row = np.zeros((256, 3), dtype=np.uint8)
    row[127] = (254, 253, 18)
    row[128] = (17, 249, 249)
    row[63] = (200, 90, 0)
    row[192] = (5, 95, 199)
    return np.repeat(row[None, :, :], 256, axis=0)


def test_phase_ordered_ramp_puts_the_aliasing_wrap_at_the_velocity_extremes() -> None:
    ramp = flow_color_velocity_ramp(_phase_ordered_table(), phase_ordered=True)

    # Renderers index linearly over [-limit, +limit]: cyan at the away extreme, yellow at
    # the toward extreme, black through zero flow in the middle.
    assert ramp[0].tolist() == [17, 249, 249]
    assert ramp[255].tolist() == [254, 253, 18]
    assert ramp[127].tolist() == [0, 0, 0]
    assert ramp[128].tolist() == [0, 0, 0]
    assert ramp[64].tolist() == [5, 95, 199]
    assert ramp[191].tolist() == [200, 90, 0]


def test_linear_ramp_is_left_alone_when_not_phase_ordered() -> None:
    table = _phase_ordered_table()

    assert flow_color_velocity_ramp(table, phase_ordered=False).tolist() == table[0].tolist()


def test_flow_color_velocity_ramp_rejects_a_wrong_shape() -> None:
    with pytest.raises(ValueError, match=r"\(256, 256, 3\)"):
        flow_color_velocity_ramp(np.zeros((256, 3), dtype=np.uint8))


def test_tdi_raw_channels_pack_little_endian_into_native_codes() -> None:
    channels = np.zeros((1, 1, 2, 2), dtype=np.uint8)
    channels[0, 0, 0] = (0x34, 0x12)
    channels[0, 0, 1] = (0xFF, 0x00)

    codes = tdi_raw_channels_to_native_codes(channels)

    assert codes.dtype == np.uint16
    assert codes[0, 0].tolist() == [0x1234, 0x00FF]


def _write_array(path: Path, values: np.ndarray) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / ".zarray").write_text(
        json.dumps(
            {
                "zarr_format": 2,
                "shape": list(values.shape),
                "chunks": list(values.shape),
                "dtype": values.dtype.str,
                "compressor": None,
                "fill_value": 0,
                "order": "C",
                "filters": None,
            }
        ),
        encoding="utf-8",
    )
    (path / ".".join("0" for _ in values.shape)).write_bytes(values.tobytes())


def _write_store(root: Path) -> Path:
    store = root / "recording_test.zarr"
    store.mkdir(parents=True, exist_ok=True)
    (store / ".zgroup").write_text(json.dumps({"zarr_format": 2}), encoding="utf-8")
    (store / ".zattrs").write_text(
        json.dumps(
            {
                "schema_version": "EchoXFlow-2",
                "recording_manifest": {
                    "manifest_type": "2d",
                    "sectors": [
                        {
                            "semantic_id": "2d_color_doppler_velocity",
                            "frames": {"zarr_path": "data/2d_color_doppler_raw"},
                            "timestamps": {"zarr_path": "timestamps/2d_color_doppler"},
                            "velocity_limit_mps": 0.6,
                            "storage_encoding": "ge_color_flow_rx_ry_power_aux_u8x4",
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    _write_array(store / "derived/2d_color_doppler_velocity", np.zeros((2, 3, 4), dtype=np.float32))
    _write_array(store / "timestamps/2d_color_doppler", np.asarray([0.0, 0.1], dtype=np.float64))
    _write_array(store / "data/lookup_tables/flow_256x256x3", np.zeros((256, 256, 3), dtype=np.uint8))
    _write_array(store / "data/lookup_tables/color_256x3", np.zeros((256, 3), dtype=np.uint8))
    return store


def test_lookup_tables_are_discovered_and_classified_by_shape(tmp_path: Path) -> None:
    store = _write_store(tmp_path)

    assert lookup_table_paths_at(store) == (
        "data/lookup_tables/color_256x3",
        "data/lookup_tables/flow_256x256x3",
    )
    assert lookup_table_path_at(store, "flow_color_rgb_table") == "data/lookup_tables/flow_256x256x3"
    assert lookup_table_path_at(store, "color_rgb_table") == "data/lookup_tables/color_256x3"
    assert lookup_table_path_at(store, "fenc_table") is None


@pytest.mark.parametrize("mode", ["pulsed", "continuous"])
def test_spectral_table_excludes_sector_tables_and_refuses_ambiguity(tmp_path: Path, mode: str) -> None:
    grey = "data/lookup_tables/grey"
    gold = "data/lookup_tables/gold"
    _write_array(tmp_path / grey, np.zeros((256, 3), dtype=np.uint8))
    _write_array(tmp_path / gold, np.ones((256, 3), dtype=np.uint8))
    attrs = {
        "recording_manifest": {
            "sectors": [
                {"semantic_id": "bmode", "lookup_tables": [{"role": "color_rgb_table", "data": {"zarr_path": grey}}]}
            ]
        }
    }
    metadata = StreamMetadata(data_path=f"data/1d_{mode}_wave_doppler")
    assert _with_spectral_colormap(metadata, store_path=tmp_path, group_attrs=attrs).colormap_path == gold
    _write_array(tmp_path / "data/lookup_tables/other", np.ones((256, 3), dtype=np.uint8))
    assert _with_spectral_colormap(metadata, store_path=tmp_path, group_attrs=attrs).colormap_path is None
    assert (
        _with_spectral_colormap(
            replace(metadata, colormap_path=gold), store_path=tmp_path, group_attrs=attrs
        ).colormap_path
        == gold
    )


def test_explicit_spectral_table_wins_over_unclaimed_tables(tmp_path: Path) -> None:
    metadata = StreamMetadata(data_path="data/1d_pulsed_wave_doppler")
    track = {
        "data": {"zarr_path": metadata.data_path},
        "lookup_tables": [{"role": "color_rgb_table", "data": {"zarr_path": "data/lookup_tables/explicit"}}],
    }
    attrs = {"recording_manifest": {"tracks": [track]}}
    _write_array(tmp_path / "data/lookup_tables/explicit", np.zeros((256, 3), dtype=np.uint8))
    result = _with_spectral_colormap(metadata, store_path=tmp_path, group_attrs=attrs)
    assert result.colormap_path == "data/lookup_tables/explicit"
    track["lookup_tables"] = []
    assert _with_spectral_colormap(metadata, store_path=tmp_path, group_attrs=attrs).colormap_path is None


def test_store_resolves_a_short_colour_doppler_path_and_attaches_its_colour_table(tmp_path: Path) -> None:
    store_path = _write_store(tmp_path)
    store = open_recording(store_path)

    assert store.resolve_array_path("data/2d_color_doppler_velocity") == "derived/2d_color_doppler_velocity"
    assert store.has_array("data/2d_color_doppler_velocity")
    assert store.array_shape("data/2d_color_doppler_velocity") == (2, 3, 4)

    stream = store.load_stream("data/2d_color_doppler_velocity")

    assert stream.kind == "color_doppler_velocity"
    assert stream.metadata.colormap_path == "data/lookup_tables/flow_256x256x3"
    # Colour flow does not decode through a FencTable, so none may be attached.
    assert stream.metadata.fenc_table_path is None


def test_annotation_names_cover_the_annotation_layout() -> None:
    paths = (
        "data/2d_brightness_mode",
        "annotations/4ch_contour",
        "annotations/3d_left_ventricle_mesh/point_values",
        "annotations/3d_right_ventricle_mesh_1/point_values",
        "annotations/measurement_document_2/measurement_3_points",
        "annotations/1d_pulsed_wave_doppler_annotation_1",
        "annotations/measurement_document_2/1d_continuous_wave_doppler_annotation_4",
        "annotations/1d_motion_mode_annotation_1",
        "annotations/bmode_ecg_qrs",
        "annotations/la_trace/plane_0/polygon_pixels",
        "data/ecg",
    )
    assert annotation_names(paths, ("2d_left_ventricular_strain",)) == (
        "cw_doppler_annotation",
        "ecg_qrs",
        "la_trace",
        "lv_mesh",
        "lv_strain_contour",
        "measurement_points",
        "mmode_annotation",
        "pw_doppler_annotation",
        "rv_mesh",
        "strain_contour",
    )
    # A contour without a strain content type is not a strain contour; coordinates are not annotations.
    assert (
        annotation_names(("annotations/x_contour", "coordinates/3d_left_ventricle_mesh"), ("2d_brightness_mode",)) == ()
    )


def test_spectral_palette_selection_provenance_and_invalid_explicit_ref(tmp_path: Path) -> None:
    metadata = StreamMetadata(data_path="data/1d_pulsed_wave_doppler")
    rgb = "data/lookup_tables/rgb"
    _write_array(tmp_path / rgb, np.zeros((256, 3), dtype=np.uint8))
    unique = _with_spectral_colormap(metadata, store_path=tmp_path, group_attrs={})
    assert unique.colormap_selection == "unique_unclaimed"
    assert unique.colormap_candidates == (rgb,)
    for association in [[], [{"role": "color_rgb_table", "data": {"zarr_path": "data/missing"}}]]:
        attrs = {
            "recording_manifest": {
                "merged_documents": [
                    {"tracks": [{"data": {"zarr_path": metadata.data_path}, "lookup_tables": association}]}
                ]
            }
        }
        result = _with_spectral_colormap(metadata, store_path=tmp_path, group_attrs=attrs)
        assert result.colormap_path is None
        assert result.colormap_selection == "explicit_unresolved"
    attrs = {
        "recording_manifest": {
            "merged_documents": [
                {
                    "tracks": [
                        {
                            "data": {"zarr_path": metadata.data_path},
                            "lookup_tables": [{"role": "color_rgb_table", "data": {"zarr_path": rgb}}],
                        }
                    ]
                }
            ]
        }
    }
    explicit = _with_spectral_colormap(metadata, store_path=tmp_path, group_attrs=attrs)
    assert explicit.colormap_selection == "explicit"
    assert explicit.colormap_path == rgb


@pytest.mark.parametrize("group", ["data", "timestamps"])
@pytest.mark.parametrize("index", [0, 1, 2, 12])
def test_plane_aliases_resolve_both_ways(group: str, index: int) -> None:
    stored = f"{group}/2d_brightness_mode_plane_{index}"
    display = f"{group}/2d_brightness_mode_{index}"
    assert resolve_array_path(display, {stored}) == stored
    assert resolve_array_path(stored, {display}) == display
    assert resolve_array_path(stored, {stored, display}) == stored
    assert resolve_array_path(display, {stored, display}) == display
