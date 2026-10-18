"""Short array names resolve to the arrays a store actually holds."""

import json
from pathlib import Path

import numpy as np
import zarr
from conftest import make_record

from echoxflow import find_recordings, open_recording
from echoxflow.croissant import CroissantCatalog, RecordingRecord
from echoxflow.plotting.renderer import _ecg_trace, _resolve_modality_paths
from echoxflow.schema import display_array_path


def _record(recording_id: str, array_paths: tuple[str, ...]) -> RecordingRecord:
    return make_record(
        recording_id,
        array_paths=array_paths,
        modes=("2d",),
        content_types=("2d_brightness_mode", "2d_color_doppler"),
        frame_counts={},
    )


def test_find_recordings_matches_short_array_names() -> None:
    color = _record(
        "color",
        (
            "data/2d_brightness_mode",
            "data/2d_color_doppler_raw",
            "derived/2d_color_doppler_velocity",
            "derived/2d_color_doppler_power",
        ),
    )
    bmode_only = _record("bmode", ("data/2d_brightness_mode",))
    catalog = CroissantCatalog(path=Path("croissant.json"), recordings=(color, bmode_only))

    found = find_recordings(
        croissant=catalog,
        array_paths=("2d_brightness_mode", "2d_color_doppler_velocity", "2d_color_doppler_power"),
        require_all=True,
    )

    assert [record.recording_id for record in found] == ["color"]
    assert color.has_array_path("2d_color_doppler_velocity")
    assert not bmode_only.has_array_path("2d_color_doppler_velocity")


def test_display_array_path_maps_stored_names_onto_short_names() -> None:
    assert display_array_path("derived/2d_color_doppler_velocity") == "data/2d_color_doppler_velocity"
    assert display_array_path("derived/2d_tissue_doppler_display_velocity") == "data/tissue_doppler"
    assert display_array_path("data/2d_brightness_mode_plane_1") == "data/2d_brightness_mode_1"
    assert display_array_path("data/2d_brightness_mode") == "data/2d_brightness_mode"
    assert display_array_path("data/2d_color_doppler_raw") == "data/2d_color_doppler_raw"


def _write_color_store(path: Path) -> Path:
    group = zarr.open_group(path, mode="w", zarr_format=2)
    group.create_array("data/2d_brightness_mode", data=np.zeros((2, 4, 5), dtype=np.uint8))
    group.create_array("data/2d_color_doppler_raw", data=np.zeros((2, 4, 5, 4), dtype=np.uint8))
    group.create_array("derived/2d_color_doppler_velocity", data=np.zeros((2, 4, 5), dtype=np.float32))
    group.create_array("derived/2d_color_doppler_power", data=np.zeros((2, 4, 5), dtype=np.float32))
    group.create_array("data/lookup_tables/abc", data=np.zeros((256, 3), dtype=np.uint8))
    zarr.consolidate_metadata(path)
    return path


def test_plot_modalities_use_derived_products_under_short_names(tmp_path: Path) -> None:
    store = open_recording(_write_color_store(tmp_path / "rec.zarr"))

    paths = _resolve_modality_paths(store, None)

    assert set(paths) == {
        "data/2d_brightness_mode",
        "data/2d_color_doppler_velocity",
        "data/2d_color_doppler_power",
    }
    assert _resolve_modality_paths(store, ("2d_color_doppler",)) == (
        "data/2d_color_doppler_velocity",
        "data/2d_color_doppler_power",
    )


def test_recording_without_ecg_gets_an_empty_trace(tmp_path: Path) -> None:
    store = open_recording(_write_color_store(tmp_path / "rec.zarr"))

    trace = _ecg_trace(store)

    assert trace.signal.size == 0
    assert trace.timestamps.size == 0


def test_store_without_consolidated_metadata_lists_arrays_from_its_attributes(tmp_path: Path) -> None:
    path = tmp_path / "rec.zarr"
    data = path / "data" / "ecg"
    data.mkdir(parents=True)
    (path / ".zgroup").write_text(json.dumps({"zarr_format": 2}), encoding="utf-8")
    (path / ".zattrs").write_text(
        json.dumps({"arrays": [{"name": "data/ecg"}, {"name": "timestamps/ecg"}, {"role": "missing name"}]}),
        encoding="utf-8",
    )

    assert open_recording(path).array_paths == ("data/ecg", "timestamps/ecg")
