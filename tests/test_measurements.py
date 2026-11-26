from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import make_record

from echoxflow import CroissantCatalog, Measurement, open_recording, plot_measurements
from echoxflow.plotting.measurements import _columns_at_times

zarr = pytest.importorskip("zarr")

SOURCE_POINTS = np.asarray([[0.01, 0.06], [-0.005, 0.08]], dtype=np.float32)
IMAGE_POINTS = SOURCE_POINTS * np.asarray([-1.0, 1.0], dtype=np.float32)
FRAME_TIMES = (1.0, 1.02, 1.04)


def _ref(path: str) -> dict[str, str]:
    return {"zarr_path": path, "format": "zarr_array"}


def _sector(name: str = "2d_brightness_mode") -> dict[str, Any]:
    return {
        "semantic_id": "bmode",
        "frames": _ref(f"data/{name}"),
        "timestamps": _ref(f"timestamps/{name}"),
        "geometry": {"depth_start_m": 0.0, "depth_end_m": 0.12, "tilt_rad": 0.0, "width_rad": 1.2},
    }


def _measurement(index: int, coordinate_type: str = "spatial", *, prefix: str = "", **extra: Any) -> dict[str, Any]:
    links = {
        "coordinate_type": coordinate_type,
        "geometry_kind": "caliper",
        "measure_id": "Cardiac/2D/Dimension/LVOT Diam",
        "measurement_kind": "dimension_lvot_diam",
        **extra.pop("links", {}),
    }
    return {
        "kind": "measurement",
        "target": {"type": "recording", "semantic_id": "global", "field": "measurement_points"},
        "value": _ref(f"annotations/{prefix}measurement_{index:03d}_points"),
        "links": links,
        **extra,
    }


def _store(
    tmp_path: Path,
    annotations: list[dict[str, Any]],
    *,
    frame_time: float | None = FRAME_TIMES[1],
    sectors: list[dict[str, Any]] | None = None,
    merged_documents: list[dict[str, Any]] | None = None,
    tracks: list[dict[str, Any]] | None = None,
) -> Path:
    path = tmp_path / "recording.zarr"
    group = zarr.open_group(path, mode="w")
    for sector in [_sector()] if sectors is None else sectors:
        frames = np.arange(3 * 8 * 6, dtype=np.uint8).reshape(3, 8, 6)
        group.create_array(sector["frames"]["zarr_path"], data=frames)
        group.create_array(sector["timestamps"]["zarr_path"], data=np.asarray(FRAME_TIMES, dtype=np.float64))
    documents = [(annotations, "")] + [(doc["annotations"], "document_00/") for doc in merged_documents or []]
    for entries, _ in documents:
        for entry in entries:
            if "value" in entry:
                group.create_array(entry["value"]["zarr_path"], data=SOURCE_POINTS)
            if "time" in entry and frame_time is None:
                del entry["time"]
            elif "time" in entry:
                group.create_array(entry["time"]["zarr_path"], data=np.asarray([frame_time], dtype=np.float64))
    group.create_array("annotations/bmode_ecg_qrs", data=np.asarray([0.5, 1.5], dtype=np.float64))
    manifest: dict[str, Any] = {
        "manifest_type": "2d",
        "sectors": [_sector()] if sectors is None else sectors,
        "tracks": tracks or [],
        "annotations": annotations,
    }
    if merged_documents:
        manifest["merged_documents"] = merged_documents
    group.attrs["recording_manifest"] = manifest
    return path


def _spatial(index: int = 0, **extra: Any) -> dict[str, Any]:
    return _measurement(index, time=_ref(f"timestamps/measurement_{index:03d}_frame_time"), **extra)


def _load_one(path: Path, **filters: Any) -> Measurement:
    with open_recording(path) as store:
        measurements = store.load_measurements(**filters)
    assert len(measurements) == 1
    return measurements[0]


def test_spatial_points_are_returned_in_the_image_frame_with_their_frame(tmp_path: Path) -> None:
    measurement = _load_one(_store(tmp_path, [_spatial()]))

    np.testing.assert_allclose(measurement.points, IMAGE_POINTS)
    assert measurement.kind == "dimension_lvot_diam"
    assert measurement.measure_id == "Cardiac/2D/Dimension/LVOT Diam"
    assert measurement.axes == ("x_m", "y_m")
    assert measurement.data_path == "data/2d_brightness_mode"
    assert (measurement.frame_index, measurement.frame_status) == (1, "exact")
    assert measurement.time_s == pytest.approx(FRAME_TIMES[1])
    assert measurement.geometry is not None and measurement.geometry.depth_end_m == pytest.approx(0.12)
    assert measurement.length_m == pytest.approx(float(np.linalg.norm(SOURCE_POINTS[1] - SOURCE_POINTS[0])))


def test_points_declared_as_already_transformed_are_not_reflected_again(tmp_path: Path) -> None:
    entry = _spatial(links={"source_to_canonical_transform": "reflect_x"})

    np.testing.assert_allclose(_load_one(_store(tmp_path, [entry])).points, SOURCE_POINTS)


@pytest.mark.parametrize(
    ("frame_time", "frame_index", "status"),
    [
        (1.02 + 5e-5, 1, "exact"),
        (1.027, 1, "between_frames"),
        (1.10, None, "no_matching_frame"),
        (0.95, None, "no_matching_frame"),
        (None, None, "no_time"),
    ],
)
def test_frame_is_only_resolved_when_the_store_holds_it(
    tmp_path: Path, frame_time: float | None, frame_index: int | None, status: str
) -> None:
    measurement = _load_one(_store(tmp_path, [_spatial()], frame_time=frame_time))

    assert (measurement.frame_index, measurement.frame_status) == (frame_index, status)
    assert (measurement.frame_time_offset_s is None) == (frame_index is None)
    np.testing.assert_allclose(measurement.points, IMAGE_POINTS)


def test_two_plane_recording_leaves_the_plane_unresolved(tmp_path: Path) -> None:
    sectors = [_sector("2d_brightness_mode_plane_0"), _sector("2d_brightness_mode_plane_1")]
    measurement = _load_one(_store(tmp_path, [_spatial()], sectors=sectors))

    assert (measurement.data_path, measurement.frame_index, measurement.frame_status) == (None, None, "unknown_plane")
    np.testing.assert_allclose(measurement.points, IMAGE_POINTS)


def test_spectral_and_time_depth_points_are_returned_as_stored(tmp_path: Path) -> None:
    tracks = [{"semantic_id": "mmode", "data": _ref("data/1d_motion_mode")}]
    annotations = [
        _measurement(0, "spectral", target_data=_ref("data/1d_pulsed_wave_doppler")),
        _measurement(1, "time_depth", links={"measurement_kind": "tapse"}),
    ]
    with open_recording(_store(tmp_path, annotations, tracks=tracks)) as store:
        spectral, time_depth = store.load_measurements()
        only_tapse = store.load_measurements("tapse")
        only_spectral = store.load_measurements(coordinate_type="spectral")

    np.testing.assert_allclose(spectral.points, SOURCE_POINTS)
    np.testing.assert_allclose(time_depth.points, SOURCE_POINTS)
    assert (spectral.axes, spectral.data_path) == (("time_s", "velocity_mps"), "data/1d_pulsed_wave_doppler")
    assert (time_depth.axes, time_depth.data_path) == (("time_s", "depth_m"), "data/1d_motion_mode")
    assert spectral.frame_status == time_depth.frame_status == "not_applicable"
    assert spectral.length_m is None
    assert [item.kind for item in only_tapse] == ["tapse"]
    assert [item.coordinate_type for item in only_spectral] == ["spectral"]


def test_measurements_of_merged_documents_use_their_own_sectors(tmp_path: Path) -> None:
    nested = _measurement(0, prefix="document_00/", time=_ref("timestamps/document_00/measurement_000_frame_time"))
    merged = [{"manifest_type": "2d", "sectors": [_sector()], "annotations": [nested]}]
    with open_recording(_store(tmp_path, [_spatial()], merged_documents=merged)) as store:
        measurements = store.load_measurements()

    assert [item.ref.document_path for item in measurements if item.ref is not None] == [(), (0,)]
    assert all(item.frame_status == "exact" for item in measurements)


def test_list_annotations_describes_entries_with_values_and_unlisted_arrays(tmp_path: Path) -> None:
    inventory = {"kind": "measurement_inventory", "links": {"measurement_count": 1}}
    with open_recording(_store(tmp_path, [_spatial(), inventory])) as store:
        infos = store.list_annotations()

    assert [(info.kind, info.name, info.path, info.shape) for info in infos] == [
        ("measurement", "dimension_lvot_diam", "annotations/measurement_000_points", (2, 2)),
        ("array", "bmode_ecg_qrs", "annotations/bmode_ecg_qrs", (2,)),
    ]
    assert infos[0].coordinate_type == "spatial"


def test_plot_measurements_draws_on_the_measured_frame(tmp_path: Path) -> None:
    with open_recording(_store(tmp_path, [_spatial()])) as store:
        measurement = store.load_measurements()[0]
        figure = plot_measurements(store, measurement, height_px=64)

    axes = figure.axes[0]
    np.testing.assert_allclose(axes.lines[0].get_xydata(), IMAGE_POINTS, rtol=1e-6)
    assert axes.texts[0].get_text() == "LVOT Diam\n25.0 mm"
    pixels = axes.images[0].get_array()
    assert pixels.shape[0] == 64
    assert np.ma.getmaskarray(pixels)[0, 0] and not np.ma.getmaskarray(pixels).all()
    assert figure.patch.get_alpha() == 0.0


def test_plot_measurements_rejects_measurements_without_a_frame(tmp_path: Path) -> None:
    with open_recording(_store(tmp_path, [_spatial()], frame_time=None)) as store:
        measurement = store.load_measurements()[0]
        with pytest.raises(ValueError, match="no stored B-mode frame"):
            plot_measurements(store, measurement)


def _spectral_store(tmp_path: Path, coordinate_type: str = "spectral") -> Path:
    path = tmp_path / "spectral.zarr"
    group = zarr.open_group(path, mode="w")
    group.create_array("data/1d_pulsed_wave_doppler", data=np.arange(25, dtype=np.float32).reshape(5, 5))
    group.create_array("timestamps/1d_pulsed_wave_doppler", data=np.asarray([0.0, 0.1, 0.2, 0.3, 0.4]))
    group.create_array("annotations/measurement_000_points", data=np.asarray([[0.1, 0.0], [0.25, -2.0]]))
    entry = _measurement(
        0,
        coordinate_type,
        target_data=_ref("data/1d_pulsed_wave_doppler"),
        links={"geometry_kind": "polyline", "measure_id": "Cardiac/SD/Aortic/LVOT Trace/Manual"},
    )
    track = {
        "semantic_id": "1d_pulsed_wave_doppler",
        "data": _ref("data/1d_pulsed_wave_doppler"),
        "timestamps": _ref("timestamps/1d_pulsed_wave_doppler"),
        "nyquist_mps": 2.0,
        "spectral_row_baseline_frac": 0.25,
    }
    group.attrs["recording_manifest"] = {"manifest_type": "2d", "tracks": [track], "annotations": [entry]}
    return path


def test_plot_measurements_draws_spectral_points_on_their_spectrum(tmp_path: Path) -> None:
    with open_recording(_spectral_store(tmp_path)) as store:
        figure = plot_measurements(store, store.load_measurements(), height_px=64)

    axes = figure.axes[0]
    # Rows run from +2 m/s to -2 m/s with zero velocity on row 1, columns are 0.1 s apart.
    np.testing.assert_allclose(axes.lines[0].get_xydata(), [[1.0, 1.0], [2.5, 4.0]])
    assert axes.texts[0].get_text() == "LVOT\nTrace/Manual"
    assert axes.images[0].get_array().shape == (5, 5)


def test_plot_measurements_rejects_time_depth_measurements(tmp_path: Path) -> None:
    with open_recording(_spectral_store(tmp_path, "time_depth")) as store:
        with pytest.raises(ValueError, match="not drawn"):
            plot_measurements(store, store.load_measurements())


def test_times_in_overlapping_sweeps_are_not_placed() -> None:
    clock = np.asarray([0.0, 0.1, 0.2, 0.1, 0.2, 0.3])
    columns, runs = _columns_at_times(clock, np.asarray([0.05, 0.15, 0.25]))

    np.testing.assert_allclose(columns, [0.5, np.nan, 4.5])
    assert runs.tolist() == [0, -1, 1]


def test_catalog_names_measurements_and_finds_their_recordings(tmp_path: Path) -> None:
    _store(tmp_path, [_spatial(), _measurement(1, "time_depth", links={"measurement_kind": "tapse"})])
    _spectral_store(tmp_path)

    def record(name: str, annotations: tuple[str, ...] = ("measurement_points",)) -> Any:
        return replace(make_record(name), source_root=tmp_path, annotations=annotations)

    _la_trace_stores(tmp_path)
    records = (
        record("recording"),
        record("spectral"),
        record("not_downloaded"),
        record("plain", ()),
        record("still", ("la_trace",)),
    )
    catalog = CroissantCatalog(path=tmp_path / "croissant.json", recordings=records)

    assert catalog.measurement_names() == ("dimension_lvot_diam", "tapse", "volume_laesv_a4c")
    assert catalog.with_measurement("dimension_lvot_diam") == records[:2]
    assert catalog.with_measurement("volume_laesv_a4c") == records[4:]
    assert catalog.with_measurement("tapse") == records[:1]
    assert catalog.with_measurement("absent") == ()


def _la_trace_stores(tmp_path: Path, *, listed: bool = True) -> Path:
    """A still holding a recovered left-atrial outline, and the loop recording its frame belongs to."""
    plane = _sector("2d_brightness_mode_plane_0")
    loop = zarr.open_group(tmp_path / "loop.zarr", mode="w")
    loop.create_array("data/2d_brightness_mode_plane_0", data=np.zeros((3, 8, 6), dtype=np.uint8))
    loop.create_array("timestamps/2d_brightness_mode_plane_0", data=np.asarray(FRAME_TIMES, dtype=np.float64))
    loop.attrs["recording_manifest"] = {"manifest_type": "2d", "sectors": [plane], "annotations": []}

    still = zarr.open_group(tmp_path / "still.zarr", mode="w")
    square = np.asarray([[0.0, 0.05], [0.01, 0.05], [0.01, 0.06], [0.0, 0.06]], dtype=np.float32)
    still.create_array("derived/la_trace/plane_0/polygon_xy_m", data=square)
    still.create_array("timestamps/la_trace/plane_0/frame_time", data=np.asarray([FRAME_TIMES[2]]))
    trace = {
        "kind": "recovered_la_trace",
        "target": {"type": "linked_panel", "semantic_id": "source_plane_0"},
        "target_data": {
            "recording_id": "loop",
            "recording_zarr_path": "exams/exam/loop.zarr",
            "array_path": "data/2d_brightness_mode_plane_0",
        },
        "time": _ref("timestamps/la_trace/plane_0/frame_time"),
        "value": _ref("derived/la_trace/plane_0/polygon_xy_m"),
        "links": {"geometry_kind": "polygon", "plane_index": 0, "view": "A4C"},
    }
    burned_in = {
        "kind": "burned_in_trace",
        "links": {
            "family": "la_volume",
            "measure_ids": ["Cardiac/2D/Volume/LAESV A4C"],
            "measurement_kinds": ["volume_laesv_a4c"],
            "planes": [{"view": "A4C", "measure_id": "Cardiac/2D/Volume/LAESV A4C", "plane_index": 0}],
        },
    }
    annotations = [trace, burned_in] if listed else [trace]
    still.attrs["recording_manifest"] = {"manifest_type": "2d", "sectors": [], "annotations": annotations}
    return tmp_path / "still.zarr"


def test_left_atrial_outline_is_a_measurement_on_its_loop_frame(tmp_path: Path) -> None:
    with open_recording(_la_trace_stores(tmp_path)) as store:
        outline = store.load_measurements("volume_laesv_a4c")[0]
        figure = plot_measurements(store, outline, height_px=64)

    # Recovered outlines are stored in the image frame, so they are not reflected.
    np.testing.assert_allclose(outline.points[1], [0.01, 0.05])
    assert (outline.geometry_kind, outline.measure_id) == ("polygon", "Cardiac/2D/Volume/LAESV A4C")
    assert outline.recording is not None and outline.recording.relative_zarr_path == "loop.zarr"
    assert (outline.data_path, outline.frame_index, outline.frame_status) == (
        "data/2d_brightness_mode_plane_0",
        2,
        "exact",
    )
    assert outline.area_m2 == pytest.approx(1e-4)
    assert outline.length_m is None
    axes = figure.axes[0]
    assert len(axes.lines[0].get_xydata()) == 5  # the outline is closed
    assert axes.texts[0].get_text() == "LAESV A4C\n1.0 cm²"


def test_left_atrial_outline_without_a_listed_measurement_is_named_after_its_view(tmp_path: Path) -> None:
    with open_recording(_la_trace_stores(tmp_path, listed=False)) as store:
        assert [item.kind for item in store.load_measurements()] == ["la_trace_a4c"]


def test_left_atrial_outline_is_returned_when_its_loop_is_not_on_disk(tmp_path: Path) -> None:
    still = _la_trace_stores(tmp_path)
    (tmp_path / "loop.zarr").rename(tmp_path / "elsewhere.zarr")

    with open_recording(still) as store:
        outline = store.load_measurements()[0]
        with pytest.raises(ValueError, match="no stored B-mode frame"):
            plot_measurements(store, outline)

    assert (outline.frame_index, outline.frame_status) == (None, "linked_recording_missing")
    assert outline.points.shape == (4, 2)
    assert outline.recording is not None and outline.recording.recording_id == "loop"
