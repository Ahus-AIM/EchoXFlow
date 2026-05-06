from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from echoxflow import (
    PredictionArray,
    RecordingArray,
    RecordingRecord,
    build_prediction_manifest,
    find_derived_recordings,
    find_recordings,
    find_source_recordings,
    load_croissant,
    open_recording,
    prediction_array_entry,
    rasterize_beamspace_mask,
    rasterize_beamspace_volume_mask,
    rasterize_packed_mesh_volume_masks,
    resample_sector,
    write_prediction_recording,
    write_recording,
)
from echoxflow.croissant import RecordingRelationship, linked_frame_timestamp_paths
from echoxflow.scan import (
    SectorGeometry,
    SphericalGeometry,
    mesh_frame_indices_for_volume_timestamps,
    spherical_geometry_from_metadata,
)
from echoxflow.streams import AnnotationMaskStream, MeshFaceStream, MeshFrameOffsetsStream, MeshPointStream

zarr = pytest.importorskip("zarr")


def test_recording_store_loads_temporal_stream_slice(tmp_path: Path) -> None:
    group = zarr.open_group(tmp_path / "case.zarr", mode="w")
    group.create_array("data/2d_brightness_mode", data=np.arange(5 * 2 * 2, dtype=np.uint8).reshape(5, 2, 2))
    group.create_array("timestamps/2d_brightness_mode", data=np.arange(5, dtype=np.float32))

    stream = open_recording(tmp_path / "case.zarr").load_stream_slice("2d_brightness_mode", 1, 4)

    assert stream.data.shape == (3, 2, 2)
    assert stream.timestamps is not None
    assert stream.timestamps.tolist() == [1.0, 2.0, 3.0]


def test_recording_store_caches_complete_arrays_for_sliced_reads(tmp_path: Path) -> None:
    group = zarr.open_group(tmp_path / "case.zarr", mode="w")
    data = np.arange(5 * 2 * 2, dtype=np.uint8).reshape(5, 2, 2)
    group.create_array("data/2d_brightness_mode", data=data)
    group.create_array("timestamps/2d_brightness_mode", data=np.arange(5, dtype=np.float32))
    cache_dir = tmp_path / "cache"

    first = open_recording(tmp_path / "case.zarr", cache_dir=cache_dir).load_stream_slice("2d_brightness_mode", 1, 4)
    second = open_recording(tmp_path / "case.zarr", cache_dir=cache_dir).load_stream_slice("2d_brightness_mode", 2, 5)

    assert first.data.tolist() == data[1:4].tolist()
    assert second.data.tolist() == data[2:5].tolist()
    assert len(list((cache_dir / "arrays").glob("*/*.npz"))) == 2


def test_recording_store_cache_include_limits_written_arrays(tmp_path: Path) -> None:
    group = zarr.open_group(tmp_path / "case.zarr", mode="w")
    group.create_array("data/2d_brightness_mode", data=np.arange(5 * 2 * 2, dtype=np.uint8).reshape(5, 2, 2))
    group.create_array("timestamps/2d_brightness_mode", data=np.arange(5, dtype=np.float32))
    cache_dir = tmp_path / "cache"

    stream = open_recording(
        tmp_path / "case.zarr",
        cache_dir=cache_dir,
        cache_include=("data/2d_brightness_mode",),
    ).load_stream_slice("2d_brightness_mode", 1, 4)

    assert stream.timestamps is not None
    assert len(list((cache_dir / "arrays").glob("*/*.npz"))) == 1


def test_recording_store_reads_3d_frame_timestamps_from_manifest_timelines(tmp_path: Path) -> None:
    group = zarr.open_group(tmp_path / "case.zarr", mode="w")
    group.create_array("data/3d_brightness_mode", data=np.zeros((2, 1, 1, 1), dtype=np.uint8))
    group.create_array("timelines/volume_frame_timestamps", data=np.asarray([0.25, 0.5], dtype=np.float32))
    group.attrs["recording_manifest"] = {
        "manifest_type": "3d",
        "sectors": [
            {
                "semantic_id": "bmode",
                "frames": {"array_path": "data/3d_brightness_mode", "format": "zarr_array"},
                "geometry": {
                    "coordinate_system": "spherical_sector_3d",
                    "DepthStart": 0.0,
                    "DepthEnd": 0.1,
                    "Width": 0.4,
                },
            }
        ],
        "timelines": {"frame_timestamps": {"array_path": "timelines/volume_frame_timestamps", "format": "zarr_array"}},
    }

    stream = open_recording(tmp_path / "case.zarr").load_stream("3d_brightness_mode")

    assert stream.timestamps_path == "timelines/volume_frame_timestamps"
    assert stream.timestamps is not None
    assert stream.timestamps.tolist() == [0.25, 0.5]


def test_mesh_frame_indices_align_by_qrs_phase_not_video_start() -> None:
    mesh_times = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    volume_times = np.asarray([10.0, 10.25, 10.5, 10.75, 10.99], dtype=np.float64)
    metadata = {"metadata": {"qrs_trigger_times": [10.0, 11.0, 12.0]}}

    indices = mesh_frame_indices_for_volume_timestamps(
        mesh_times,
        volume_times,
        metadata,
        mesh_frame_count=mesh_times.size,
    )

    assert indices == (0, 1, 2, 3, 4)


def test_spectral_stream_slice_uses_current_reader_time_axis(tmp_path: Path) -> None:
    group = zarr.open_group(tmp_path / "case.zarr", mode="w")
    group.create_array("data/1d_pulsed_wave_doppler", data=np.arange(6 * 4, dtype=np.float32).reshape(6, 4))
    group.create_array("timestamps/1d_pulsed_wave_doppler", data=np.arange(6, dtype=np.float32))

    stream = open_recording(tmp_path / "case.zarr").load_stream_slice("1d_pulsed_wave_doppler", 2, 5)

    assert stream.data.shape == (3, 4)
    assert stream.data.tolist() == np.arange(6 * 4, dtype=np.float32).reshape(6, 4)[2:5].tolist()
    assert stream.timestamps is not None
    assert stream.timestamps.tolist() == [2.0, 3.0, 4.0]


def test_spectral_timestamp_validation_uses_current_reader_time_axis() -> None:
    from echoxflow.streams import stream_from_arrays

    with pytest.raises(ValueError, match="temporal samples"):
        stream_from_arrays(
            data_path="data/1d_pulsed_wave_doppler",
            data=np.zeros((6, 4), dtype=np.float32),
            timestamps_path="timestamps/1d_pulsed_wave_doppler",
            timestamps=np.zeros((4,), dtype=np.float32),
            sample_rate_hz=None,
        )


def test_open_recording_supports_zip_zarr_store(tmp_path: Path) -> None:
    storage = pytest.importorskip("zarr.storage")
    zip_store = storage.ZipStore(tmp_path / "published.zarr", mode="w")
    try:
        group = zarr.open_group(store=zip_store, mode="w")
        group.create_array("data/ecg", data=np.asarray([1.0, 2.0], dtype=np.float32))
    finally:
        zip_store.close()

    store = open_recording(tmp_path / "published.zarr")

    assert store.load_stream("ecg").data.tolist() == [1.0, 2.0]


def test_croissant_relationship_helpers_find_source_and_derived_records(tmp_path: Path) -> None:
    _write_croissant(
        tmp_path,
        [
            _row(
                "bmode",
                content_types=["2d_brightness_mode"],
                array_paths=["data/2d_brightness_mode", "timestamps/2d_brightness_mode"],
            ),
            _row(
                "mesh",
                content_types=["3d_left_ventricle_mesh"],
                array_paths=["data/3d_left_ventricle_mesh_points"],
                source_recording_id="bmode",
            ),
        ],
    )
    catalog = load_croissant(root=tmp_path)
    bmode = catalog.by_recording_id("bmode")
    mesh = catalog.by_recording_id("mesh")

    assert bmode is not None
    assert mesh is not None
    assert find_source_recordings(mesh, croissant=catalog) == (bmode,)
    assert find_derived_recordings(bmode, croissant=catalog) == (mesh,)


def test_croissant_relationship_helpers_parse_public_recording_links(tmp_path: Path) -> None:
    _write_croissant(
        tmp_path,
        [
            _row("derived", content_types=["3d_left_ventricle_mesh"], array_paths=["data/3d_left_ventricle_mesh"]),
            _row("source", content_types=["2d_brightness_mode"], array_paths=["data/2d_brightness_mode"]),
        ],
        recording_links=[
            {
                "recording_links/source_recording_id": "derived",
                "recording_links/linked_recording_id": "source",
                "recording_links/panel_role_id": "2d_brightness_mode",
                "recording_links/linked_frame_array_id": "source/data/2d_brightness_mode",
                "recording_links/linked_timestamp_array_id": "source/timestamps/2d_brightness_mode",
            }
        ],
    )
    catalog = load_croissant(root=tmp_path)
    derived = catalog.by_recording_id("derived")
    source = catalog.by_recording_id("source")

    assert derived is not None
    assert source is not None
    assert find_source_recordings(derived, croissant=catalog) == (source,)
    assert source.relationships[0].panel_role_id == "2d_brightness_mode"
    assert linked_frame_timestamp_paths(source, "2d_brightness_mode")[0] == "timestamps/2d_brightness_mode"


def test_find_recordings_filters_by_croissant_frame_counts(tmp_path: Path) -> None:
    _write_croissant(
        tmp_path,
        [
            _row(
                "single_frame",
                content_types=["2d_brightness_mode", "2d_color_doppler_velocity", "2d_color_doppler_power"],
                array_paths=[
                    "data/2d_brightness_mode",
                    "data/2d_color_doppler_velocity",
                    "data/2d_color_doppler_power",
                ],
                frame_counts={
                    "2d_brightness_mode": 1,
                    "2d_color_doppler_velocity": 4,
                    "2d_color_doppler_power": 4,
                },
            ),
            _row(
                "multi_frame",
                content_types=["2d_brightness_mode", "2d_color_doppler_velocity", "2d_color_doppler_power"],
                array_paths=[
                    "data/2d_brightness_mode",
                    "data/2d_color_doppler_velocity",
                    "data/2d_color_doppler_power",
                ],
                frame_counts={
                    "2d_brightness_mode": 3,
                    "2d_color_doppler_velocity": 4,
                    "2d_color_doppler_power": 4,
                },
            ),
        ],
    )

    records = find_recordings(
        root=tmp_path,
        array_paths=("data/2d_brightness_mode", "data/2d_color_doppler_velocity", "data/2d_color_doppler_power"),
        require_all=True,
        min_frame_counts={"data/2d_brightness_mode": 2},
        max_frame_counts={"2d_color_doppler_velocity": 4},
    )

    assert [record.recording_id for record in records] == ["multi_frame"]
    assert records[0].frame_count("data/2d_brightness_mode") == 3


def test_find_recordings_exposes_and_filters_3d_stitch_beat_count(tmp_path: Path) -> None:
    _write_croissant(
        tmp_path,
        [
            _row(
                "single_beat_3d",
                content_types=["3d_brightness_mode"],
                array_paths=["data/3d_brightness_mode"],
                frame_counts={"3d_brightness_mode": 2},
                stitch_beat_count=1,
            ),
            _row(
                "stitched_3d",
                content_types=["3d_brightness_mode"],
                array_paths=["data/3d_brightness_mode"],
                frame_counts={"3d_brightness_mode": 8},
                stitch_beat_count=4,
            ),
        ],
    )

    catalog = load_croissant(root=tmp_path)
    stitched = find_recordings(croissant=catalog, content_type="3d_brightness_mode", min_stitch_beat_count=2)

    assert [record.recording_id for record in stitched] == ["stitched_3d"]
    assert stitched[0].stitch_beat_count == 4
    assert stitched[0].is_stitched_3d is True


def test_linked_frame_timestamp_paths_prefers_explicit_relationships() -> None:
    record = RecordingRecord(
        exam_id="exam",
        recording_id="rec",
        zarr_path="rec.zarr",
        modes=("2d_color_doppler_velocity",),
        content_types=("2d_color_doppler_velocity",),
        frame_counts_by_content_type={},
        median_delta_time_by_content_type={},
        array_paths=("data/2d_color_doppler_velocity", "timestamps/2d_color_doppler"),
        relationships=(
            RecordingRelationship(
                exam_id="exam",
                source_recording_id="rec",
                target_recording_id="rec",
                relationship_type="timestamps",
                source_array_path="data/2d_color_doppler_velocity",
                target_array_path="timestamps/2d_color_doppler",
            ),
        ),
    )

    assert linked_frame_timestamp_paths(record, "2d_color_doppler_velocity")[0] == "timestamps/2d_color_doppler"


def test_typed_annotation_and_mesh_streams_validate_shapes() -> None:
    from echoxflow.streams import stream_from_arrays

    assert isinstance(
        stream_from_arrays(
            data_path="data/3d_left_ventricle_mask",
            data=np.zeros((2, 3, 3), dtype=np.float32),
            timestamps_path=None,
            timestamps=None,
            sample_rate_hz=None,
        ),
        AnnotationMaskStream,
    )
    assert isinstance(
        stream_from_arrays(
            data_path="data/3d_left_ventricle_mesh_points",
            data=np.zeros((4, 3), dtype=np.float32),
            timestamps_path=None,
            timestamps=None,
            sample_rate_hz=None,
        ),
        MeshPointStream,
    )
    assert isinstance(
        stream_from_arrays(
            data_path="data/3d_left_ventricle_mesh_faces",
            data=np.asarray([[0, 1, 2]], dtype=np.int32),
            timestamps_path=None,
            timestamps=None,
            sample_rate_hz=None,
        ),
        MeshFaceStream,
    )
    assert isinstance(
        stream_from_arrays(
            data_path="data/3d_left_ventricle_mesh_frame_offsets",
            data=np.asarray([0, 2, 4], dtype=np.int64),
            timestamps_path=None,
            timestamps=None,
            sample_rate_hz=None,
        ),
        MeshFrameOffsetsStream,
    )


def test_packed_mesh_annotation_loader_validates_conventional_paths(tmp_path: Path) -> None:
    group = zarr.open_group(tmp_path / "case.zarr", mode="w")
    group.create_array("data/3d_left_ventricle_mesh_points", data=np.zeros((4, 3), dtype=np.float32))
    group.create_array("data/3d_left_ventricle_mesh_faces", data=np.asarray([[0, 1, 2]], dtype=np.int32))
    group.create_array("data/3d_left_ventricle_mesh_frame_offsets", data=np.asarray([0, 2, 4], dtype=np.int64))

    mesh = open_recording(tmp_path / "case.zarr").load_packed_mesh_annotation()

    assert mesh.points.shape == (4, 3)
    assert mesh.faces.tolist() == [[0, 1, 2]]


def test_packed_mesh_annotation_loader_exposes_grouped_frame_views(tmp_path: Path) -> None:
    group = zarr.open_group(tmp_path / "case.zarr", mode="w")
    mesh_group = group.create_group("data/3d_left_ventricle_mesh")
    mesh_group.create_array(
        "point_values",
        data=np.asarray(
            [
                [0.0, 0.0, 1.0],
                [0.1, 0.0, 1.0],
                [0.0, 0.1, 1.0],
                [0.0, 0.0, 1.2],
                [0.1, 0.0, 1.2],
                [0.0, 0.1, 1.2],
            ],
            dtype=np.float32,
        ),
    )
    mesh_group.create_array("face_values", data=np.asarray([[0, 1, 2], [0, 1, 2]], dtype=np.int32))
    mesh_group.create_array("point_frame_offsets", data=np.asarray([0, 3, 6], dtype=np.int64))
    mesh_group.create_array("face_frame_offsets", data=np.asarray([0, 1, 2], dtype=np.int64))
    mesh_group.create_array("point_component_offsets", data=np.asarray([0, 2], dtype=np.int64))
    mesh_group.create_array("face_component_offsets", data=np.asarray([0, 2], dtype=np.int64))
    mesh_group.create_array("component_names", data=np.asarray(["lv"], dtype="S2"))
    group.create_array("timestamps/3d_left_ventricle_mesh", data=np.asarray([0.0, 1.0], dtype=np.float32))

    mesh = open_recording(tmp_path / "case.zarr").load_packed_mesh_annotation()

    assert mesh.frame_count == 2
    assert mesh.component_names == ("lv",)
    assert mesh.frame(1).points.shape == (3, 3)
    assert mesh.frame(1).faces.tolist() == [[0, 1, 2]]
    assert mesh.frame(1).timestamp == 1.0


def test_spectral_metadata_extracts_cursor_baseline_and_velocity_axis(tmp_path: Path) -> None:
    group = zarr.open_group(tmp_path / "case.zarr", mode="w")
    group.create_array("data/1d_pulsed_wave_doppler", data=np.zeros((3, 5), dtype=np.float32))
    group.attrs["recording_manifest"] = {
        "manifest_type": "2d",
        "tracks": [{"semantic_id": "1d_pulsed_wave_doppler"}],
        "spectral_metadata": {
            "cursor_box": [1, 2, 3, 4],
            "baseline_row": 2,
            "nyquist_limit_mps": 1.0,
        },
    }

    metadata = open_recording(tmp_path / "case.zarr").spectral_metadata("1d_pulsed_wave_doppler")

    assert metadata.cursor_box is not None
    assert metadata.baseline_row == 2.0
    assert metadata.row_velocity_mps is not None
    assert metadata.row_velocity_mps.tolist() == [1.0, 0.5, 0.0, -0.5, -1.0]


def test_sector_resampling_and_beamspace_rasterization() -> None:
    source = SectorGeometry.from_center_width(
        depth_start_m=0.0,
        depth_end_m=2.0,
        tilt_rad=0.0,
        width_rad=1.0,
        grid_shape=(3, 3),
    )
    target = SectorGeometry.from_center_width(
        depth_start_m=0.0,
        depth_end_m=2.0,
        tilt_rad=0.0,
        width_rad=1.0,
        grid_shape=(3, 3),
    )
    image = np.arange(9, dtype=np.float32).reshape(3, 3)

    resampled = resample_sector(image, source, target, interpolation="nearest")
    mask = rasterize_beamspace_mask(
        np.asarray([[-0.1, 1.0], [0.1, 1.0], [0.0, 1.5]], dtype=np.float32),
        source,
        output_shape=(8, 8),
    )

    assert np.allclose(resampled, image)
    assert mask.any()


def test_mesh_volume_rasterization_returns_frame_volume_masks(tmp_path: Path) -> None:
    group = zarr.open_group(tmp_path / "case.zarr", mode="w")
    mesh_group = group.create_group("data/3d_left_ventricle_mesh")
    mesh_group.create_array(
        "point_values",
        data=np.asarray(
            [[-0.05, 0.0, 1.0], [0.05, 0.0, 1.0], [0.0, 0.05, 1.0], [0.0, 0.0, 1.2]],
            dtype=np.float32,
        ),
    )
    mesh_group.create_array("face_values", data=np.asarray([[0, 1, 2], [0, 1, 3]], dtype=np.int32))
    mesh_group.create_array("point_frame_offsets", data=np.asarray([0, 4], dtype=np.int64))
    mesh_group.create_array("face_frame_offsets", data=np.asarray([0, 2], dtype=np.int64))
    mesh = open_recording(tmp_path / "case.zarr").load_packed_mesh_annotation()
    geometry = SphericalGeometry(
        depth_start_m=0.5,
        depth_end_m=1.5,
        azimuth_width_rad=0.4,
        elevation_width_rad=0.4,
    )

    mask = rasterize_beamspace_volume_mask(mesh.frame(0).points, mesh.frame(0).faces, geometry, output_shape=(8, 8, 8))
    masks = rasterize_packed_mesh_volume_masks(mesh, geometry, output_shape=(8, 8, 8))

    assert mask.shape == (8, 8, 8)
    assert mask.any()
    assert masks.shape == (1, 8, 8, 8)
    assert masks.any()


def test_prediction_manifest_helpers_are_typed() -> None:
    entry = prediction_array_entry("3d_left_ventricle_mask", np.zeros((2, 3, 4), dtype=np.float32))
    manifest = build_prediction_manifest(
        exam_id="exam",
        recording_id="prediction",
        source_recording_id="source",
        arrays=[entry],
    )

    assert manifest.to_dict()["arrays"][0]["data_path"] == "data/3d_left_ventricle_mask"
    assert manifest.to_dict()["source_recording_id"] == "source"


def test_prediction_recording_writer_round_trips_through_open_recording(tmp_path: Path) -> None:
    output_path = tmp_path / "prediction.zarr"
    manifest = write_prediction_recording(
        output_path,
        exam_id="exam",
        recording_id="prediction",
        source_recording_id="source",
        arrays=[
            PredictionArray(
                "3d_left_ventricle_mask",
                np.ones((2, 3, 4), dtype=np.float32),
                timestamps=np.asarray([0.0, 1.0], dtype=np.float32),
            )
        ],
        zarr_path="prediction.zarr",
        croissant_path=tmp_path / "croissant.json",
    )

    store = open_recording(output_path)

    assert manifest.to_dict()["arrays"][0]["timestamps_path"] == "timestamps/3d_left_ventricle_mask"
    assert store.load_stream("3d_left_ventricle_mask").data.shape == (2, 3, 4)
    assert store.load_timestamps("3d_left_ventricle_mask") is not None
    catalog = load_croissant(root=tmp_path)
    record = catalog.by_recording_id("prediction")
    assert record is not None
    assert record.has_array_path("data/3d_left_ventricle_mask")


def test_recording_writer_round_trips_ecg_and_croissant_metadata(tmp_path: Path) -> None:
    output_path = tmp_path / "preview.zarr"

    record = write_recording(
        output_path,
        exam_id="exam",
        recording_id="preview",
        source_recording_id="source",
        arrays=[
            RecordingArray(
                "2d_brightness_mode",
                np.zeros((2, 4, 4), dtype=np.uint8),
                timestamps=np.asarray([0.0, 0.1], dtype=np.float32),
            ),
            RecordingArray(
                "ecg",
                np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
                timestamps=np.asarray([0.0, 0.05, 0.1], dtype=np.float32),
            ),
        ],
        attrs={"recording_manifest": {"sectors": [{"semantic_id": "bmode"}]}},
        zarr_path="preview.zarr",
        croissant_path=tmp_path / "croissant.json",
    )

    store = open_recording(record, root=tmp_path)
    catalog = load_croissant(root=tmp_path)

    assert store.load_stream("2d_brightness_mode").data.shape == (2, 4, 4)
    assert store.load_stream("ecg").data.tolist() == [0.0, 1.0, 0.0]
    assert catalog.by_recording_id("preview") is not None
    assert record.median_delta_time("2d_brightness_mode") == pytest.approx(0.1)


def test_recording_writer_publishes_split_3d_volume_annotation_links(tmp_path: Path) -> None:
    write_recording(
        tmp_path / "volume.zarr",
        exam_id="exam",
        recording_id="volume",
        arrays=[
            RecordingArray(
                "3d_brightness_mode",
                np.zeros((4, 2, 2, 2), dtype=np.uint8),
                timestamps=np.asarray([0.0, 0.1, 0.2, 0.3], dtype=np.float32),
            )
        ],
        attrs={"recording_manifest": {"stitch_beat_count": 4}},
        zarr_path="volume.zarr",
        croissant_path=tmp_path / "croissant.json",
    )
    write_recording(
        tmp_path / "mesh.zarr",
        exam_id="exam",
        recording_id="mesh",
        arrays=[
            RecordingArray("3d_left_ventricle_mesh_points", np.zeros((4, 3), dtype=np.float32)),
            RecordingArray("3d_left_ventricle_mesh_faces", np.asarray([[0, 1, 2]], dtype=np.int32)),
        ],
        attrs={
            "recording_manifest": {
                "linked_volume": {
                    "recording_id": "volume",
                    "frames_array_path": "data/3d_brightness_mode",
                    "timestamps_array_path": "timestamps/3d_brightness_mode",
                },
                "linked_mesh_sequences": [
                    {
                        "mesh_data": {"zarr_path": "data/3d_left_ventricle_mesh", "format": "zarr_group"},
                        "mesh_key": "LV",
                    }
                ],
            }
        },
        zarr_path="mesh.zarr",
        croissant_path=tmp_path / "croissant.json",
    )

    catalog = load_croissant(root=tmp_path)
    volume = catalog.by_recording_id("volume")
    mesh = catalog.by_recording_id("mesh")
    mesh_store = zarr.open_group(tmp_path / "mesh.zarr", mode="r")
    links = json.loads((tmp_path / "croissant.json").read_text(encoding="utf-8"))["recordSet"][1]["data"]

    assert volume is not None
    assert mesh is not None
    assert volume.stitch_beat_count == 4
    assert mesh.stitch_beat_count is None
    assert "data/3d_brightness_mode" not in mesh_store
    assert find_source_recordings(mesh, croissant=catalog) == (volume,)
    assert links == [
        {
            "recording_links/exam_id": "exam",
            "recording_links/source_recording_id": "mesh",
            "recording_links/linked_recording_id": "volume",
            "recording_links/panel_role_id": "3d_brightness_mode",
            "recording_links/linked_frame_array_id": "volume/data/3d_brightness_mode",
            "recording_links/linked_timestamp_array_id": "volume/timestamps/3d_brightness_mode",
        }
    ]
    assert linked_frame_timestamp_paths(volume, "3d_brightness_mode")[0] == "timestamps/3d_brightness_mode"


def test_prediction_recording_writer_publishes_recording_manifest_without_internal_annotation_plumbing(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "prediction.zarr"
    write_prediction_recording(
        output_path,
        exam_id="exam",
        recording_id="prediction",
        arrays=[
            PredictionArray(
                "3d_left_ventricle_mask",
                np.ones((2, 3, 4), dtype=np.float32),
            )
        ],
        attrs={
            "recording_manifest": {
                "manifest_type": "2d+3d+strain",
                "annotation_type": "left_ventricular_strain",
                "sectors": [
                    {"semantic_id": "bmode"},
                    {
                        "semantic_id": "bmode",
                        "frames": {"array_path": "data/3d_brightness_mode", "format": "zarr_array"},
                        "timestamps": {"array_path": "timestamps/3d_brightness_mode", "format": "zarr_array"},
                        "geometry": {
                            "coordinate_system": "spherical_sector_3d",
                            "DepthStart": 0.5,
                            "DepthEnd": 1.5,
                            "Width": 0.4,
                            "ElevationWidth": 0.3,
                        },
                    },
                ],
                "array_refs": [{"name": "data/4ch_contour", "role": "contour"}],
                "metadata": {
                    "time_reference": {"volume_time_origin_s": 0.0},
                    "qrs_trigger_times": [0.0, 1.0],
                    "geometry": {
                        "DepthStart": 0.5,
                        "DepthEnd": 1.5,
                        "Width": 0.4,
                        "ElevationWidth": 0.3,
                    },
                },
                "linked_panels": [
                    {
                        "role_id": "4ch",
                        "linked_recording": {"recording_id": "source"},
                    }
                ],
                "annotations": [
                    {
                        "kind": "contour",
                        "source_kind": "vendor_pipeline",
                        "target": {
                            "type": "linked_panel",
                            "semantic_id": "4ch",
                            "field": "contour_points",
                        },
                        "target_data": {"zarr_path": "data/2d_brightness_mode", "format": "zarr_array"},
                        "time": {"zarr_path": "timestamps/4ch_contour", "format": "zarr_array"},
                        "value": {"zarr_path": "data/4ch_contour", "format": "zarr_array"},
                    }
                ],
                "curve_groups": [{"name": "global", "annotation_ids": ["internal"]}],
                "contour_sequences": [{"name": "4ch", "annotation_ids": ["internal"]}],
            }
        },
    )

    attrs = dict(zarr.open_group(output_path, mode="r").attrs)
    document = attrs["recording_manifest"]

    assert "render_inputs" not in attrs
    assert "documents" not in document
    assert "arrays" not in document
    assert "render_metadata" not in document
    assert document["manifest_type"] == "2d+3d+strain"
    assert document["annotation_type"] == "left_ventricular_strain"
    assert document["array_refs"] == [{"name": "data/4ch_contour", "role": "contour"}]
    assert "geometry" not in document["metadata"]
    assert document["metadata"]["time_reference"] == {"volume_time_origin_s": 0.0}
    assert document["sectors"][1]["geometry"]["DepthStart"] == 0.5
    assert document["linked_panels"][0]["role_id"] == "4ch"
    assert "annotation_id" not in document["annotations"][0]
    assert "entity_type" not in document["annotations"][0]["target"]
    assert "target_role_id" not in document["annotations"][0]["target"]
    assert "source_kind" not in document["annotations"][0]
    assert "annotation_ids" not in document["curve_groups"][0]
    assert "annotation_ids" not in document["contour_sequences"][0]
    assert document["annotations"][0]["target"]["field"] == "contour_points"


def test_spherical_geometry_reads_public_3d_sector_geometry() -> None:
    geometry = spherical_geometry_from_metadata(
        {
            "sectors": [
                {
                    "semantic_id": "bmode",
                    "frames": {"array_path": "data/3d_brightness_mode", "format": "zarr_array"},
                    "geometry": {
                        "coordinate_system": "spherical_sector_3d",
                        "DepthStart": 0.5,
                        "DepthEnd": 1.5,
                        "Width": 0.4,
                        "ElevationWidth": 0.3,
                    },
                }
            ]
        }
    )

    assert geometry.depth_start_m == 0.5
    assert geometry.depth_end_m == 1.5
    assert geometry.azimuth_width_rad == 0.4
    assert geometry.elevation_width_rad == 0.3


def test_spherical_geometry_accepts_internal_restored_render_metadata() -> None:
    geometry = spherical_geometry_from_metadata(
        {
            "render_metadata": {
                "DepthStart": 0.5,
                "DepthEnd": 1.5,
                "Width": 0.4,
                "ElevationWidth": 0.3,
            }
        }
    )

    assert geometry.depth_start_m == 0.5
    assert geometry.depth_end_m == 1.5
    assert geometry.azimuth_width_rad == 0.4
    assert geometry.elevation_width_rad == 0.3


def _row(
    recording_id: str,
    *,
    content_types: list[str],
    array_paths: list[str],
    frame_counts: dict[str, int] | None = None,
    source_recording_id: str | None = None,
    stitch_beat_count: int | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "recordings/exam_id": "exam",
        "recordings/recording_id": recording_id,
        "recordings/zarr_path": f"{recording_id}.zarr",
        "recordings/modes": content_types,
        "recordings/content_types": content_types,
        "recordings/frame_counts_by_content_type": frame_counts or {},
        "recordings/median_delta_time_by_content_type": {},
        "recordings/array_paths": array_paths,
    }
    if source_recording_id is not None:
        row["recordings/source_recording_id"] = source_recording_id
    if stitch_beat_count is not None:
        row["recordings/stitch_beat_count"] = stitch_beat_count
    return row


def _write_croissant(
    root: Path,
    rows: list[dict[str, object]],
    *,
    recording_links: list[dict[str, object]] | None = None,
) -> None:
    record_sets: list[dict[str, object]] = [{"@id": "recordings", "name": "recordings", "data": rows}]
    if recording_links is not None:
        record_sets.append({"@id": "recording_links", "name": "recording_links", "data": recording_links})
    (root / "croissant.json").write_text(
        json.dumps({"recordSet": record_sets}),
        encoding="utf-8",
    )
