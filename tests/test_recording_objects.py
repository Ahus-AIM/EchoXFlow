from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from echoxflow import RecordingRecord, load_recording_object, open_recording
from echoxflow.objects import ArrayRef, RecordingObject

zarr = pytest.importorskip("zarr")


def test_strain_object_parses_self_referenced_rv_panel(tmp_path: Path) -> None:
    group = zarr.open_group(tmp_path / "rv.zarr", mode="w")
    group.create_array("data/2d_brightness_mode", data=np.zeros((2, 4, 4), dtype=np.uint8))
    group.create_array("timestamps/2d_brightness_mode", data=np.asarray([0.0, 0.1], dtype=np.float32))
    group.create_array("data/rv_contour", data=np.zeros((1, 5, 2), dtype=np.float32))
    group.create_array("timestamps/rv_contour", data=np.asarray([0.0], dtype=np.float32))
    group.attrs["recording_manifest"] = _strain_document(("rv",), self_recording_id="rv")
    record = _record(
        "rv",
        "rv.zarr",
        array_paths=(
            "data/2d_brightness_mode",
            "timestamps/2d_brightness_mode",
            "data/rv_contour",
            "timestamps/rv_contour",
        ),
    )
    store = open_recording(record, root=tmp_path)

    obj = store.load_object()
    panel = obj.panels[0]

    assert isinstance(obj, RecordingObject)
    assert obj.kind == "strain"
    assert len(obj.panels) == 1
    assert panel.role_id == "rv"
    assert panel.bmode.recording.is_self is True
    assert panel.bmode.data_path == "data/2d_brightness_mode"
    assert panel.annotations[0].field == "contour_points"
    assert store.open_reference(panel.bmode.recording) is store
    assert store.load_array_ref(panel.annotations[0].value).shape == (1, 5, 2)


def test_strain_object_accepts_recording_manifest_attr(tmp_path: Path) -> None:
    group = zarr.open_group(tmp_path / "rv.zarr", mode="w")
    group.create_array("data/2d_brightness_mode", data=np.zeros((1, 2, 2), dtype=np.uint8))
    group.create_array("timestamps/2d_brightness_mode", data=np.asarray([0.0], dtype=np.float32))
    group.create_array("data/rv_contour", data=np.zeros((1, 3, 2), dtype=np.float32))
    group.create_array("data/rv_curve", data=np.asarray([0.0], dtype=np.float32))
    document = _strain_document(("rv",), self_recording_id="rv")
    panels = document["linked_panels"]
    assert isinstance(panels, list)
    linked_recording = panels[0]["linked_recording"]
    assert isinstance(linked_recording, dict)
    linked_recording.pop("timestamps_array_path")
    panels[0]["frame_timestamps"] = {"zarr_path": "timestamps/2d_brightness_mode", "format": "zarr_array"}
    annotations = document["annotations"]
    assert isinstance(annotations, list)
    annotations.append(
        {
            "kind": "curve",
            "target": {"type": "linked_curve", "semantic_id": "rv", "field": "strain_curve"},
            "target_data": {"zarr_path": "data/2d_brightness_mode", "format": "zarr_array"},
            "value": {"zarr_path": "data/rv_curve", "format": "zarr_array"},
        }
    )
    group.attrs["recording_manifest"] = document
    record = _record(
        "rv",
        "rv.zarr",
        array_paths=("data/2d_brightness_mode", "timestamps/2d_brightness_mode", "data/rv_contour", "data/rv_curve"),
    )

    obj = open_recording(record, root=tmp_path).load_object()

    assert obj.kind == "strain"
    assert obj.panels[0].role_id == "rv"
    assert obj.panels[0].bmode.timestamps_path == "timestamps/2d_brightness_mode"
    assert sorted((annotation.field, annotation.target_semantic_id) for annotation in obj.panels[0].annotations) == [
        ("contour_points", "rv"),
        ("strain_curve", "rv"),
    ]


def test_strain_object_resolves_external_lv_panels(tmp_path: Path) -> None:
    for index, role in enumerate(("2ch", "3ch", "4ch"), start=1):
        source = zarr.open_group(tmp_path / f"{role}.zarr", mode="w")
        source.create_array("data/2d_brightness_mode", data=np.full((1, 2, 2), index, dtype=np.uint8))
        source.create_array("timestamps/2d_brightness_mode", data=np.asarray([0.0], dtype=np.float32))
    strain = zarr.open_group(tmp_path / "strain.zarr", mode="w")
    for role in ("2ch", "3ch", "4ch"):
        strain.create_array(f"data/{role}_contour", data=np.zeros((1, 3, 2), dtype=np.float32))
        strain.create_array(f"timestamps/{role}_contour", data=np.asarray([0.0], dtype=np.float32))
    strain.attrs["recording_manifest"] = _strain_document(("2ch", "3ch", "4ch"))
    record = _record(
        "strain",
        "strain.zarr",
        content_types=("2d_left_ventricular_strain",),
        array_paths=tuple(
            item for role in ("2ch", "3ch", "4ch") for item in (f"data/{role}_contour", f"timestamps/{role}_contour")
        ),
    )
    store = open_recording(record, root=tmp_path)

    obj = store.load_object()

    assert obj.kind == "strain"
    assert [panel.role_id for panel in obj.panels] == ["2ch", "3ch", "4ch"]
    assert all(not panel.bmode.recording.is_self for panel in obj.panels)
    assert store.load_array_ref(obj.panels[1].annotations[0].value).shape == (1, 3, 2)
    assert store.load_array_ref(ArrayRef(obj.panels[1].bmode.data_path), obj.panels[1].bmode.recording).shape == (
        1,
        2,
        2,
    )


def test_strain_object_accepts_one_and_two_panel_metadata(tmp_path: Path) -> None:
    for roles in (("4ch",), ("2ch", "4ch")):
        group = zarr.open_group(tmp_path / f"{'_'.join(roles)}.zarr", mode="w")
        group.attrs["recording_manifest"] = _strain_document(roles, self_recording_id="strain")
        record = _record("strain", f"{'_'.join(roles)}.zarr", content_types=("2d_left_ventricular_strain",))

        obj = open_recording(record, root=tmp_path).load_object()

        assert obj.kind == "strain"
        assert [panel.role_id for panel in obj.panels] == list(roles)


def test_3d_mesh_sequence_is_recording_object_reference(tmp_path: Path) -> None:
    group = zarr.open_group(tmp_path / "three_d.zarr", mode="w")
    group.create_array("data/3d_brightness_mode", data=np.zeros((1, 2, 2, 2), dtype=np.uint8))
    mesh_group = group.create_group("data/3d_left_ventricle_mesh")
    mesh_group.create_array("point_values", data=np.zeros((3, 3), dtype=np.float32))
    group.create_array("timestamps/3d_left_ventricle_mesh", data=np.asarray([0.0], dtype=np.float32))
    group.attrs["recording_manifest"] = {
        "manifest_type": "3d",
        "sectors": [
            {
                "semantic_id": "bmode",
                "frames": {"array_path": "data/3d_brightness_mode", "format": "zarr_array"},
                "timestamps": {"array_path": "timestamps/3d_brightness_mode", "format": "zarr_array"},
                "geometry": {
                    "coordinate_system": "spherical_sector_3d",
                    "DepthStart": 0.0,
                    "DepthEnd": 0.1,
                    "Width": 0.5,
                },
            }
        ],
        "annotations": [
            {
                "kind": "mesh",
                "target": {
                    "type": "linked_mesh_sequence",
                    "semantic_id": "3d_left_ventricle_mesh",
                    "field": "mesh_data",
                },
                "target_data": {"zarr_path": "data/3d_brightness_mode", "format": "zarr_array"},
                "value": {"zarr_path": "data/3d_left_ventricle_mesh", "format": "zarr_group"},
            }
        ],
        "linked_mesh_sequences": [
            {
                "mesh_data": {"zarr_path": "data/3d_left_ventricle_mesh", "format": "zarr_group"},
                "timestamps": {"zarr_path": "timestamps/3d_left_ventricle_mesh", "format": "zarr_array"},
                "mesh_key": "LV",
            }
        ],
    }
    record = _record(
        "three_d",
        "three_d.zarr",
        content_types=("3d_brightness_mode",),
        array_paths=(
            "data/3d_brightness_mode",
            "data/3d_left_ventricle_mesh/point_values",
            "timestamps/3d_left_ventricle_mesh",
        ),
    )

    obj = load_recording_object(record, root=tmp_path)

    assert obj.kind == "3d_bmode"
    assert len(obj.mesh_sequences) == 1
    assert obj.mesh_sequences[0].role_id == "3d_left_ventricle_mesh"
    assert obj.mesh_sequences[0].label == "LV"
    assert obj.mesh_sequences[0].mesh_group.path == "data/3d_left_ventricle_mesh"


def test_bmode_stream_does_not_expose_reverse_annotation_ownership(tmp_path: Path) -> None:
    group = zarr.open_group(tmp_path / "rv.zarr", mode="w")
    group.create_array("data/2d_brightness_mode", data=np.zeros((1, 2, 2), dtype=np.uint8))
    group.create_array("timestamps/2d_brightness_mode", data=np.asarray([0.0], dtype=np.float32))
    group.attrs["recording_manifest"] = _strain_document(("rv",), self_recording_id="rv")
    record = _record(
        "rv",
        "rv.zarr",
        array_paths=("data/2d_brightness_mode", "timestamps/2d_brightness_mode"),
    )

    stream = open_recording(record, root=tmp_path).load_stream("2d_brightness_mode")

    assert not hasattr(stream, "annotations")
    assert not hasattr(stream, "mesh_sequences")


def _strain_document(roles: tuple[str, ...], *, self_recording_id: str | None = None) -> dict[str, object]:
    return {
        "manifest_type": "strain",
        "schema_version": 4,
        "annotation_type": "right_ventricular_strain" if roles == ("rv",) else "left_ventricular_strain",
        "linked_panels": [
            {
                "role_id": role,
                "view_code": role.upper(),
                "geometry": {
                    "depth_start_m": 0.0,
                    "depth_end_m": 0.1,
                    "tilt_rad": 0.0,
                    "width_rad": 0.5,
                },
                "frame_timestamps": {"zarr_path": "timestamps/2d_brightness_mode", "format": "zarr_array"},
                "linked_recording": {
                    "recording_id": self_recording_id or f"{role}_source",
                    "recording_zarr_path": f"{role}.zarr" if self_recording_id is None else "rv.zarr",
                    "relative_zarr_path": f"{role}.zarr" if self_recording_id is None else "rv.zarr",
                    "frames_array_path": "data/2d_brightness_mode",
                    "timestamps_array_path": "timestamps/2d_brightness_mode",
                },
            }
            for role in roles
        ],
        "annotations": [
            {
                "kind": "contour",
                "target": {
                    "type": "linked_panel",
                    "semantic_id": role,
                    "field": "contour_points",
                },
                "target_data": {"zarr_path": "data/2d_brightness_mode", "format": "zarr_array"},
                "time": {"zarr_path": f"timestamps/{role}_contour", "format": "zarr_array"},
                "value": {"zarr_path": f"data/{role}_contour", "format": "zarr_array"},
            }
            for role in roles
        ],
    }


def _record(
    recording_id: str,
    zarr_path: str,
    *,
    content_types: tuple[str, ...] = ("2d_right_ventricular_strain",),
    array_paths: tuple[str, ...] = (),
) -> RecordingRecord:
    return RecordingRecord(
        exam_id="exam",
        recording_id=recording_id,
        zarr_path=zarr_path,
        modes=("2d", "aps"),
        content_types=content_types,
        frame_counts_by_content_type={content_type: 1 for content_type in content_types},
        median_delta_time_by_content_type={},
        array_paths=array_paths,
    )
