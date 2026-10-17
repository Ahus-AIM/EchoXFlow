from pathlib import Path

import numpy as np
import pytest
import zarr

from echoxflow import load_segmental_strain, open_recording


def _store(path: Path) -> object:
    group = zarr.open_group(path, mode="w")
    group.attrs["recording_id"] = "strain-example"
    group.attrs["arrays"] = [{"name": "annotations/curve_stack", "units": "arbitrary"}]
    values = np.arange(24, dtype=np.float32).reshape(8, 3)
    group.create_array("annotations/curve_stack", data=values)
    # A reset is intentional: the reader must retain the native clock.
    group.create_array("timestamps/curve", data=np.array([9.1, 9.2, 2.0]))
    group.create_array("timestamps/qrs", data=np.array([8.0, 9.15]))
    group.attrs["recording_manifest"] = {
        "linked_panels": [{"semantic_id": "seq01", "linked_recording": {"recording_id": "source"}}],
        "annotations": [
            {
                "kind": "aps_segmental_curves",
                "target": {"semantic_id": "seq01", "field": "curve_stack"},
                "value": {"zarr_path": "annotations/curve_stack"},
                "time": {"zarr_path": "timestamps/curve"},
                "links": {
                    "curve_record_axis": 0,
                    "curve_record_indices": [3, 1],
                    "curve_record_order": "source_order",
                    "linked_panel_semantic_id": "seq01",
                    "anatomical_segment_labels_status": "unresolved",
                    "alignment_status": "resolved",
                },
            },
            {
                "kind": "aps_curve",
                "value": {"zarr_path": "annotations/curve_stack"},
                "links": {
                    "record_roles": [
                        "aux_header",
                        "paired_aux",
                        "aux_header",
                        "paired_aux",
                        "body",
                        "body",
                        "body",
                        "other_aux",
                    ]
                },
            },
            {
                "kind": "event_markers",
                "target": {"semantic_id": "seq01", "field": "ecg_qrs"},
                "time": {"zarr_path": "timestamps/qrs"},
            },
        ],
    }
    return group


def test_declared_order_native_clock_and_alternative_sources(tmp_path: Path) -> None:
    _store(tmp_path / "strain.zarr")
    result = load_segmental_strain(tmp_path / "strain.zarr")
    panel = result.panels[0]

    assert result.recording_id == "strain-example"
    assert panel.record_indices == (3, 1)
    np.testing.assert_array_equal(panel.values, panel.source_values[[3, 1]])
    np.testing.assert_array_equal(panel.timestamps, [9.1, 9.2, 2.0])
    np.testing.assert_array_equal(panel.qrs_times, [8.0, 9.15])
    assert panel.header_record_indices == (0, 2)
    np.testing.assert_array_equal(panel.header_values, panel.source_values[[0, 2]])
    assert panel.units == "arbitrary"
    assert panel.anatomical_segment_labels_status == "unresolved"
    assert panel.linked_panel["linked_recording"]["recording_id"] == "source"
    assert panel.provenance["links"]["curve_record_indices"] == [3, 1]
    assert panel.qrs_timestamps_path == "timestamps/qrs"
    assert panel.source_metadata is None


@pytest.mark.parametrize("indices", [[-1], [8], [1, 1], [True], [1.0], []])
def test_invalid_record_selection_fails(tmp_path: Path, indices: list[object]) -> None:
    group = _store(tmp_path / "strain.zarr")
    manifest = group.attrs["recording_manifest"]
    manifest["annotations"][0]["links"]["curve_record_indices"] = indices
    group.attrs["recording_manifest"] = manifest
    with pytest.raises(ValueError, match="curve_record_indices"):
        load_segmental_strain(tmp_path / "strain.zarr")


def test_missing_selection_does_not_guess_from_source_shape(tmp_path: Path) -> None:
    group = _store(tmp_path / "strain.zarr")
    manifest = group.attrs["recording_manifest"]
    manifest["annotations"].pop(0)
    group.attrs["recording_manifest"] = manifest
    assert load_segmental_strain(tmp_path / "strain.zarr").panels == ()


def test_mismatched_native_clock_fails(tmp_path: Path) -> None:
    group = _store(tmp_path / "strain.zarr")
    group.create_array("timestamps/curve", data=np.array([1.0, 2.0]), overwrite=True)
    with pytest.raises(ValueError, match="native clock"):
        load_segmental_strain(tmp_path / "strain.zarr")


def test_missing_roles_and_qrs_remain_missing_and_existing_store_is_reusable(tmp_path: Path) -> None:
    group = _store(tmp_path / "strain.zarr")
    manifest = group.attrs["recording_manifest"]
    manifest["annotations"] = manifest["annotations"][:1]
    group.attrs["recording_manifest"] = manifest
    store = open_recording(tmp_path / "strain.zarr")
    panel = load_segmental_strain(store).panels[0]
    assert panel.record_roles == ()
    assert panel.header_values.shape == (0, 3)
    assert panel.qrs_times is None
    np.testing.assert_array_equal(store.load_array(panel.value_path), panel.source_values)


def test_zip_path_closes_owned_store_after_loading(tmp_path: Path) -> None:
    from zarr.storage import ZipStore

    source = _store(tmp_path / "strain.zarr")
    zip_path = tmp_path / "strain.zip"
    with ZipStore(zip_path, mode="w") as zipped:
        target = zarr.open_group(zipped, mode="w")
        target.attrs.update(dict(source.attrs))
        for path in ("annotations/curve_stack", "timestamps/curve", "timestamps/qrs"):
            target.create_array(path, data=source[path][:])
    panel = load_segmental_strain(zip_path).panels[0]
    np.testing.assert_array_equal(panel.values, source["annotations/curve_stack"][[3, 1], :])
    # Loaded arrays remain usable after the reader closes its own ZipStore.
    assert panel.values.shape == (2, 3)


def _afi_store(path: Path) -> object:
    group = _store(path)
    source = np.ones((8, 24), dtype=np.float32)
    source[:7, [2, 20]] = 0
    words = np.zeros(24, dtype="<u4")
    words[:8] = [0, 0, 1072693248, 1, 1, 0, 5, 0]
    points = np.arange(10, dtype=np.float32).reshape(5, 2) / 10
    for index in range(4):
        words[8 + 4 * index : 10 + 4 * index] = [2, 1]
        words[10 + 4 * index : 12 + 4 * index] = points[index].view("<u4")
    source[7] = words.view("<f4")
    canonical = points.copy()
    canonical[:, 0] *= -1
    group.create_array("annotations/curve_stack", data=source, overwrite=True)
    group.create_array("annotations/contour", data=canonical[None])
    # Keep a native clock reset even when source metadata is decoded.
    clock = np.arange(24, dtype=np.float64)
    clock[12:] -= 20
    group.create_array("timestamps/curve", data=clock, overwrite=True)
    manifest = group.attrs["recording_manifest"]
    manifest["annotations"][0]["links"]["view_code"] = "A2C"
    manifest["annotations"][1]["links"]["record_roles"][0] = "opener"
    manifest["annotations"].append(
        {
            "kind": "aps_contour",
            "target": {"semantic_id": "seq01"},
            "value": {"zarr_path": "annotations/contour"},
            "links": {"source_to_canonical_transform": "reflect_x"},
        }
    )
    group.attrs["recording_manifest"] = manifest
    return group


def test_guarded_source_view_global_and_reference_candidates(tmp_path: Path) -> None:
    group = _afi_store(tmp_path / "strain.zarr")
    panel = load_segmental_strain(tmp_path / "strain.zarr").panels[0]
    metadata = panel.source_metadata
    assert metadata is not None
    assert metadata.source_view == "A3C"
    assert metadata.source_view_id == 1
    assert metadata.declared_view == "A2C"
    assert metadata.view_conflict
    assert metadata.residue_record_indices == (7,)
    assert metadata.biological_record_indices == tuple(range(7))
    assert metadata.global_record_index == 0
    np.testing.assert_array_equal(metadata.global_values, panel.source_values[0])
    assert metadata.common_zero_frame_indices == (2, 20)
    assert metadata.first_common_zero_frame_index == 2
    assert metadata.last_common_zero_frame_index == 20
    assert metadata.reference_status == "common_zero_candidates"
    np.testing.assert_array_equal(panel.timestamps, group["timestamps/curve"][:])
    assert np.diff(panel.timestamps).min() < 0
    np.testing.assert_array_equal(panel.source_values, group["annotations/curve_stack"][:])
    np.testing.assert_array_equal(panel.values, panel.source_values[[3, 1]])


@pytest.mark.parametrize("failure", ["count", "enum", "coordinate", "transform", "duplicate"])
def test_source_metadata_fails_closed_without_matching_contour(tmp_path: Path, failure: str) -> None:
    group = _afi_store(tmp_path / "strain.zarr")
    source = group["annotations/curve_stack"][:]
    if failure in ("count", "enum"):
        words = source[7].view("<u4")
        words[4 if failure == "count" else 3] = 2 if failure == "count" else 4
    elif failure == "duplicate":
        source[6] = source[7]
    elif failure == "coordinate":
        contour = group["annotations/contour"][:]
        contour[0, 1, 0] = np.nextafter(contour[0, 1, 0], np.float32(0))
        group["annotations/contour"][:] = contour
    else:
        manifest = group.attrs["recording_manifest"]
        manifest["annotations"][-1]["links"]["source_to_canonical_transform"] = "unknown"
        group.attrs["recording_manifest"] = manifest
    group["annotations/curve_stack"][:] = source
    panel = load_segmental_strain(tmp_path / "strain.zarr").panels[0]
    assert panel.source_metadata is None
    np.testing.assert_array_equal(panel.source_values, source)


def test_verified_source_without_common_zero_does_not_invent_ed(tmp_path: Path) -> None:
    group = _afi_store(tmp_path / "strain.zarr")
    source = group["annotations/curve_stack"][:]
    source[:7] = 1
    group["annotations/curve_stack"][:] = source
    metadata = load_segmental_strain(tmp_path / "strain.zarr").panels[0].source_metadata
    assert metadata is not None
    assert metadata.common_zero_frame_indices == ()
    assert metadata.first_common_zero_frame_index is None
    assert metadata.last_common_zero_frame_index is None
    assert metadata.reference_status == "unresolved"


@pytest.mark.parametrize("declared", [None, "APLAX"])
def test_missing_or_equivalent_declared_view_has_no_conflict(tmp_path: Path, declared: str | None) -> None:
    group = _afi_store(tmp_path / "strain.zarr")
    manifest = group.attrs["recording_manifest"]
    manifest["annotations"][0]["links"]["view_code"] = declared
    group.attrs["recording_manifest"] = manifest
    metadata = load_segmental_strain(tmp_path / "strain.zarr").panels[0].source_metadata
    assert metadata is not None
    assert metadata.declared_view == ("A3C" if declared is not None else None)
    assert metadata.provenance["declared_view_raw"] == declared
    assert not metadata.view_conflict
