"""Synthetic corrected-source contracts; scanner accuracy is audited separately."""

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

from echoxflow import load_aha_strain, open_recording
from echoxflow._afi_recovery import weighted_windows


def _exact(value: Any) -> Any:
    if isinstance(value, float):
        return {"ieee754_hex": value.hex()}
    if isinstance(value, dict):
        return {key: _exact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_exact(item) for item in value]
    return value


def _digest(values):
    return hashlib.sha256(np.asarray(values, dtype="<f4").tobytes()).hexdigest()


def _corrected_store(path: Path):
    group = zarr.open_group(path, mode="w")
    group.attrs["recording_id"] = "synthetic-aha"
    states: list[dict[str, Any]] = []
    globals_: list[dict[str, Any]] = []
    regions: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    contours: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    for group_id, view in enumerate(("A2C", "A3C", "A4C"), 1):
        clock = np.arange(5, dtype=np.float64) / 10 + group_id
        base = np.column_stack([np.arange(13) / 10, np.sin(np.arange(13) / 12 * np.pi)]).astype(np.float32)
        scales = [0.98, 1, 0.82 + group_id * 0.02, 1.3 if group_id == 3 else 1.02 + group_id * 0.02, 1]
        points = np.stack([base * scale for scale in scales])[:, :, None].repeat(5, axis=2)
        mid = points[:, :, 2].astype(np.float64)
        density = np.linalg.norm(np.diff(mid[1], axis=0), axis=-1)
        eps = 100 * (np.linalg.norm(np.diff(mid, axis=1), axis=-1).T / density[:, None] - 1)
        centers = np.linspace(1, 11, 60)
        fine = weighted_windows(np.column_stack([centers - 1, centers + 1]), density) @ eps
        bounds = np.column_stack([np.arange(0, 12, 2), np.arange(2, 14, 2)])
        physical = [0, 2, 4, 5, 3, 1] if view == "A4C" else [1, 3, 5, 4, 2, 0]
        secondary = (weighted_windows(bounds, density) @ eps)[np.argsort(physical)]
        primary = secondary.copy()
        trimmed = weighted_windows(np.array([[6 / 9, 2], [10, 6 + 8 * 6 / 9]]), density) @ eps
        primary[[0, 1] if view == "A4C" else [1, 0]] = trimmed
        global_values = weighted_windows(np.array([[0, 12]]), density) @ eps
        stack = np.concatenate([global_values, fine, primary, secondary]).astype(np.float32)
        prefix = f"annotations/{view}"
        for key, data in (("values", stack), ("clock", clock)):
            group.create_array(f"{prefix}/{key}", data=data)
        annotations.append(
            {
                "target": {"entity_type": "curve_group", "field": "curve_stack"},
                "links": {"group_id": group_id},
                "value": {"zarr_path": f"{prefix}/values"},
                "time": {"zarr_path": f"{prefix}/clock"},
            }
        )
        descriptors = {}
        record = 0
        for role, count in (("global", 1), ("fine_profile", 60), ("primary_curve", 6), ("secondary_curve", 6)):
            for index in range(count):
                descriptors[(role, index)] = {
                    "source_offset": 1000 * group_id + record,
                    "sample_count": 5,
                    "native_reference_time": float(clock[1]),
                }
                bindings.append(
                    {
                        "source_view": view,
                        "source_role": role,
                        "source_order_index": index,
                        "curve_group_id": group_id,
                        "curve_stack_record_index": record,
                        "source_offset": 1000 * group_id + record,
                        "source_values_sha256": _digest(stack[record]),
                    }
                )
                record += 1
        globals_.append(
            {
                "source_view": view,
                "curve": descriptors[("global", 0)],
                "fine_profile_curves": [descriptors[("fine_profile", i)] for i in range(60)],
            }
        )
        for index in range(6):
            values = stack[61 + index, :4]
            low, high = np.argmin(values), np.argmax(values)
            peak = high if values[high] >= 0.75 * abs(values[low]) else low
            regions.append(
                {
                    "source_view": view,
                    "source_region_order_index": index,
                    "primary_curve": descriptors[("primary_curve", index)],
                    "secondary_curve": descriptors[("secondary_curve", index)],
                    "source_region_prefix": {"accepted_bool": (view, index) not in {("A2C", 0), ("A3C", 4)}},
                    "timed_summary": {
                        "records": [
                            {
                                "source_quantity_id": 3,
                                "value": float(values[peak]),
                                "native_marker_time": float(clock[peak]),
                            }
                        ]
                    },
                }
            )
        states.append(
            {
                "source_view": view,
                "native_clock": clock.tolist(),
                "avc_seconds": 0.2,
                "unresolved_pre_avc_u32": 0,
                "unresolved_pre_avc_bool": True,
                "native_cursor_time": float(clock[2]),
                "tracking_state": {
                    "native_tip_material_index": 6,
                    "tracked_contours": {
                        "frame_count": 5,
                        "longitudinal_point_count": 13,
                        "wall_point_count": 5,
                        "source_point_order": "longitudinal_then_wall",
                    },
                },
            }
        )
        canonical = points.reshape(5, 65, 2).copy()
        canonical[..., 0] *= -1
        group.create_array(f"{prefix}/points", data=canonical)
        group.create_array(f"{prefix}/frames", data=np.arange(5))
        annotations.append(
            {
                "target": {"entity_type": "contour_sequence", "field": "points"},
                "links": {"sequence_id": group_id},
                "value": {"zarr_path": f"{prefix}/points"},
                "time": {"zarr_path": f"{prefix}/frames", "role": "frame_index"},
            }
        )
        contours.append(
            {
                "source_view": view,
                "contour_sequence_id": group_id,
                "native_frame_count": 5,
                "longitudinal_point_count": 13,
                "export_coordinate_transform": "reflect_x",
                "source_points_sha256": _digest(points),
            }
        )
    state = {
        "source_serialization_version": 6,
        "source_payload_sha256": "a" * 64,
        "unresolved_header_u32_fields": [3, 4, 3, 19, 0, 0],
        "gui_states": states,
        "global_summaries": globals_,
        "source_regions": regions,
    }
    group.attrs["recording_manifest"] = {
        "annotations": annotations,
        "native_afi_state": {
            "schema": "native_afi_state_v1",
            "decode_status": "validated_layout",
            "numeric_encoding": "ieee754_hex",
            "state": _exact(state),
            "export_curve_bindings": bindings,
            "export_contour_bindings": contours,
        },
    }
    return group


def test_structured_order_native_clocks_masks_and_cap(tmp_path):
    path = tmp_path / "aha.zarr"
    group = _corrected_store(path)
    result = load_aha_strain(path)
    assert result.recording_id == "synthetic-aha"
    assert [segment.aha_segment_id for segment in result.segments] == list(range(1, 18))
    assert all(len(segment.contributors) == 1 for segment in result.segments[:16])
    assert result.segments[0].accepted is False
    assert result.segments[12].accepted is False
    assert result.segments[13].accepted is False
    trace = result.segments[0].contributors[0]
    np.testing.assert_array_equal(trace.values, group["annotations/A2C/values"][61])
    assert trace.units == "percent"
    assert trace.native_ed_time == 1.1
    assert trace.native_avc_time == trace.native_ed_time + 0.2
    assert trace.nearest_avc_frame_index == 3
    cap = result.segments[16]
    assert cap.accepted is None and cap.acceptance_status == "unvalidated"
    assert len(cap.contributors) == 3
    assert {item.source_view for item in cap.contributors} == {"A2C", "A3C", "A4C"}
    assert all(item.accepted is None for item in cap.contributors)
    assert len({item.timestamps[0] for item in cap.contributors}) == 3
    low = np.mean([item.values[:4].min() for item in cap.contributors])
    high = np.mean([item.values[:4].max() for item in cap.contributors])
    assert cap.signed_peak_percent == (high if high >= 0.75 * abs(low) else low)
    assert not np.isclose(cap.signed_peak_percent, np.mean([item.signed_peak_percent for item in cap.contributors]))
    assert result.provenance["JPEG_values_used"] is False
    assert result.provenance["original_payload_opened"] is False


@pytest.mark.parametrize(
    "mutation",
    ["missing", "duplicate", "fingerprint", "clock", "points", "float", "acceptance", "tracking_law", "source_offset"],
)
def test_packet_binding_tampering_fails(tmp_path, mutation):
    path = tmp_path / "aha.zarr"
    group = _corrected_store(path)
    manifest = group.attrs["recording_manifest"]
    packet = manifest["native_afi_state"]
    if mutation == "missing":
        del manifest["native_afi_state"]
    elif mutation == "duplicate":
        packet["export_curve_bindings"][1] = packet["export_curve_bindings"][0]
    elif mutation == "fingerprint":
        packet["export_curve_bindings"][0]["source_values_sha256"] = "0" * 64
    elif mutation == "clock":
        clock = group["annotations/A2C/clock"][:]
        clock[0] = np.nextafter(clock[0], np.inf)
        group["annotations/A2C/clock"][:] = clock
    elif mutation == "points":
        group["annotations/A2C/points"][0, 0, 0] += 0.01
    elif mutation == "tracking_law":
        group["annotations/A2C/points"][2, 7, 1] += 0.01
        points = group["annotations/A2C/points"][:].reshape(5, 13, 5, 2)
        points[..., 0] *= -1
        packet["export_contour_bindings"][0]["source_points_sha256"] = _digest(points)
    elif mutation == "source_offset":
        packet["export_curve_bindings"][0]["source_offset"] += 1
    elif mutation == "float":
        packet["state"]["gui_states"][0]["avc_seconds"] = 0.2
    elif mutation == "acceptance":
        packet["state"]["source_regions"][0]["source_region_prefix"]["accepted_bool"] = "false"
    group.attrs["recording_manifest"] = manifest
    with pytest.raises(ValueError):
        load_aha_strain(path)


def test_caller_store_stays_open_and_owned_zip_is_closed(tmp_path, monkeypatch):
    from zarr.storage import ZipStore

    path = tmp_path / "aha.zarr"
    source = _corrected_store(path)
    store = open_recording(path)
    load_aha_strain(store)
    assert store.load_array("annotations/A2C/values").shape == (73, 5)
    zip_path = tmp_path / "aha.zip"
    with ZipStore(zip_path, mode="w") as zipped:
        target = zarr.open_group(zipped, mode="w")
        target.attrs.update(dict(source.attrs))
        for view in ("A2C", "A3C", "A4C"):
            for name in ("values", "clock", "points", "frames"):
                key = f"annotations/{view}/{name}"
                target.create_array(key, data=source[key][:])
    owned = open_recording(zip_path)
    monkeypatch.setattr("echoxflow.aha_strain.open_recording", lambda *args, **kwargs: owned)
    loaded = load_aha_strain(zip_path)
    assert owned.group.store._is_open is False
    assert loaded.segments[16].contributors[0].values.shape == (5,)


def test_asymmetric_cursor_coordinate_drift_and_wrong_anchor(tmp_path):
    path = tmp_path / "aha.zarr"
    group = _corrected_store(path)
    original = load_aha_strain(path)
    document = group.attrs["recording_manifest"]
    packet = document["native_afi_state"]
    for state, binding in zip(packet["state"]["gui_states"], packet["export_contour_bindings"], strict=True):
        view = state["source_view"]
        clock = group[f"annotations/{view}/clock"][:]
        cursor = float(clock[1] + 0.08)
        state["native_cursor_time"] = _exact(cursor)
        points = group[f"annotations/{view}/points"][:].reshape(5, 13, 5, 2)
        points[..., 0] *= -1
        beta = np.interp(clock, [clock[1], cursor, clock[-1]], [0, 0.5, 1])
        beta[0] = 0.5 * (clock[0] - clock[1]) / (cursor - clock[1])
        drift = points[1, :, 2].astype(float) * 0.03
        points[:, :, 2] += ((beta - 0.5)[:, None, None] * drift[None]).astype(np.float32)
        binding["source_points_sha256"] = _digest(points)
        points[..., 0] *= -1
        group[f"annotations/{view}/points"][:] = points.reshape(5, 65, 2)
    group.attrs["recording_manifest"] = document
    recovered = load_aha_strain(path)
    for expected, actual in zip(original.segments, recovered.segments, strict=True):
        for old, new in zip(expected.contributors, actual.contributors, strict=True):
            np.testing.assert_allclose(new.values, old.values, atol=2e-4)
    state = packet["state"]["gui_states"][0]
    state["native_cursor_time"] = _exact(float.fromhex(state["native_cursor_time"]["ieee754_hex"]) + 0.04)
    group.attrs["recording_manifest"] = document
    with pytest.raises(ValueError, match="unsupported native tracking law"):
        load_aha_strain(path)


def _materialized_store(path: Path):
    group = _corrected_store(path)
    recovered = load_aha_strain(path)
    document = group.attrs["recording_manifest"]
    panels = {panel["source_view"]: panel for panel in recovered.provenance["panels"]}
    segments = []
    for segment in recovered.segments:
        contributors = []
        for trace in segment.contributors:
            semantic = f"aha{segment.aha_segment_id:02d}_{segment.name.replace(' ', '_')}_{trace.source_view.lower()}"
            value_path = f"annotations/anatomical/{semantic}"
            time_path = f"annotations/anatomical/clock_{trace.source_view}"
            group.create_array(value_path, data=trace.values)
            if time_path not in group:
                group.create_array(time_path, data=trace.timestamps)
            metadata = {
                "aha_segment_id": segment.aha_segment_id,
                "segment_name": segment.name,
                "curve_semantic_id": semantic,
                "source_view": trace.source_view,
                "reconstruction_method": trace.reconstruction_method,
                "representation": "anatomical_annotation",
                "source_payload_sha256": "a" * 64,
                "acceptance_status": segment.acceptance_status,
                "signed_peak_percent": trace.signed_peak_percent,
                "peak_method": "native_start_through_nearest_AVC_signed_0.75_rule",
                "native_ed_time": trace.native_ed_time,
                "avc_seconds": trace.avc_seconds,
                "native_avc_time": trace.native_avc_time,
                "nearest_avc_frame_index": trace.nearest_avc_frame_index,
                "native_closure_frame_index": panels[trace.source_view]["native_closure_frame_index"],
                "native_coordinate_correction_anchor_time": trace.provenance[
                    "native_coordinate_correction_anchor_time"
                ],
                "native_tracking_source_error_percent": panels[trace.source_view][
                    "max_native_tracking_source_error_percent"
                ],
            }
            if trace.accepted is not None:
                metadata["accepted"] = trace.accepted
            for key in ("source_region_order_index", "source_offset"):
                if key in trace.provenance:
                    metadata[key] = trace.provenance[key]
            links = _exact(metadata)
            value_ref = {"zarr_path": value_path, "format": "zarr_array"}
            time_ref = {"zarr_path": time_path, "format": "zarr_array"}
            document["annotations"].append(
                {
                    "kind": "aha_segmental_strain",
                    "target": {"type": "derived_curve", "semantic_id": semantic, "field": "values"},
                    "links": links,
                    "value": value_ref,
                    "time": time_ref,
                }
            )
            contributors.append({**links, "value": value_ref, "time": time_ref})
        entry = {
            "aha_segment_id": segment.aha_segment_id,
            "segment_name": segment.name,
            "units": "percent",
            "contributors": contributors,
            "acceptance_status": segment.acceptance_status,
            "signed_peak_percent": _exact(segment.signed_peak_percent),
            "peak_method": segment.peak_method,
        }
        if segment.accepted is not None:
            entry["accepted"] = segment.accepted
        segments.append(entry)
    document["anatomical_strain"] = {
        "schema": "anatomical_strain_v1",
        "segment_model": "aha17",
        "units": "percent",
        "recovery_status": "validated_native_tracking",
        "source_payload_sha256": "a" * 64,
        "representation": "sixteen_native_clock_trajectories_and_three_central_contributors",
        "central_acceptance_status": "unvalidated",
        "central_combined_trajectory_status": "unvalidated",
        "clock_policy": "independent_native_source_view_clocks",
        "segments": segments,
    }
    group.attrs["recording_manifest"] = document
    return group, recovered


def test_materialized_named_curves_are_bound_to_exported_paths(tmp_path):
    group, expected = _materialized_store(tmp_path / "materialized.zarr")
    actual = load_aha_strain(tmp_path / "materialized.zarr")
    assert actual.provenance["materialized_export_validated"] is True
    for old_segment, new_segment in zip(expected.segments, actual.segments, strict=True):
        assert old_segment.name == new_segment.name
        for old, new in zip(old_segment.contributors, new_segment.contributors, strict=True):
            np.testing.assert_array_equal(new.values, old.values)
            np.testing.assert_array_equal(new.timestamps, old.timestamps)
            assert new.value_path and new.timestamps_path
            assert group[new.value_path][:].tobytes() == new.values.tobytes()
            assert new.provenance["value_path"] == new.value_path
    assert len({trace.value_path for segment in actual.segments for trace in segment.contributors}) == 19
    assert len({trace.timestamps_path for segment in actual.segments for trace in segment.contributors}) == 3
    assert expected.segments[0].contributors[0].value_path is None
    assert expected.provenance["materialized_export_validated"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "order",
        "name",
        "units",
        "mask",
        "accepted_int",
        "CAP_mask",
        "status",
        "source_hash",
        "missing_annotation",
        "duplicate_annotation",
        "target",
        "annotation_mask",
        "annotation_ref",
        "value_ulp",
        "value_signed_zero",
        "clock_ulp",
        "missing_array",
        "peak",
        "AVC",
        "unencoded_float",
        "contributors",
        "CAP_order",
        "ref_units",
        "dtype",
    ],
)
def test_materialized_annotations_fail_closed_on_tampering(tmp_path, mutation):
    path = tmp_path / "materialized.zarr"
    group, _ = _materialized_store(path)
    document = group.attrs["recording_manifest"]
    materialized = document["anatomical_strain"]
    segment = materialized["segments"][0]
    contributor = segment["contributors"][0]
    annotation = next(item for item in document["annotations"] if item.get("kind") == "aha_segmental_strain")
    if mutation == "order":
        materialized["segments"][0], materialized["segments"][1] = (
            materialized["segments"][1],
            materialized["segments"][0],
        )
    elif mutation == "name":
        segment["segment_name"] = "basal inferior"
    elif mutation == "units":
        segment["units"] = "fraction"
    elif mutation == "mask":
        segment["accepted"] = not segment["accepted"]
    elif mutation == "accepted_int":
        segment["accepted"] = int(segment["accepted"])
    elif mutation == "CAP_mask":
        materialized["segments"][16]["accepted"] = True
    elif mutation == "status":
        materialized["recovery_status"] = "unvalidated"
    elif mutation == "source_hash":
        materialized["source_payload_sha256"] = "b" * 64
    elif mutation == "missing_annotation":
        document["annotations"].remove(annotation)
    elif mutation == "duplicate_annotation":
        document["annotations"].append(annotation)
    elif mutation == "target":
        annotation["target"]["semantic_id"] = "wrong_segment"
    elif mutation == "annotation_mask":
        annotation["links"]["accepted"] = not annotation["links"]["accepted"]
    elif mutation == "annotation_ref":
        annotation["value"] = materialized["segments"][1]["contributors"][0]["value"]
    elif mutation in {"value_ulp", "value_signed_zero", "clock_ulp"}:
        key = contributor["time" if mutation == "clock_ulp" else "value"]["zarr_path"]
        values = group[key][:]
        index = 1 if mutation == "value_signed_zero" else 2
        values[index] = (
            -0.0 if mutation == "value_signed_zero" else np.nextafter(values[index], np.inf, dtype=values.dtype)
        )
        group[key][:] = values
    elif mutation == "missing_array":
        del group[contributor["value"]["zarr_path"]]
    elif mutation == "dtype":
        key = contributor["value"]["zarr_path"]
        values = group[key][:].astype(np.float64)
        del group[key]
        group.create_array(key, data=values)
    elif mutation == "peak":
        segment["signed_peak_percent"] = _exact(10.0)
    elif mutation == "AVC":
        contributor["avc_seconds"] = _exact(0.3)
    elif mutation == "unencoded_float":
        contributor["avc_seconds"] = 0.2
    elif mutation == "contributors":
        segment["contributors"] = []
    elif mutation == "CAP_order":
        materialized["segments"][16]["contributors"].reverse()
    elif mutation == "ref_units":
        contributor["value"]["units"] = "fraction"
        annotation["value"]["units"] = "fraction"
    group.attrs["recording_manifest"] = document
    with pytest.raises(ValueError, match="[Mm]aterialized|exact|native float|missing"):
        load_aha_strain(path)


@pytest.mark.parametrize("mutation", [None, "value_units", "clock_units", "dtype", "shape", "missing", "duplicate"])
def test_materialized_root_catalog_calibration_is_validated(tmp_path, mutation):
    path = tmp_path / "catalog.zarr"
    group, _ = _materialized_store(path)
    document = group.attrs["recording_manifest"]
    refs = {
        item[field]["zarr_path"]: field
        for segment in document["anatomical_strain"]["segments"]
        for item in segment["contributors"]
        for field in ("value", "time")
    }
    catalog = [
        {
            "name": key,
            "units": "percent" if field == "value" else "seconds",
            "dtype": str(group[key].dtype),
            "shape": list(group[key].shape),
        }
        for key, field in refs.items()
    ]
    value_entry = next(item for item in catalog if item["units"] == "percent")
    clock_entry = next(item for item in catalog if item["units"] == "seconds")
    if mutation == "value_units":
        value_entry["units"] = "fraction"
    elif mutation == "clock_units":
        clock_entry["units"] = "milliseconds"
    elif mutation == "dtype":
        value_entry["dtype"] = "float64"
    elif mutation == "shape":
        value_entry["shape"] = [6]
    elif mutation == "missing":
        catalog.remove(value_entry)
    elif mutation == "duplicate":
        catalog.append(value_entry)
    group.attrs["arrays"] = catalog
    if mutation is None:
        assert load_aha_strain(path).provenance["materialized_export_validated"] is True
    else:
        with pytest.raises(ValueError, match="[Mm]aterialized.*catalog"):
            load_aha_strain(path)
