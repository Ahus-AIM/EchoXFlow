"""Anatomical strain trajectories from a corrected native automated function imaging (AFI) export packet."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from echoxflow._afi_packet import bind_native_packet, decode_exact_packet
from echoxflow._afi_recovery import AHA_SEGMENT_NAMES, AFIRecoveryError, recover_native_arrays, signed_peak
from echoxflow.croissant import RecordingRecord
from echoxflow.loading import RecordingStore, open_recording
from echoxflow.manifest import manifest_documents


@dataclass(frozen=True)
class AHAStrainContributor:
    """One native-clock trajectory, expressed in percent.

    AVC is retained both as the native interval from ED and as absolute native
    time. Peak selection starts at the first stored sample (including pre-ED)
    and ends at the nearest native AVC sample. CAP contributors have no validated
    individual acceptance flag; their numeric trajectories remain available.
    """

    values: np.ndarray
    timestamps: np.ndarray
    source_view: str
    accepted: bool | None
    units: str
    native_ed_time: float
    avc_seconds: float
    native_avc_time: float
    nearest_avc_frame_index: int
    signed_peak_percent: float
    reconstruction_method: str
    provenance: Mapping[str, Any]
    value_path: str | None = None
    timestamps_path: str | None = None


@dataclass(frozen=True)
class AHAStrainSegment:
    """An AHA segment and its explicitly represented native contributors.

    Segments 1–16 each contain one trajectory. Segment 17 contains three native
    trajectories with independent clocks, rather than an invented combined
    trajectory. Its scalar averages the three negative extrema and three positive
    extrema separately, then chooses positive when >= .75 * abs(negative).
    Its acceptance is unknown even when some regional source flags are rejected.
    """

    aha_segment_id: int
    name: str
    contributors: tuple[AHAStrainContributor, ...]
    accepted: bool | None
    acceptance_status: str
    signed_peak_percent: float
    peak_method: str


@dataclass(frozen=True)
class AHAStrain:
    """Seventeen AHA-ordered segment groups and the reconstruction provenance."""

    recording_id: str | None
    store_path: Path
    segments: tuple[AHAStrainSegment, ...]
    provenance: Mapping[str, Any]


def load_aha_strain(
    record: RecordingStore | RecordingRecord | str | Path,
    *,
    root: str | Path | None = None,
) -> AHAStrain:
    """Recover source-validated AHA trajectories from a corrected recording.

    Requires the validated ``native_afi_state_v1`` packet with exact source
    clocks, calibration and array fingerprints. Older exports lacking this packet
    raise ValueError. Unsupported tracking configurations or changed source arrays
    also fail closed. No scanner JPEG or original DICOM is read.

    When named anatomical annotations are materialized in the export, their
    labels, acceptance metadata, values and native clocks must agree exactly
    with reconstruction. Contributors then expose their exported array paths.

    Caller-owned stores remain open. Arrays returned here remain usable after
    stores opened by this function are closed. CAP quality remains unvalidated;
    this API does not claim a scanner-defined single CAP trace or acceptance flag.
    """
    store = record if isinstance(record, RecordingStore) else open_recording(record, root=root)
    try:
        documents: list[dict[str, Any]] = []
        anatomical_documents: list[dict[str, Any]] = []
        for metadata in manifest_documents(store.group.attrs):
            items = metadata.get("documents", [metadata])
            if isinstance(items, list):
                documents.extend(item for item in items if isinstance(item, dict) and "native_afi_state" in item)
                anatomical_documents.extend(
                    item for item in items if isinstance(item, dict) and "anatomical_strain" in item
                )
        if len(documents) != 1:
            raise AFIRecoveryError("Recording needs exactly one corrected native AFI document")
        document = documents[0]
        if anatomical_documents and (len(anatomical_documents) != 1 or anatomical_documents[0] is not document):
            raise AFIRecoveryError("Materialized anatomical strain must belong to the unique native AFI document")

        def read_array(ref: Any) -> np.ndarray:
            if not isinstance(ref, dict) or not isinstance(ref.get("zarr_path"), str) or not ref["zarr_path"]:
                raise AFIRecoveryError("Corrected recording array needs an explicit zarr_path")
            array_path = ref["zarr_path"]
            if array_path not in store.group:
                raise AFIRecoveryError(f"Corrected native source array is missing: {array_path}")
            values = store.load_array(array_path)
            if "shape" in ref and list(values.shape) != ref["shape"]:
                raise AFIRecoveryError("Corrected native array shape differs from its reference")
            if "dtype" in ref and values.dtype != np.dtype(ref["dtype"]):
                raise AFIRecoveryError("Corrected native array dtype differs from its reference")
            if not np.isfinite(values).all():
                raise AFIRecoveryError("Corrected native array contains nonfinite values")
            return values

        try:
            decoded, inputs = bind_native_packet(document, read_array)
            manifest, arrays = recover_native_arrays(decoded, inputs)
            segments = _segments(manifest, arrays, decoded)
            if "anatomical_strain" in document:
                catalog = store.group.attrs.get("arrays")
                if "arrays" in store.group.attrs and (
                    not isinstance(catalog, list) or any(not isinstance(item, dict) for item in catalog)
                ):
                    raise AFIRecoveryError("Materialized anatomical recording has a malformed array catalog")
                segments = _materialized_segments(document, segments, manifest, decoded, read_array, catalog)
        except (KeyError, TypeError, ValueError, AttributeError, IndexError, OverflowError) as error:
            if isinstance(error, AFIRecoveryError):
                raise
            raise AFIRecoveryError(f"Malformed corrected native AFI recording: {error}") from error
        recording_id = store.record.recording_id if store.record is not None else store.group.attrs.get("recording_id")
        provenance = dict(manifest)
        provenance.update(
            input_contract="corrected_EchoXFlow_recording_and_source_arrays_only",
            original_payload_opened=False,
            materialized_export_validated="anatomical_strain" in document,
            input_manifest_sha256=hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest(),
        )
        return AHAStrain(recording_id, store.path, segments, provenance)
    finally:
        if not isinstance(record, RecordingStore):
            store.close()


def _expect_fields(actual: dict[str, Any], expected: dict[str, Any], context: str) -> None:
    for key, value in expected.items():
        observed = actual.get(key)
        equal = type(observed) is type(value) and observed == value
        if isinstance(value, float):
            equal = isinstance(observed, float) and observed.hex() == value.hex()
        if not equal:
            raise AFIRecoveryError(f"Materialized anatomical {context} has incompatible {key}")


def _materialized_segments(
    document: dict[str, Any],
    segments: tuple[AHAStrainSegment, ...],
    manifest: dict[str, Any],
    decoded: dict[str, Any],
    read_array: Callable[[Any], np.ndarray],
    array_catalog: list[dict[str, Any]] | None,
) -> tuple[AHAStrainSegment, ...]:
    """Verify named exported trajectories and attach their explicit paths."""
    raw = document["anatomical_strain"]
    if not isinstance(raw, dict):
        raise AFIRecoveryError("Materialized anatomical strain document must be an object")
    materialized = decode_exact_packet(raw)
    _expect_fields(
        materialized,
        {
            "schema": "anatomical_strain_v1",
            "segment_model": "aha17",
            "units": "percent",
            "recovery_status": "validated_native_tracking",
            "source_payload_sha256": decoded["source_payload_sha256"],
            "representation": "sixteen_native_clock_trajectories_and_three_central_contributors",
            "central_acceptance_status": "unvalidated",
            "central_combined_trajectory_status": "unvalidated",
            "clock_policy": "independent_native_source_view_clocks",
        },
        "document",
    )
    entries = materialized.get("segments")
    if not isinstance(entries, list) or len(entries) != 17:
        raise AFIRecoveryError("Materialized anatomical strain needs seventeen ordered segments")
    annotations = document.get("annotations")
    if not isinstance(annotations, list):
        raise AFIRecoveryError("Materialized anatomical strain needs annotation records")
    anatomical_annotations = [
        item for item in annotations if isinstance(item, dict) and item.get("kind") == "aha_segmental_strain"
    ]
    if len(anatomical_annotations) != 19:
        raise AFIRecoveryError("Materialized anatomical strain needs nineteen curve annotations")
    panels = {panel["source_view"]: panel for panel in manifest["panels"]}
    annotation_ids = set()
    value_paths = set()
    time_paths: dict[str, str] = {}
    result = []
    for segment, entry in zip(segments, entries, strict=True):
        if not isinstance(entry, dict):
            raise AFIRecoveryError("Materialized anatomical segment must be an object")
        expected = {
            "aha_segment_id": segment.aha_segment_id,
            "segment_name": segment.name,
            "units": "percent",
            "acceptance_status": segment.acceptance_status,
            "signed_peak_percent": segment.signed_peak_percent,
            "peak_method": segment.peak_method,
        }
        if segment.aha_segment_id != 17:
            expected["accepted"] = segment.accepted
        elif "accepted" in entry:
            raise AFIRecoveryError("Materialized central apex cannot assert an acceptance flag")
        _expect_fields(entry, expected, "segment")
        contributors = entry.get("contributors")
        if not isinstance(contributors, list) or len(contributors) != len(segment.contributors):
            raise AFIRecoveryError("Materialized segment contributor count differs from native recovery")
        traces = []
        for trace, exported in zip(segment.contributors, contributors, strict=True):
            if not isinstance(exported, dict):
                raise AFIRecoveryError("Materialized contributor must be an object")
            panel = panels[trace.source_view]
            semantic = f"aha{segment.aha_segment_id:02d}_{segment.name.replace(' ', '_')}_{trace.source_view.lower()}"
            metadata = {
                "aha_segment_id": segment.aha_segment_id,
                "segment_name": segment.name,
                "curve_semantic_id": semantic,
                "source_view": trace.source_view,
                "reconstruction_method": trace.reconstruction_method,
                "representation": "anatomical_annotation",
                "source_payload_sha256": decoded["source_payload_sha256"],
                "acceptance_status": segment.acceptance_status,
                "signed_peak_percent": trace.signed_peak_percent,
                "peak_method": "native_start_through_nearest_AVC_signed_0.75_rule",
                "native_ed_time": trace.native_ed_time,
                "avc_seconds": trace.avc_seconds,
                "native_avc_time": trace.native_avc_time,
                "nearest_avc_frame_index": trace.nearest_avc_frame_index,
                "native_closure_frame_index": panel["native_closure_frame_index"],
                "native_coordinate_correction_anchor_time": panel["native_coordinate_correction_anchor_time"],
                "native_tracking_source_error_percent": panel["max_native_tracking_source_error_percent"],
            }
            if trace.accepted is not None:
                metadata["accepted"] = trace.accepted
            elif "accepted" in exported:
                raise AFIRecoveryError("Materialized central contributor cannot assert an acceptance flag")
            for key in ("source_region_order_index", "source_offset"):
                if key in trace.provenance:
                    metadata[key] = trace.provenance[key]
            _expect_fields(exported, metadata, "contributor")
            if "units" in exported and exported["units"] != "percent":
                raise AFIRecoveryError("Materialized anatomical contributor has incompatible units")
            matches = [
                annotation
                for annotation in anatomical_annotations
                if isinstance(annotation.get("links"), dict)
                and annotation["links"].get("curve_semantic_id") == semantic
            ]
            if len(matches) != 1:
                raise AFIRecoveryError("Materialized contributor needs exactly one matching annotation")
            annotation = matches[0]
            target = annotation.get("target")
            if not isinstance(target, dict) or (
                target.get("type", target.get("entity_type")),
                target.get("semantic_id", target.get("target_role_id")),
                target.get("field"),
            ) != ("derived_curve", semantic, "values"):
                raise AFIRecoveryError("Materialized anatomical annotation target differs from its contributor")
            links = decode_exact_packet(annotation["links"])
            _expect_fields(links, metadata, "annotation")
            if "units" in links and links["units"] != "percent":
                raise AFIRecoveryError("Materialized anatomical annotation has incompatible units")
            if trace.accepted is None and "accepted" in links:
                raise AFIRecoveryError("Materialized central annotation cannot assert an acceptance flag")
            paths = []
            for field, native, units in (
                ("value", trace.values, {"percent"}),
                ("time", trace.timestamps, {"seconds", "s"}),
            ):
                ref = exported.get(field)
                if not isinstance(ref, dict) or annotation.get(field) != ref:
                    raise AFIRecoveryError("Materialized anatomical annotation reference differs from its contributor")
                if ref.get("format") != "zarr_array" or ("units" in ref and ref["units"] not in units):
                    raise AFIRecoveryError("Materialized anatomical array has incompatible format or units")
                values = read_array(ref)
                if values.shape != native.shape or values.dtype != native.dtype or values.tobytes() != native.tobytes():
                    raise AFIRecoveryError(
                        "Materialized anatomical values or native clock differ bitwise from recovery"
                    )
                if array_catalog is not None:
                    entries = [item for item in array_catalog if item.get("name") == ref["zarr_path"]]
                    if len(entries) != 1:
                        raise AFIRecoveryError("Materialized anatomical array needs one matching catalog entry")
                    catalog_entry = entries[0]
                    if catalog_entry.get("units") not in units:
                        raise AFIRecoveryError("Materialized anatomical catalog units differ from native calibration")
                    if "dtype" in catalog_entry and (
                        not isinstance(catalog_entry["dtype"], str) or np.dtype(catalog_entry["dtype"]) != values.dtype
                    ):
                        raise AFIRecoveryError("Materialized anatomical catalog dtype differs from its array")
                    if "shape" in catalog_entry and (
                        not isinstance(catalog_entry["shape"], list)
                        or any(type(size) is not int for size in catalog_entry["shape"])
                        or catalog_entry["shape"] != list(values.shape)
                    ):
                        raise AFIRecoveryError("Materialized anatomical catalog shape differs from its array")
                paths.append(ref["zarr_path"])
            if paths[0] in value_paths or (
                trace.source_view in time_paths and time_paths[trace.source_view] != paths[1]
            ):
                raise AFIRecoveryError("Materialized anatomical paths must bind unique values and one clock per view")
            value_paths.add(paths[0])
            time_paths[trace.source_view] = paths[1]
            annotation_ids.add(id(annotation))
            traces.append(
                replace(
                    trace,
                    value_path=paths[0],
                    timestamps_path=paths[1],
                    provenance={**trace.provenance, "value_path": paths[0], "timestamps_path": paths[1]},
                )
            )
        result.append(replace(segment, contributors=tuple(traces)))
    if len(annotation_ids) != 19 or len(set(time_paths.values())) != 3:
        raise AFIRecoveryError("Materialized anatomical annotations or native view clocks are ambiguous")
    return tuple(result)


def _segments(
    manifest: dict[str, Any], arrays: dict[str, np.ndarray], decoded: dict[str, Any]
) -> tuple[AHAStrainSegment, ...]:
    segments: dict[int, AHAStrainSegment] = {}
    caps = []
    states = {state["source_view"]: state for state in decoded["gui_states"]}
    for panel in manifest["panels"]:
        view = panel["source_view"]
        clock = arrays[f"{view}_timestamps"]
        stop = panel["nearest_native_AVC_frame_index"]
        ed_time = float(clock[panel["native_reference_frame_index"]])

        def contributor(
            values: np.ndarray, accepted: bool | None, method: str, source: dict[str, Any]
        ) -> AHAStrainContributor:
            return AHAStrainContributor(
                values.copy(),
                clock.copy(),
                view,
                accepted,
                "percent",
                ed_time,
                panel["exact_AVC_seconds"],
                ed_time + panel["exact_AVC_seconds"],
                stop,
                signed_peak(values, stop),
                method,
                {
                    **source,
                    "source_payload_sha256": decoded["source_payload_sha256"],
                    "native_coordinate_correction_anchor_time": states[view]["native_cursor_time"],
                    "native_closure_frame_index": panel["native_closure_frame_index"],
                },
            )

        for index, aha_id in enumerate(panel["basal_mid_aha_ids_in_primary_order"]):
            accepted = panel["basal_mid_accepted"][index]
            region = next(
                item
                for item in decoded["source_regions"]
                if item["source_view"] == view and item["source_region_order_index"] == index
            )
            trace = contributor(
                arrays[f"{view}_primary6"][index],
                accepted,
                "native_primary_source_curve",
                {"source_region_order_index": index, "source_offset": region["primary_curve"]["source_offset"]},
            )
            segments[aha_id] = _segment(aha_id, trace)
        for target in panel["candidate_apical_targets"]:
            trace = contributor(
                arrays[target["array_key"]],
                target["accepted"],
                target["reconstruction_method"],
                {"source_array_key": target["array_key"], "source_validation": dict(panel)},
            )
            segments[target["candidate_aha_segment_id"]] = _segment(target["candidate_aha_segment_id"], trace)
        caps.append(
            contributor(
                arrays[panel["cap_contributor_array_key"]],
                None,
                panel["tracking_reconstruction_method"],
                {"source_validation": dict(panel)},
            )
        )
    if set(segments) != set(range(1, 17)) or {cap.source_view for cap in caps} != {"A2C", "A3C", "A4C"}:
        raise AFIRecoveryError("Recovery did not produce sixteen segments and three central contributors")
    segments[17] = AHAStrainSegment(
        17,
        AHA_SEGMENT_NAMES[16],
        tuple(sorted(caps, key=lambda cap: cap.source_view)),
        None,
        "unvalidated",
        manifest["candidate_CAP_signed_peak_percent"],
        "mean_native_contributor_extrema_then_signed_0.75_rule",
    )
    return tuple(segments[index] for index in range(1, 18))


def _segment(aha_id: int, contributor: AHAStrainContributor) -> AHAStrainSegment:
    return AHAStrainSegment(
        aha_id,
        AHA_SEGMENT_NAMES[aha_id - 1],
        (contributor,),
        contributor.accepted,
        "accepted" if contributor.accepted else "rejected",
        contributor.signed_peak_percent,
        "native_start_through_nearest_AVC_signed_0.75_rule",
    )
