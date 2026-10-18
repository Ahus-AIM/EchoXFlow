"""Validate corrected native automated function imaging (AFI, GE regional strain) metadata and its source arrays."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from typing import Any

import numpy as np

from echoxflow._afi_recovery import AFIRecoveryError

VIEWS = {"A3C", "A4C", "A2C"}
ROLE_COUNTS = {"global": 1, "fine_profile": 60, "primary_curve": 6, "secondary_curve": 6}
PANEL_KEYS = {"global": "global", "fine_profile": "fine", "primary_curve": "primary", "secondary_curve": "secondary"}


def decode_exact_packet(value: Any) -> Any:
    """Reverse the proposal's Python float.hex encoding without decimal rounding."""
    if isinstance(value, dict):
        if "ieee754_hex" in value:
            if set(value) != {"ieee754_hex"} or not isinstance(value["ieee754_hex"], str):
                raise AFIRecoveryError("Malformed exact native float packet")
            try:
                number = float.fromhex(value["ieee754_hex"])
            except ValueError as error:
                raise AFIRecoveryError("Invalid exact native float") from error
            if not math.isfinite(number) or number.hex() != value["ieee754_hex"]:
                raise AFIRecoveryError("Nonfinite or noncanonical native float packet")
            return number
        return {key: decode_exact_packet(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_exact_packet(item) for item in value]
    if isinstance(value, float):
        raise AFIRecoveryError("Native float must use exact hexadecimal encoding")
    return value


def _integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise AFIRecoveryError(f"Invalid {name}: expected a nonnegative integer")
    return value


def _verify_source_fingerprint(binding: dict[str, Any], field: str, values: np.ndarray) -> None:
    expected = binding.get(field)
    actual = hashlib.sha256(np.asarray(values, dtype="<f4").tobytes()).hexdigest()
    if not isinstance(expected, str) or len(expected) != 64 or expected != actual:
        raise AFIRecoveryError(f"Missing or incompatible native source fingerprint: {field}")


def _annotation(
    manifest: dict[str, Any], entity: str, field: str, link: str, identifier: int, binding: dict[str, Any]
) -> dict[str, Any]:
    annotations = manifest.get("annotations")
    if not isinstance(annotations, list):
        raise AFIRecoveryError("Raw package lacks annotation records")
    semantic = binding.get("export_semantic_id")
    export_entity = binding.get("export_entity_type", entity)
    export_field = binding.get("export_field", field)
    if semantic is not None and (not isinstance(semantic, str) or not semantic):
        raise AFIRecoveryError("Invalid controlled export semantic binding")
    matches = []
    for item in annotations:
        if not isinstance(item, dict) or not isinstance(item.get("target"), dict):
            continue
        target = item["target"]
        if semantic is not None:
            match = (
                target.get("type", target.get("entity_type")) == export_entity
                and target.get("semantic_id", target.get("target_role_id")) == semantic
                and target.get("field") == export_field
            )
        else:
            match = (
                target.get("entity_type") == entity
                and target.get("field") == field
                and item.get("links", {}).get(link) == identifier
            )
        if match:
            matches.append(item)
    if len(matches) != 1:
        raise AFIRecoveryError(f"Expected exactly one {entity}/{field} annotation for {link}={identifier}")
    return matches[0]


def _bind_packet(
    manifest: dict[str, Any], read_array: Callable[[Any], np.ndarray]
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    packet = manifest.get("native_afi_state")
    if not isinstance(packet, dict) or (
        packet.get("schema"),
        packet.get("decode_status"),
        packet.get("numeric_encoding"),
    ) != ("native_afi_state_v1", "validated_layout", "ieee754_hex"):
        raise AFIRecoveryError("Missing or unsupported native AFI state packet")
    decoded = decode_exact_packet(packet.get("state"))
    if not isinstance(decoded, dict) or decoded.get("source_serialization_version") != 6:
        raise AFIRecoveryError("Unsupported native AFI serialization version")
    states = {state["source_view"]: state for state in decoded["gui_states"]}
    if len(decoded["gui_states"]) != 3 or set(states) != VIEWS:
        raise AFIRecoveryError("Native GUI state requires three distinct canonical source views")
    descriptors = {}
    if len(decoded["global_summaries"]) != 3 or len(decoded["source_regions"]) != 18:
        raise AFIRecoveryError("Unsupported native summary descriptor counts")
    for item in decoded["global_summaries"]:
        view = item["source_view"]
        if len(item["fine_profile_curves"]) != 60 or (view, "global", 0) in descriptors:
            raise AFIRecoveryError("Unsupported or duplicate native global descriptor")
        descriptors[(view, "global", 0)] = item["curve"]
        for index, curve in enumerate(item["fine_profile_curves"]):
            descriptors[(view, "fine_profile", index)] = curve
    for item in decoded["source_regions"]:
        for role in ("primary_curve", "secondary_curve"):
            key = (item["source_view"], role, item["source_region_order_index"])
            if key in descriptors:
                raise AFIRecoveryError("Duplicate native regional descriptor")
            descriptors[key] = item[role]
    expected = {(view, role, index) for view in VIEWS for role, count in ROLE_COUNTS.items() for index in range(count)}
    if set(descriptors) != expected:
        raise AFIRecoveryError("Incomplete native source curve descriptors")
    bindings = packet.get("export_curve_bindings", packet.get("source_curve_bindings"))
    if not isinstance(bindings, list) or len(bindings) != len(expected):
        raise AFIRecoveryError("Missing or incomplete native source curve bindings")
    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    values: dict[tuple[str, str, int], np.ndarray] = {}
    used_records = set()
    group_semantics: dict[int, str | None] = {}
    for binding in bindings:
        key = (binding["source_view"], binding["source_role"], _integer(binding["source_order_index"], "source order"))
        if key not in expected or key in values:
            raise AFIRecoveryError("Unsupported or duplicate native source curve binding")
        group = _integer(binding["curve_group_id"], "curve group")
        record = _integer(binding["curve_stack_record_index"], "curve stack record")
        semantic = binding.get("export_semantic_id")
        if group in group_semantics and group_semantics[group] != semantic:
            raise AFIRecoveryError("Curve group has conflicting controlled semantic bindings")
        group_semantics[group] = semantic
        if (group, record) in used_records:
            raise AFIRecoveryError("Two native source roles bind to the same exported record")
        used_records.add((group, record))
        if group not in cache:
            annotation = _annotation(manifest, "curve_group", "curve_stack", "group_id", group, binding)
            stack = read_array(annotation.get("value"))
            clock = read_array(annotation.get("time"))
            if stack.ndim != 2 or stack.dtype != np.dtype("float32") or clock.dtype != np.dtype("float64"):
                raise AFIRecoveryError("Native curve stack must be float32 [record,time] with float64 native clock")
            cache[group] = stack, clock
        stack, clock = cache[group]
        descriptor = descriptors[key]
        native_clock = np.asarray(states[key[0]]["native_clock"], dtype=np.float64)
        if (
            record >= len(stack)
            or stack.shape[1] != descriptor["sample_count"]
            or clock.shape != native_clock.shape
            or not np.array_equal(clock.view(np.uint64), native_clock.view(np.uint64))
        ):
            raise AFIRecoveryError("Native curve binding has incompatible record count or native clock")
        if binding.get("source_offset") != descriptor["source_offset"]:
            raise AFIRecoveryError("Native curve binding disagrees with retained source provenance")
        _verify_source_fingerprint(binding, "source_values_sha256", stack[record])
        values[key] = stack[record].copy()
    panels: dict[str, dict[str, np.ndarray]] = {
        view: {
            PANEL_KEYS[role]: values[(view, role, 0)]
            if role == "global"
            else np.stack([values[(view, role, index)] for index in range(count)])
            for role, count in ROLE_COUNTS.items()
        }
        for view in VIEWS
    }
    contours = packet.get("export_contour_bindings", packet.get("source_contour_bindings"))
    if not isinstance(contours, list) or len(contours) != 3:
        raise AFIRecoveryError("Missing native source contour bindings")
    used_views, used_sequences = set(), set()
    for binding in contours:
        view = binding["source_view"]
        sequence = _integer(binding["contour_sequence_id"], "contour sequence")
        if view not in VIEWS or view in used_views or sequence in used_sequences:
            raise AFIRecoveryError("Unsupported or duplicate native source contour binding")
        if binding.get("export_coordinate_transform") != "reflect_x":
            raise AFIRecoveryError("Unsupported exported contour coordinate transform")
        used_views.add(view)
        used_sequences.add(sequence)
        count = _integer(binding["native_frame_count"], "native frame count")
        longitudinal = _integer(binding["longitudinal_point_count"], "longitudinal point count")
        native_contour = states[view]["tracking_state"]["tracked_contours"]
        if (
            count != len(states[view]["native_clock"])
            or count != native_contour["frame_count"]
            or longitudinal != native_contour["longitudinal_point_count"]
            or native_contour["wall_point_count"] != 5
            or native_contour["source_point_order"] != "longitudinal_then_wall"
            or longitudinal < 2
        ):
            raise AFIRecoveryError("Invalid native contour dimensions")
        if binding.get("export_entity_type", "contour_sequence") not in {"contour_sequence", "linked_panel"}:
            raise AFIRecoveryError("Unsupported controlled contour target type")
        if binding.get("export_field", "points") not in {"points", "contour_points"}:
            raise AFIRecoveryError("Unsupported controlled contour target field")
        annotation = _annotation(manifest, "contour_sequence", "points", "sequence_id", sequence, binding)
        points = read_array(annotation.get("value"))
        time_ref = annotation.get("time")
        if not isinstance(time_ref, dict):
            raise AFIRecoveryError("Native contour requires a retained frame order or clock reference")
        frame_axis = read_array(time_ref)
        if (
            time_ref.get("role") == "frame_index"
            or binding.get("export_entity_type", "contour_sequence") == "contour_sequence"
        ):
            valid_axis = frame_axis.dtype == np.dtype("int64") and np.array_equal(frame_axis, np.arange(count))
        else:
            native_clock = np.asarray(states[view]["native_clock"], dtype=np.float64)
            valid_axis = (
                frame_axis.dtype == np.dtype("float64")
                and frame_axis.shape == native_clock.shape
                and np.array_equal(frame_axis.view(np.uint64), native_clock.view(np.uint64))
            )
        if not valid_axis:
            raise AFIRecoveryError("Native contour frame order or clock differs from retained source state")
        if points.dtype != np.dtype("float32") or points.shape != (count, 5 * longitudinal, 2):
            raise AFIRecoveryError("Native contour must be float32 [native frame,5*longitudinal point,xy]")
        points = points.reshape(count, longitudinal, 5, 2).copy()
        points[..., 0] *= -1
        _verify_source_fingerprint(binding, "source_points_sha256", points)
        panels[view]["points"] = points
    return decoded, panels


def bind_native_packet(
    document: dict[str, Any], read_array: Callable[[Any], np.ndarray]
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    """Refuse unsupported packets, reused bindings, changed arrays or clocks."""
    try:
        decoded, panels = _bind_packet(document, read_array)
        digest = decoded.get("source_payload_sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise AFIRecoveryError("Missing or invalid retained source payload fingerprint")
        if any(
            type(region["source_region_prefix"]["accepted_bool"]) is not bool for region in decoded["source_regions"]
        ):
            raise AFIRecoveryError("Native regional acceptance flags must be booleans")
        return decoded, panels
    except (KeyError, TypeError, ValueError, AttributeError, IndexError, OverflowError) as error:
        if isinstance(error, AFIRecoveryError):
            raise
        raise AFIRecoveryError(f"Malformed corrected native AFI packet: {error}") from error
