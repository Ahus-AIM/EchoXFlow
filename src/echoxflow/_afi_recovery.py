"""Guarded native automated function imaging (AFI, GE regional strain) coordinate reconstruction from export arrays.

The package core has no DICOM, JPEG or raw-file dependency. The stored native
curves independently validate the fixed tracking law before anatomical output.
The central apex retains three contributors on their original native clocks.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class AFIRecoveryError(ValueError):
    """The corrected native state or tracking reconstruction is unsupported."""


def regional_index_overlap(bounds: np.ndarray, edge_count: int) -> np.ndarray:
    edges = np.arange(edge_count)
    return np.asarray(np.maximum(0, np.minimum(bounds[:, 1, None], edges + 1) - np.maximum(bounds[:, 0, None], edges)))


BASAL_MID = {"A2C": [1, 4, 7, 10], "A3C": [2, 5, 8, 11], "A4C": [3, 6, 9, 12]}
APICAL = {"A2C": [15, 13], "A4C": [14, 16]}
AHA_SEGMENT_NAMES = (
    "basal anterior",
    "basal anteroseptal",
    "basal inferoseptal",
    "basal inferior",
    "basal inferolateral",
    "basal anterolateral",
    "mid anterior",
    "mid anteroseptal",
    "mid inferoseptal",
    "mid inferior",
    "mid inferolateral",
    "mid anterolateral",
    "apical anterior",
    "apical septal",
    "apical inferior",
    "apical lateral",
    "apex",
)


def weighted_windows(bounds: np.ndarray, density: np.ndarray) -> np.ndarray:
    weights = regional_index_overlap(bounds, len(density)) * density
    return np.asarray(weights / weights.sum(axis=1, keepdims=True))


def signed_peak(values: np.ndarray, stop: int) -> float:
    low, high = float(values[: stop + 1].min()), float(values[: stop + 1].max())
    return high if high >= 0.75 * abs(low) else low


def recover_native_arrays(
    decoded: dict[str, Any], native_panels: dict[str, dict[str, np.ndarray]]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Apply the same research contract to guarded original or exported inputs."""
    if decoded["unresolved_header_u32_fields"] != [3, 4, 3, 19, 0, 0]:
        raise AFIRecoveryError("Unvalidated AFI source configuration for spatial recovery")
    if set(native_panels) != {"A2C", "A3C", "A4C"}:
        raise AFIRecoveryError("Missing or unexpected native input views")
    arrays: dict[str, np.ndarray] = {}
    panels = []
    cap_minima, cap_maxima = [], []
    accepted = {
        (region["source_view"], region["source_region_order_index"]): region["source_region_prefix"]["accepted_bool"]
        for region in decoded["source_regions"]
    }
    for state in decoded["gui_states"]:
        if (state["unresolved_pre_avc_u32"], state["unresolved_pre_avc_bool"]) != (0, True):
            raise AFIRecoveryError("Unvalidated native GUI configuration for spatial recovery")
        view = state["source_view"]
        global_source = next(item for item in decoded["global_summaries"] if item["source_view"] == view)
        inputs = native_panels[view]
        global_values, fine, primary, secondary = (
            np.asarray(inputs[key]) for key in ("global", "fine", "primary", "secondary")
        )
        clock = np.asarray(state["native_clock"], dtype=np.float64)
        if not np.isfinite(clock).all() or not np.all(np.diff(clock) > 0):
            raise AFIRecoveryError("Native source clock must be finite and strictly increasing")
        tracking = state["tracking_state"]["tracked_contours"]
        points = np.asarray(inputs["points"], dtype=np.float64)
        expected_shapes = (
            (len(clock),),
            (60, len(clock)),
            (6, len(clock)),
            (6, len(clock)),
            (len(clock), tracking["longitudinal_point_count"], 5, 2),
        )
        for values, shape in zip((global_values, fine, primary, secondary, points), expected_shapes, strict=True):
            if values.shape != shape or not np.isfinite(values).all():
                raise AFIRecoveryError(f"{view}: malformed native array shape or values")
        if not np.array_equal(primary[2:], secondary[2:]):
            raise AFIRecoveryError(f"{view}: unvalidated primary/secondary nonbasal curve relation")
        biological = np.concatenate([global_values[None], fine, primary, secondary])
        zeros = np.flatnonzero(np.all(biological == 0, axis=0))
        reference = np.flatnonzero(clock == global_source["curve"]["native_reference_time"])
        if len(zeros) != 2 or len(reference) != 1 or zeros[0] != reference[0]:
            raise AFIRecoveryError("Unsupported biological reference/closure layout")
        ed, closure = (int(index) for index in zeros)
        stop = int(np.argmin(abs(clock - (clock[ed] + state["avc_seconds"]))))
        regions = {
            region["source_region_order_index"]: region
            for region in decoded["source_regions"]
            if region["source_view"] == view
        }
        if set(regions) != set(range(6)):
            raise AFIRecoveryError(f"{view}: unsupported native regional summary indices")
        for index, curve_values in enumerate(primary):
            window = curve_values[: stop + 1]
            low_index, high_index = int(np.argmin(window)), int(np.argmax(window))
            peak_index = high_index if window[high_index] >= 0.75 * abs(window[low_index]) else low_index
            native = next(
                item for item in regions[index]["timed_summary"]["records"] if item["source_quantity_id"] == 3
            )
            if float(window[peak_index]) != native["value"] or clock[peak_index] != native["native_marker_time"]:
                raise AFIRecoveryError(f"{view}: unsupported native signed-peak value or marker policy")
        edge_count = points.shape[1] - 1
        tip = state["tracking_state"]["native_tip_material_index"]
        if not 0 < tip < edge_count:
            raise AFIRecoveryError("Native material tip does not define two sides")
        reference_points = (points[ed, :, 2] + points[closure, :, 2]) / 2
        density = np.linalg.norm(np.diff(reference_points, axis=0), axis=-1)
        if np.any(density <= 0):
            raise AFIRecoveryError("Nonpositive native material reference length")
        cursor = float(state["native_cursor_time"])
        if not np.isfinite(cursor) or not clock[ed] < cursor < clock[closure]:
            raise AFIRecoveryError(f"{view}: native tracking correction anchor must lie inside ED and closure")
        phase = np.where(
            clock <= cursor,
            0.5 * (clock - clock[ed]) / (cursor - clock[ed]),
            0.5 + 0.5 * (clock - cursor) / (clock[closure] - cursor),
        )
        midwall = points[:, :, 2]
        corrected = midwall - (phase - 0.5)[:, None, None] * (midwall[closure] - midwall[ed])[None]
        corrected_lengths = np.linalg.norm(np.diff(corrected, axis=1), axis=-1).T
        tracking_strain = 100 * (corrected_lengths / density[:, None] - 1)
        centers = np.linspace(edge_count / 12, 11 * edge_count / 12, 60)
        fine_bounds = np.column_stack([centers - edge_count / 12, centers + edge_count / 12])
        cuts = np.r_[np.linspace(0, tip, 4), np.linspace(tip, edge_count, 4)[1:]]
        secondary_bounds = np.column_stack([cuts[:-1], cuts[1:]])
        primary_bounds = np.array(
            [[tip / 9, tip / 3], [tip + 2 * (edge_count - tip) / 3, tip + 8 * (edge_count - tip) / 9]]
        )
        operator = np.concatenate(
            [
                weighted_windows(fine_bounds, density),
                weighted_windows(secondary_bounds, density),
                weighted_windows(np.array([[0, edge_count]]), density),
                weighted_windows(primary_bounds, density),
            ]
        )
        physical_order = [0, 2, 4, 5, 3, 1] if view == "A4C" else [1, 3, 5, 4, 2, 0]
        observations = np.concatenate(
            [fine, secondary[physical_order], global_values[None], primary[[0, 1] if view == "A4C" else [1, 0]]]
        )
        tracking_residual = float(np.max(abs(operator @ tracking_strain - observations)))
        if tracking_residual > 2e-4:
            raise AFIRecoveryError(
                f"{view}: unsupported native tracking law, source trajectory error {tracking_residual}"
            )
        inverse = np.linalg.pinv(operator, rcond=1e-8)
        singular_values = np.linalg.svd(operator, compute_uv=False)
        retained = singular_values[singular_values > 1e-8 * singular_values[0]]
        residual = float(np.max(abs(operator @ inverse @ observations - observations)))
        if residual > 2e-4:
            raise AFIRecoveryError(f"{view}: unsupported source spatial operator, residual {residual}")
        right = edge_count - tip
        candidate_bounds = np.array(
            [[2 * tip / 3, 5 * tip / 6], [tip + right / 6, tip + right / 3], [5 * tip / 6, tip + right / 6]]
        )
        target = weighted_windows(candidate_bounds, density)
        null_residual = np.linalg.norm(target - target @ inverse @ operator, axis=1)
        arrays.update(
            {
                f"{view}_timestamps": clock,
                f"{view}_global": global_values,
                f"{view}_fine60": fine,
                f"{view}_primary6": primary,
                f"{view}_secondary6": secondary,
                f"{view}_reference_edge_lengths": density,
            }
        )
        targets = []
        for row, aha_id in enumerate(APICAL.get(view, [])):
            source_region = (4 + row) if view == "A4C" else (5 - row)
            is_accepted = accepted[(view, source_region)] and accepted[("A3C", source_region)]
            identifiable = bool(null_residual[row] < 1e-8)
            key = f"candidate_AHA{aha_id}_{view}"
            target_values = target[row] @ tracking_strain
            arrays[key] = target_values
            targets.append(
                {
                    "candidate_aha_segment_id": aha_id,
                    "accepted": is_accepted,
                    "identifiable_from_source_operator": identifiable,
                    "recoverable_from_native_tracking": True,
                    "reconstruction_method": "native_cursor_anchored_coordinate_drift",
                    "rowspace_residual": float(null_residual[row]),
                    "array_key": key,
                    "candidate_signed_peak_percent": signed_peak(target_values, stop),
                }
            )
        cap = target[2] @ tracking_strain
        arrays[f"candidate_CAP_{view}"] = cap
        cap_minima.append(float(cap[: stop + 1].min()))
        cap_maxima.append(float(cap[: stop + 1].max()))
        panels.append(
            {
                "source_view": view,
                "native_reference_frame_index": ed,
                "native_closure_frame_index": closure,
                "nearest_native_AVC_frame_index": stop,
                "exact_AVC_seconds": state["avc_seconds"],
                "native_coordinate_correction_anchor_time": cursor,
                "tracking_reconstruction_method": "native_cursor_anchored_coordinate_drift",
                "max_native_tracking_source_error_percent": tracking_residual,
                "native_tracking_source_error_tolerance_percent": 2e-4,
                "source_operator_rank": len(retained),
                "source_operator_relative_singular_cutoff": 1e-8,
                "source_operator_retained_condition": float(retained[0] / retained[-1]),
                "source_edge_count": edge_count,
                "max_source_operator_residual_percent": residual,
                "basal_mid_aha_ids_in_primary_order": BASAL_MID[view],
                "basal_mid_accepted": [accepted[(view, index)] for index in range(4)],
                "candidate_apical_targets": targets,
                "cap_contributor_array_key": f"candidate_CAP_{view}",
            }
        )
    low, high = float(np.mean(cap_minima)), float(np.mean(cap_maxima))
    manifest = {
        "schema": "afi_source_recovery_research_v2",
        "source_payload_sha256": decoded["source_payload_sha256"],
        "full_AHA17_recovery": False,
        "JPEG_values_used": False,
        "candidate_spatial_mapping_status": "native_tracking_law_validated_against_stored_source_trajectories",
        "aha1_to16_trajectories_recovered": True,
        "all_numeric_aha17_summaries_available": True,
        "central_apex_representation": "three_native_clock_contributors_with_scalar_aggregation",
        "nullspace_prior_used": False,
        "native_clocks_preserved": True,
        "central_apex_acceptance_status": "unvalidated",
        "candidate_CAP_signed_peak_percent": high if high >= 0.75 * abs(low) else low,
        "panels": panels,
        "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
    }
    return manifest, arrays
