"""Recording manifest attribute helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_INTERNAL_MANIFEST_KEYS = {
    "annotation_dataset_id",
    "field_type",
    "panel_role_id",
    "sequence_id",
    "curve_group_id",
    "annotation_id",
    "target_role_id",
    "entity_type",
    "source_kind",
    "raw_source",
    "provenance",
    "pipeline",
}
_ANNOTATION_ID_CONTAINER_KEYS = {"curve_groups", "contour_sequences"}


def manifest_documents(group_attrs: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Return the public recording manifest from root Zarr attrs."""
    recording_manifest = _attr_or_metadata(group_attrs, "recording_manifest")
    return (recording_manifest,) if isinstance(recording_manifest, Mapping) else ()


def normalize_public_manifest_attrs(attrs: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize exported public attrs to the current `recording_manifest` schema."""
    normalized = dict(attrs or {})
    recording_manifest = normalized.pop("recording_manifest", None)
    if isinstance(recording_manifest, Mapping):
        normalized["recording_manifest"] = _recording_manifest_document(recording_manifest)
    return normalized


def sanitize_public_manifest(value: Any, *, _container_key: str | None = None) -> Any:
    """Strip internal annotation plumbing from public controlled metadata."""
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            text_key = str(key)
            if text_key in _INTERNAL_MANIFEST_KEYS:
                continue
            if text_key == "geometry" and _container_key == "metadata":
                continue
            if text_key == "annotation_ids" and _container_key in _ANNOTATION_ID_CONTAINER_KEYS:
                continue
            sanitized_item = sanitize_public_manifest(item, _container_key=text_key)
            sanitized[text_key] = sanitized_item
        _set_manifest_type(sanitized)
        return sanitized
    if isinstance(value, list):
        return [sanitize_public_manifest(item, _container_key=_container_key) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_public_manifest(item, _container_key=_container_key) for item in value)
    return value


def _recording_manifest_document(recording_manifest: Mapping[str, Any]) -> dict[str, Any]:
    sanitized = sanitize_public_manifest(recording_manifest)
    return _with_manifest_type(dict(sanitized) if isinstance(sanitized, Mapping) else {})


def _with_manifest_type(manifest: dict[str, Any]) -> dict[str, Any]:
    _set_manifest_type(manifest)
    return manifest


def _set_manifest_type(manifest: dict[str, Any]) -> None:
    if manifest.get("manifest_type"):
        return
    parts = _manifest_type_parts(manifest)
    if parts:
        manifest["manifest_type"] = "+".join(part for part in ("2d", "3d", "strain") if part in parts)


def _manifest_type_parts(manifest: Mapping[str, Any]) -> set[str]:
    parts: set[str] = set()
    sectors = manifest.get("sectors")
    if isinstance(sectors, list):
        sector_parts = _sector_manifest_type_parts(sectors)
        parts.update(sector_parts)
    if any(key in manifest for key in ("tracks",)):
        parts.add("2d")
    if any(key in manifest for key in ("linked_volume", "linked_mesh_sequences")):
        parts.add("3d")
    if any(key in manifest for key in ("annotation_type", "linked_panels", "linked_curves", "output_timeline")):
        parts.add("strain")
    return parts


def _sector_manifest_type_parts(sectors: list[Any]) -> set[str]:
    parts: set[str] = set()
    for sector in sectors:
        if not isinstance(sector, Mapping):
            continue
        if _is_3d_sector(sector):
            parts.add("3d")
        else:
            parts.add("2d")
    return parts


def _is_3d_sector(sector: Mapping[str, Any]) -> bool:
    geometry = sector.get("geometry")
    coordinate_system = geometry.get("coordinate_system") if isinstance(geometry, Mapping) else None
    if coordinate_system == "spherical_sector_3d":
        return True
    frames = sector.get("frames")
    frame_path = None
    if isinstance(frames, Mapping):
        frame_path = frames.get("array_path") or frames.get("zarr_path") or frames.get("path")
    return str(frame_path or "").strip("/") == "data/3d_brightness_mode"


def _attr_or_metadata(group_attrs: Mapping[str, Any], key: str) -> Any:
    return group_attrs.get(key)
