"""Stores and manifests from a schema older than EchoXFlow-2 are rejected; newer or unversioned ones load."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from conftest import croissant_row, write_croissant

from echoxflow import load_croissant, open_recording
from echoxflow.croissant import MINIMUM_SCHEMA_GENERATION, schema_generation

zarr = pytest.importorskip("zarr")


def _store(tmp_path: Path, schema_version: str | None) -> Path:
    path = tmp_path / "rec.zarr"
    group = zarr.open_group(path, mode="w", zarr_format=2)
    group.create_array("data/2d_brightness_mode", data=np.zeros((2, 3, 3), dtype=np.uint8), chunks=(2, 3, 3))
    group.create_array("timestamps/2d_brightness_mode", data=np.array([0.0, 0.1]), chunks=(2,))
    if schema_version is not None:
        group.attrs["schema_version"] = schema_version
    zarr.consolidate_metadata(path)
    return path


def test_schema_generation_parses_only_echoxflow_tags() -> None:
    assert schema_generation("EchoXFlow-1") == 1
    assert schema_generation("EchoXFlow-12") == 12
    assert schema_generation(4) is None
    assert schema_generation("v2") is None
    assert schema_generation(None) is None
    assert MINIMUM_SCHEMA_GENERATION == 2


def test_older_store_schema_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="EchoXFlow-1 schema.*EchoXFlow-2 or newer.*1.0.0"):
        open_recording(_store(tmp_path, "EchoXFlow-1"))


@pytest.mark.parametrize("schema_version", ["EchoXFlow-2", "EchoXFlow-3", None])
def test_current_newer_and_unversioned_stores_open(tmp_path: Path, schema_version: str | None) -> None:
    store = open_recording(_store(tmp_path, schema_version))
    assert store.load_stream("2d_brightness_mode").data.shape == (2, 3, 3)


def _manifest_with_provenance(root: Path, schema_version: str) -> Path:
    write_croissant(
        root,
        [
            croissant_row(
                "recording_a",
                content_types=["2d_brightness_mode"],
                array_paths=["data/2d_brightness_mode"],
                exam_id="exam_a",
            )
        ],
    )
    manifest = root / "croissant.json"
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["prov:wasGeneratedBy"] = [
        {"@type": "prov:Activity", "name": "EchoXFlow export", "schema_version": schema_version}
    ]
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    return manifest


def test_older_manifest_schema_is_rejected(tmp_path: Path) -> None:
    manifest = _manifest_with_provenance(tmp_path, "EchoXFlow-1")
    with pytest.raises(ValueError, match="Croissant manifest .*EchoXFlow-1 schema"):
        load_croissant(manifest, cache=False)


@pytest.mark.parametrize("schema_version", ["EchoXFlow-2", "EchoXFlow-3"])
def test_current_and_newer_manifests_load(tmp_path: Path, schema_version: str) -> None:
    catalog = load_croissant(_manifest_with_provenance(tmp_path, schema_version), cache=False)
    assert [record.recording_id for record in catalog.recordings] == ["recording_a"]
