"""Catalog records locate their own data, and owned stores close on every exit."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import zarr
from conftest import croissant_row, write_croissant

from echoxflow import RecordingStore, load_croissant, open_recording


def test_catalog_root_survives_cache_and_overrides_environment(tmp_path, monkeypatch):
    row = croissant_row("r", content_types=["2d_brightness_mode"], array_paths=["data/2d_brightness_mode"])
    row["recordings/zarr_path"] = "r.zarr"
    write_croissant(tmp_path, [row])
    group = zarr.open_group(tmp_path / "r.zarr", mode="w")
    group.create_array("data/2d_brightness_mode", data=np.ones((2, 3, 4), dtype=np.uint8))
    monkeypatch.setenv("ECHOXFLOW_DATA_ROOT", str(tmp_path / "wrong"))
    for _ in range(2):
        record = load_croissant(root=tmp_path).recordings[0]
        assert record.path() == tmp_path / "r.zarr"
        assert record.path(tmp_path / "override") == tmp_path / "override/r.zarr"
        with open_recording(record) as store:
            values = store.load_stream("2d_brightness_mode").data
        assert values.sum() == 24


def test_context_manager_closes_once_even_on_error():
    close = Mock()
    store = RecordingStore(Path("synthetic.zarr"), SimpleNamespace(store=SimpleNamespace(close=close)))
    with pytest.raises(RuntimeError):
        with store:
            raise RuntimeError("failure")
    store.close()
    close.assert_called_once()
    with pytest.raises(ValueError, match="closed"):
        with store:
            pass


def test_zip_store_context_closes_handle_and_keeps_arrays(tmp_path):
    path = tmp_path / "recording.zip"
    with zarr.storage.ZipStore(path, mode="w") as archive:
        group = zarr.open_group(store=archive, mode="w")
        group.create_array("data/2d_brightness_mode", data=np.ones((2, 3, 4), dtype=np.uint8))
    with open_recording(path) as store:
        values = store.load_stream("2d_brightness_mode").data
        archive = store.group.store
    assert not archive._is_open
    assert values.sum() == 24


def test_explicit_root_with_separate_manifest_is_not_hidden_by_cache(tmp_path):
    write_croissant(tmp_path, [croissant_row("r", content_types=[], array_paths=[])])
    manifest = tmp_path / "croissant.json"
    for root in (tmp_path / "first", tmp_path / "second"):
        record = load_croissant(manifest, root=root).recordings[0]
        assert record.source_root == root
        assert record.path() == root / record.zarr_path
