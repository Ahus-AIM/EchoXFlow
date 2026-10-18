from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from scripts import croissant_summary_table as summary


def test_summary_reads_close_the_store_on_success_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    array = Mock()
    array.__getitem__ = Mock(return_value=np.array([0.0, 0.1, 0.2]))
    array.nbytes_stored.return_value = 123
    store = Mock()
    group = {"timestamps/ecg": array}

    class Group(dict):
        attrs = {"schema_version": "EchoXFlow-2"}

    group = Group(group)
    group.store = store
    monkeypatch.setattr(summary, "open_recording", lambda path: SimpleNamespace(group=group))
    path = Path("recording.zip")
    assert summary.read_zarr_attrs(path) == group.attrs
    assert summary.zarr_timestamp_delta_s(zarr_path=path, array_path="timestamps/ecg") == pytest.approx(0.1)
    assert summary.zarr_array_size_bytes(zarr_path=path, array_path="timestamps/ecg") == 123
    assert summary.zarr_timestamp_delta_s(zarr_path=path, array_path="missing") is None
    assert summary.zarr_array_size_bytes(zarr_path=path, array_path="missing") is None
    assert store.close.call_count == 5


@pytest.mark.skipif(sys.platform != "linux", reason="uses Linux descriptor counts and resource limits")
def test_many_zip_recordings_fit_within_a_small_descriptor_budget(tmp_path: Path) -> None:
    script = """
import json, os, resource, sys, zipfile
from pathlib import Path
from scripts.croissant_summary_table import read_zarr_attrs
root = Path(sys.argv[1])
for i in range(80):
    with zipfile.ZipFile(root / f'{i}.zip', 'w') as archive:
        archive.writestr('.zgroup', '{"zarr_format": 2}')
        archive.writestr('.zattrs', json.dumps({'recording_id': str(i)}))
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
budget = len(os.listdir('/proc/self/fd')) + 16
resource.setrlimit(resource.RLIMIT_NOFILE, (min(soft, budget), hard))
for i in range(80):
    assert read_zarr_attrs(root / f'{i}.zip')['recording_id'] == str(i)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env=dict(os.environ, PYTHONPATH=str(Path(summary.__file__).resolve().parents[1])),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
