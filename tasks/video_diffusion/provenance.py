"""Fail-closed identities for resumable experiments on immutable dataset exports."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any, Mapping

from echoxflow import data_root
from tasks.utils.atomic import atomic_write_json


class ProvenanceError(ValueError):
    """Existing artifacts cannot be safely reused for the requested experiment."""


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    files = sorted(p for base in ("src/echoxflow", "tasks") for p in (root / base).rglob("*.py"))
    files += [root / "tasks/video_diffusion/train.yaml", root / "uv.lock"]
    packages: dict[str, str | None] = {}
    for name in ("torch", "numpy", "zarr", "torchvision", "timm", "open_clip_torch", "torch-fidelity", "h5py"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    external = {}
    for variable in ("PANECHO_SOURCE", "ECHOJEPA_SOURCE"):
        location = os.environ.get(variable)
        if location:
            directory = Path(location).expanduser().resolve()
            external[variable] = {
                str(p.relative_to(directory)): file_digest(p) for p in sorted(directory.rglob("*.py"))
            }
    return {
        "external_source": external,
        "source": {str(p.relative_to(root)): file_digest(p) for p in files},
        "packages": packages,
    }


def inventory(root: Path) -> str:
    """Detect ordinary edits to immutable payloads without reading the entire corpus.

    Content hashes protect metadata; payload size/mtime protect local stores.
    This is not a substitute for publisher checksums against deliberate tampering.
    """
    digest = hashlib.sha256()
    if not root.exists():
        raise FileNotFoundError(root)
    paths = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    for path in paths:
        stat = path.stat()
        row = [str(path.relative_to(root)) if path != root else ".", stat.st_size, stat.st_mtime_ns]
        digest.update(json.dumps(row).encode())
        if stat.st_size <= 1024 * 1024:
            digest.update(file_digest(path).encode())
    return digest.hexdigest()


def dataset_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    data = config["data"]
    root = data_root(data.get("root_dir")).resolve()
    split = Path(data.get("split_csv") or root / "splits.csv").expanduser().resolve()
    views = Path(data.get("view_labels_csv") or "view_classifier_probabilities.csv").expanduser()
    if not views.is_absolute():
        views = root / views
    return {
        "root": str(root),
        "catalog": file_digest(root / "croissant.json"),
        "splits": file_digest(split),
        "views": file_digest(views) if views.is_file() else None,
        "payload_inventory": inventory(root / "exams"),
    }


def bind_output(directory: Path, identity: Mapping[str, Any], *, artifacts: tuple[str, ...]) -> None:
    """Record identity before work starts; never adopt unverified legacy artifacts."""
    path = directory / "provenance.json"
    # JSON normalization makes Path/tuple representation stable on the next invocation.
    expected = json.loads(json.dumps(identity, sort_keys=True, default=str))
    if path.exists():
        try:
            actual = json.loads(path.read_text())
        except (OSError, ValueError) as error:
            raise ProvenanceError(f"Unreadable provenance in {directory}; use a new output directory") from error
        if actual != expected:
            raise ProvenanceError(f"Experiment identity changed in {directory}; use a new output directory")
    elif any((directory / name).exists() for name in artifacts):
        raise ProvenanceError(f"Artifacts lack provenance in {directory}; use a new output directory")
    else:
        directory.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, expected)
