"""Write the provenance manifest of one benchmark run into the benchmark output root.

The manifest is rewritten atomically on every launch; it keeps the first creation time and the config hashes of
earlier launches (``config_hash_history``). Whether existing cells still match the configuration is checked per cell
by the launcher, from the resolved config hash every cell records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tasks.utils.atomic import atomic_write_json  # noqa: E402

#: Config files whose content defines the benchmark; their hash identifies the protocol of a run.
CONFIG_FILES = (
    "configs/benchmark_methods.yaml",
    "configs/benchmark_table.yaml",
    "configs/protocols/echoxflow_paper_v1.yaml",
    "configs/bench_smoke/color_doppler.yaml",
    "configs/bench_smoke/segmentation.yaml",
    "configs/bench_smoke/tissue_doppler.yaml",
    "tasks/color_doppler/train.yaml",
    "tasks/segmentation/train.yaml",
    "tasks/tissue_doppler/train.yaml",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split-csv", default=None)
    parser.add_argument("cell_args", nargs=argparse.REMAINDER, help="Arguments passed to every tasks.bench cell.")
    args = parser.parse_args(argv)

    repo = REPO_ROOT
    cell_args = [arg for arg in args.cell_args if arg != "--"]
    output = Path(args.output_dir) / f"{args.run_name}.manifest.json"
    previous = _read_manifest(output)
    now = datetime.now(UTC).isoformat(timespec="seconds")
    manifest = {
        "run_name": args.run_name,
        "created_utc": (previous or {}).get("created_utc") or now,
        "updated_utc": now,
        "git": _git_state(repo),
        "data_root": None if args.data_root is None else str(Path(args.data_root).expanduser().resolve()),
        "fold_split_path": args.split_csv,
        "fold_split_sha256": _file_sha256(args.split_csv),
        "dataset": _dataset_info(args.data_root),
        "cell_args": cell_args,
        "config_files": {name: _file_sha256(str(repo / name)) for name in CONFIG_FILES},
        "config_hash": _config_hash(repo, cell_args),
        "python": sys.version.split()[0],
    }
    history = list((previous or {}).get("config_hash_history") or [])
    if previous is not None and previous.get("config_hash") and previous.get("config_hash") not in history:
        history.append(previous["config_hash"])
    if manifest["config_hash"] not in history:
        history.append(manifest["config_hash"])
    manifest["config_hash_history"] = history
    atomic_write_json(output, manifest)
    print(output)
    return 0


def _read_manifest(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _git_state(repo: Path) -> dict[str, Any]:
    def git(*command: str) -> str | None:
        try:
            return subprocess.check_output(("git", *command), cwd=repo, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git("status", "--porcelain", "--untracked-files=no")
    return {"sha": git("rev-parse", "HEAD"), "dirty": None if status is None else bool(status)}


def _dataset_info(data_root: str | None) -> dict[str, Any]:
    if data_root is None:
        return {}
    root = Path(data_root).expanduser()
    info: dict[str, Any] = {}
    croissant = root / "croissant.json"
    if croissant.is_file():
        payload = json.loads(croissant.read_text(encoding="utf-8"))
        info["croissant_sha256"] = _file_sha256(str(croissant))
        info["croissant_name"] = payload.get("name")
        info["croissant_version"] = payload.get("version")
        info["croissant_conforms_to"] = payload.get("conformsTo")
    info["store_schema_version"] = _store_schema_version(root)
    return info


def _store_schema_version(root: Path) -> str | None:
    """Schema version declared by the first recording store found."""
    for attrs_path in sorted((root / "exams").glob("*/*.zarr/.zattrs"))[:1]:
        attrs = json.loads(attrs_path.read_text(encoding="utf-8"))
        value = attrs.get("schema_version")
        return str(value) if value else None
    for zarr_json in sorted((root / "exams").glob("*/*.zarr/zarr.json"))[:1]:
        value = json.loads(zarr_json.read_text(encoding="utf-8")).get("attributes", {}).get("schema_version")
        return str(value) if value else None
    return None


def _config_hash(repo: Path, cell_args: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for name in CONFIG_FILES:
        path = repo / name
        digest.update(name.encode())
        digest.update(path.read_bytes() if path.is_file() else b"<missing>")
    digest.update(json.dumps(list(cell_args)).encode())
    return digest.hexdigest()


def _file_sha256(path: str | None) -> str | None:
    if path is None or not Path(path).expanduser().is_file():
        return None
    return hashlib.sha256(Path(path).expanduser().read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
