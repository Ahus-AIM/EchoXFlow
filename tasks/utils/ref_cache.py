"""Per-recording cache of dataset scan results (discovery verdicts and sample references).

Building a benchmark cell opens every candidate recording to decide whether it is usable and which clips it yields.
Those answers depend only on the recording and on a few sampling settings, so they are cached per recording and
shared by every fold, split, method and domain: a cell assembles its sample lists from the cached per-recording
results, in the same order, and gets exactly the lists an uncached scan builds.

An entry is keyed by

- the task namespace and a hash of the settings that affect the result (plus a schema version),
- the data root and the Croissant manifest's size and modification time (a new export invalidates everything),
- the recording's Croissant record and the size and modification time of its Zarr store metadata.

Entries are small JSON files written to a temporary file and renamed into place, so concurrent cells can build the
same entry: each rename installs a complete file, and identical duplicates are harmless. Unreadable entries are
recomputed. Within a process, results are also memoized, so for example a discovery verdict computed for the
training split is reused by the selection and validation splits.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
from collections.abc import Callable, Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, cast

from echoxflow import RecordingRecord
from echoxflow import data_root as resolve_data_root

DEFAULT_REF_CACHE_DIR = Path("outputs/cache/sample_refs")
CACHE_SCHEMA_VERSION = 1

ValueT = TypeVar("ValueT")
LOG = logging.getLogger(__name__)
_MEMO: dict[tuple[str, str], object] = {}


class RecordingScanCache:
    """Cache ``compute(record)`` results for one task namespace and one set of settings.

    ``settings`` must contain everything (besides the recording itself) that the cached result depends on; values
    must be JSON-serializable. With ``cache_dir=None`` results are only memoized in this process.
    """

    def __init__(
        self,
        cache_dir: str | Path | None,
        *,
        namespace: str,
        settings: Mapping[str, object],
        data_root: str | Path | None,
        read_only: bool = False,
    ) -> None:
        self.root = resolve_data_root(data_root) if data_root is None else Path(data_root).expanduser()
        self.read_only = bool(read_only)
        context = {
            "schema": CACHE_SCHEMA_VERSION,
            "namespace": str(namespace),
            "settings": _jsonable(settings),
            "data_root": str(self.root.resolve()),
            "croissant": _file_signature(self.root / "croissant.json"),
        }
        self.context_key = _digest(context)
        self.directory = (
            None if cache_dir is None else Path(cache_dir).expanduser() / str(namespace) / f"{self.context_key}"
        )
        self.hits = 0
        self.misses = 0

    def get(self, record: RecordingRecord, compute: Callable[[RecordingRecord], ValueT]) -> ValueT:
        """Return the cached result for ``record``, computing (and storing) it on a miss."""
        key = self._record_key(record)
        memo_key = (self.context_key, key)
        if memo_key in _MEMO:
            self.hits += 1
            return cast(ValueT, _MEMO[memo_key])
        path = None if self.directory is None else self.directory / key[:2] / f"{key}.json"
        if path is not None:
            cached = _read_entry(path, record=record)
            if cached is not _MISSING:
                self.hits += 1
                _MEMO[memo_key] = cached
                return cast(ValueT, cached)
        self.misses += 1
        value = compute(record)
        encoded = _jsonable(value)
        if path is not None and not self.read_only:
            _write_entry(path, record=record, value=encoded)
        # Memoize the JSON round trip, so a hit and a miss return equal values of the same types.
        _MEMO[memo_key] = encoded
        return cast(ValueT, encoded)

    def _record_key(self, record: RecordingRecord) -> str:
        return _digest(
            {
                "record": _record_identity(record),
                "store": _store_signature(record.path(self.root)),
            }
        )


_ROW_FIELDS = (
    "exam_id",
    "recording_id",
    "zarr_path",
    "modes",
    "content_types",
    "frame_counts_by_content_type",
    "median_delta_time_by_content_type",
    "array_paths",
)


def _record_identity(record: RecordingRecord) -> dict[str, Any]:
    """The record as keyed before views, folds and annotations existed, so those additions keep every entry valid.

    Records used to carry their Croissant row as ``raw``; on the published exports that row holds exactly the
    ``_ROW_FIELDS`` under ``recordings/`` keys, so it is rebuilt from them.
    """
    identity = {name: _jsonable(getattr(record, name)) for name in (*_ROW_FIELDS, "stitch_beat_count", "relationships")}
    identity["raw"] = {f"recordings/{name}": identity[name] for name in _ROW_FIELDS}
    return identity


class _Missing:
    pass


_MISSING = _Missing()


def _read_entry(path: Path, *, record: RecordingRecord) -> object:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _MISSING
    except (OSError, ValueError) as exc:
        LOG.warning("ignoring unreadable scan cache entry %s: %s", path, exc)
        return _MISSING
    if not isinstance(payload, dict) or payload.get("recording") != _record_label(record) or "value" not in payload:
        return _MISSING
    return payload["value"]


def _write_entry(path: Path, *, record: RecordingRecord, value: object) -> None:
    text = json.dumps({"recording": _record_label(record), "value": value}, sort_keys=True, separators=(",", ":"))
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    except OSError as exc:
        LOG.warning("could not write scan cache entry %s: %s", path, exc)
    finally:
        temporary.unlink(missing_ok=True)


def _record_label(record: RecordingRecord) -> str:
    return f"{record.exam_id}/{record.recording_id}"


def _store_signature(path: Path) -> list[object] | None:
    """Size and modification time of the store's metadata (consolidated metadata when present)."""
    if path.is_file():
        return _file_signature(path)
    for name in (".zmetadata", "zarr.json", ".zattrs"):
        signature = _file_signature(path / name)
        if signature is not None:
            return [name, *signature]
    return None


def _file_signature(path: Path) -> list[object] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return [int(stat.st_size), int(stat.st_mtime_ns)]


def _digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:32]


def _jsonable(value: object) -> Any:
    """The JSON form of ``value`` (tuples become lists), so cached and fresh results compare and behave alike."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        return _jsonable(value.item())
    raise TypeError(f"cannot cache a value of type {type(value).__name__}")


def scan_cache_dir(data_config: Mapping[str, object]) -> Path | None:
    """Scan cache directory of a data config: ``data.sample_ref_cache_dir``; ``sample_ref_cache: false`` disables it."""
    enabled = data_config.get("sample_ref_cache", True)
    if isinstance(enabled, str):
        enabled = enabled.strip().lower() not in {"", "0", "false", "no", "off", "none"}
    if not enabled:
        return None
    configured = data_config.get("sample_ref_cache_dir")
    if configured is None or str(configured).strip() == "":
        return DEFAULT_REF_CACHE_DIR
    return Path(str(configured)).expanduser()


__all__ = ["DEFAULT_REF_CACHE_DIR", "RecordingScanCache", "scan_cache_dir"]
