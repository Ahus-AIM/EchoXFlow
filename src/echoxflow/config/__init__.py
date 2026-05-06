"""Configuration helpers for EchoXFlow."""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULTS_CONFIG_ENV = "ECHOXFLOW_DEFAULTS_CONFIG"
DATA_ROOT_ENV = "ECHOXFLOW_DATA_ROOT"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    version: str


@dataclass(frozen=True)
class DataConfig:
    root: Path


@dataclass(frozen=True)
class OutputConfig:
    root: Path
    temp: Path


@dataclass(frozen=True)
class EchoXFlowSettings:
    dataset: DatasetConfig
    data: DataConfig
    output: OutputConfig
    raw: Mapping[str, Any]


def _default_config_text() -> str:
    return files("echoxflow.config").joinpath("defaults.yml").read_text(encoding="utf-8")


def _config_path(path: str | Path | None) -> Path | None:
    if path is not None:
        return Path(path).expanduser()
    override = os.environ.get(DEFAULTS_CONFIG_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return None


def _load_yaml(path: str | Path | None) -> tuple[dict[str, Any], Path | None]:
    config_path = _config_path(path)
    if config_path is None:
        raw = yaml.safe_load(_default_config_text()) or {}
        origin = None
    else:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        origin = config_path.parent
    if not isinstance(raw, dict):
        source = "packaged defaults" if config_path is None else str(config_path)
        raise ValueError(f"Invalid EchoXFlow defaults config format: {source}")
    return raw, origin


def _mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid `{key}` section in EchoXFlow defaults config")
    return value


def _string(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or invalid `{key}` value in EchoXFlow defaults config")
    return value


def _path(raw: str, *, base_dir: Path | None) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    if base_dir is not None:
        return base_dir / path
    return Path.cwd() / path


def load_settings(path: str | Path | None = None) -> EchoXFlowSettings:
    """Load EchoXFlow settings from packaged defaults or a YAML override."""
    raw, base_dir = _load_yaml(path)
    dataset = _mapping(raw, "dataset")
    data = _mapping(raw, "data")
    output = _mapping(raw, "output")
    return EchoXFlowSettings(
        dataset=DatasetConfig(name=_string(dataset, "name"), version=_string(dataset, "version")),
        data=DataConfig(root=_path(_string(data, "root"), base_dir=base_dir)),
        output=OutputConfig(
            root=_path(_string(output, "root"), base_dir=base_dir),
            temp=_path(_string(output, "temp"), base_dir=base_dir),
        ),
        raw=raw,
    )


def data_root(path: str | Path | None = None) -> Path:
    """Resolve the EchoXFlow data root.

    Precedence is explicit argument, then ECHOXFLOW_DATA_ROOT, then defaults YAML.
    """
    if path is not None:
        return Path(path).expanduser()
    override = os.environ.get(DATA_ROOT_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return load_settings().data.root


def resolve_data_path(*parts: str | Path, root: str | Path | None = None) -> Path:
    """Resolve a path under the configured EchoXFlow data root."""
    resolved = data_root(root)
    for part in parts:
        resolved /= Path(part)
    return resolved
