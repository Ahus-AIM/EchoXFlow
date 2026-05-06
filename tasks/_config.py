from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class TrainingConfig:
    raw: dict[str, Any]

    @property
    def model(self) -> dict[str, Any]:
        return self.section("model")

    @property
    def optimizer(self) -> dict[str, Any]:
        return self.section("optimizer")

    @property
    def trainer(self) -> dict[str, Any]:
        return self.section("trainer")

    @property
    def data(self) -> dict[str, Any]:
        return self.section("data")

    @property
    def loss(self) -> dict[str, Any]:
        return self.section("loss")

    @property
    def metrics(self) -> dict[str, Any]:
        return self.section("metrics")

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name, {})
        if not isinstance(value, Mapping):
            raise TypeError(name)
        return {str(key): _normalize(item) for key, item in value.items()}

    def __getattr__(self, name: str) -> Any:
        for section in self.raw.values():
            if isinstance(section, Mapping) and name in section:
                return _normalize(section[name])
        raise AttributeError(name)


def load_training_config(path: str | Path | None, *, default_path: Path) -> TrainingConfig:
    config_path = Path(path) if path is not None else default_path
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"Expected YAML mapping in {config_path}")
    return TrainingConfig(raw=raw)


def load_task_training_config(path: str | Path | None, *, module_file: str) -> TrainingConfig:
    return load_training_config(path, default_path=Path(module_file).resolve().parents[1] / "train.yaml")


def _normalize(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_normalize(item) for item in value)
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items()}
    return value
