"""Shared color and colormap helpers for EchoXFlow plotting."""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence, cast

import numpy as np
import yaml

if TYPE_CHECKING:
    from matplotlib.colors import ListedColormap

COLORS_CONFIG_ENV = "ECHOXFLOW_COLORS_CONFIG"


@dataclass(frozen=True)
class ColormapStop:
    position: float
    color: tuple[int, int, int]


@dataclass(frozen=True)
class NamedColormapConfig:
    stops: tuple[ColormapStop, ...] = ()
    colors: tuple[tuple[int, int, int], ...] = ()
    alias: str | None = None


@dataclass(frozen=True)
class Colors:
    theme: Mapping[str, tuple[int, int, int]]
    doppler: Mapping[str, tuple[int, int, int]]
    ecg: Mapping[str, tuple[int, int, int]]
    plot: Mapping[str, tuple[int, int, int]]
    categorical: Mapping[str, tuple[tuple[int, int, int], ...] | tuple[int, int, int]]
    data_modalities: Mapping[str, str]
    colormaps: Mapping[str, NamedColormapConfig]


_COLORS: Colors | None = None


def _rgb_to_tuple(value: Sequence[int]) -> tuple[int, int, int]:
    if len(value) != 3:
        raise ValueError(f"Expected RGB triple, got {value!r}")
    return int(value[0]), int(value[1]), int(value[2])


def _parse_rgb_mapping(raw: Mapping[str, Any]) -> dict[str, tuple[int, int, int]]:
    return {key: _rgb_to_tuple(cast(Sequence[int], value)) for key, value in raw.items()}


def _parse_categorical(raw: Mapping[str, Any]) -> dict[str, tuple[tuple[int, int, int], ...] | tuple[int, int, int]]:
    parsed: dict[str, tuple[tuple[int, int, int], ...] | tuple[int, int, int]] = {}
    for key, value in raw.items():
        if isinstance(value, list) and value and isinstance(value[0], list):
            parsed[key] = tuple(_rgb_to_tuple(cast(Sequence[int], item)) for item in value)
        else:
            parsed[key] = _rgb_to_tuple(cast(Sequence[int], value))
    return parsed


def _parse_colormap_stop(raw: Mapping[str, Any]) -> ColormapStop:
    position = float(raw["position"])
    if not np.isfinite(position) or position < 0.0 or position > 1.0:
        raise ValueError(f"Invalid colormap stop position: {raw!r}")
    return ColormapStop(position=position, color=_rgb_to_tuple(cast(Sequence[int], raw["color"])))


def _parse_named_colormap(raw: Mapping[str, Any]) -> NamedColormapConfig:
    alias = raw.get("alias")
    if isinstance(alias, str) and alias.strip():
        return NamedColormapConfig(alias=alias)
    colors = raw.get("colors")
    if isinstance(colors, list):
        parsed_colors = tuple(_rgb_to_tuple(cast(Sequence[int], color)) for color in colors)
        if len(parsed_colors) < 2:
            raise ValueError("Named colormap color table requires at least two colors")
        return NamedColormapConfig(colors=parsed_colors)
    stops = tuple(_parse_colormap_stop(cast(Mapping[str, Any], stop)) for stop in raw.get("stops", []))
    if len(stops) < 2:
        raise ValueError("Named colormap stops require at least two entries")
    positions = [stop.position for stop in stops]
    if positions != sorted(positions):
        raise ValueError("Named colormap stop positions must be sorted")
    return NamedColormapConfig(stops=stops)


def _parse_colormaps(raw: Mapping[str, Any]) -> dict[str, NamedColormapConfig]:
    return {name: _parse_named_colormap(cast(Mapping[str, Any], spec)) for name, spec in raw.items()}


def _parse_data_modalities(raw: Mapping[str, Any]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for modality, colormap in raw.items():
        if not isinstance(colormap, str) or not colormap.strip():
            raise ValueError(f"Invalid colormap mapping for data modality `{modality}`")
        parsed[_normalize_data_modality(str(modality))] = colormap.strip()
    return parsed


def _config_path(path: str | Path | None) -> Path | None:
    if path is not None:
        return Path(path).expanduser()
    override = os.environ.get(COLORS_CONFIG_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return None


def _load_yaml(path: str | Path | None) -> dict[str, Any]:
    config_path = _config_path(path)
    if config_path is None:
        raw = yaml.safe_load(files("echoxflow.config").joinpath("colors.yml").read_text(encoding="utf-8")) or {}
    else:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        source = "packaged colors" if config_path is None else str(config_path)
        raise ValueError(f"Invalid EchoXFlow colors config format: {source}")
    return raw


def _section(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid `{key}` section in EchoXFlow colors config")
    return value


def load_colors(path: str | Path | None = None) -> Colors:
    """Load shared EchoXFlow colors and named colormaps."""
    raw = _load_yaml(path)
    return Colors(
        theme=_parse_rgb_mapping(_section(raw, "theme")),
        doppler=_parse_rgb_mapping(_section(raw, "doppler")),
        ecg=_parse_rgb_mapping(_section(raw, "ecg")),
        plot=_parse_rgb_mapping(_section(raw, "plot")),
        categorical=_parse_categorical(_section(raw, "categorical")),
        data_modalities=_parse_data_modalities(_section(raw, "data_modalities")),
        colormaps=_parse_colormaps(_section(raw, "colormaps")),
    )


def get_colors() -> Colors:
    """Return the cached packaged colors config."""
    global _COLORS
    if _COLORS is None:
        _COLORS = load_colors()
    return _COLORS


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#" + "".join(f"{int(channel):02x}" for channel in rgb)


def categorical_hex_cycle(name: str = "okabe_ito") -> tuple[str, ...]:
    palette = get_colors().categorical.get(name)
    if not palette or not isinstance(palette[0], tuple):
        raise ValueError(f"Categorical palette `{name}` is missing or invalid")
    return tuple(rgb_to_hex(cast(tuple[int, int, int], color)) for color in palette)


def categorical_palette(name: str = "okabe_ito", *, max_colors: int | None = None) -> np.ndarray:
    palette = get_colors().categorical.get(name)
    if not palette or not isinstance(palette[0], tuple):
        raise ValueError(f"Categorical palette `{name}` is missing or invalid")
    colors = np.asarray(tuple(cast(tuple[int, int, int], color) for color in palette), dtype=np.float32) / 255.0
    return colors if max_colors is None else colors[: int(max_colors)]


def categorical_hex_color(index: int, *, name: str = "okabe_ito") -> str:
    palette = categorical_hex_cycle(name)
    if not palette:
        raise ValueError(f"Categorical palette `{name}` is empty")
    return palette[int(index) % len(palette)]


def neutral_categorical_hex() -> str:
    neutral = get_colors().categorical.get("neutral_unlabeled")
    if not isinstance(neutral, tuple) or len(neutral) != 3 or isinstance(neutral[0], tuple):
        raise ValueError("Categorical color `neutral_unlabeled` is missing or invalid")
    return rgb_to_hex(neutral)


def data_modality_colormap_name(name: str) -> str | None:
    """Return the configured colormap name for a Croissant data modality."""
    colors = get_colors()
    normalized = _normalize_data_modality(name)
    return colors.data_modalities.get(normalized) or colors.data_modalities.get(str(name))


def _resolve_colormap_spec(name: str, *, seen: set[str] | None = None) -> NamedColormapConfig | None:
    colormaps = get_colors().colormaps
    resolved_name = data_modality_colormap_name(name) or str(name)
    spec = colormaps.get(resolved_name)
    if spec is None and not resolved_name.startswith("data/"):
        spec = colormaps.get(f"data/{resolved_name}")
    if spec is None or spec.alias is None:
        return spec
    seen = set() if seen is None else seen
    if resolved_name in seen:
        raise ValueError(f"Colormap alias cycle detected at `{resolved_name}`")
    seen.add(resolved_name)
    return _resolve_colormap_spec(spec.alias, seen=seen)


def named_listed_colormap(name: str, *, size: int = 256) -> ListedColormap | None:
    """Return a configured Matplotlib listed colormap by name."""
    from matplotlib.colors import ListedColormap

    spec = _resolve_colormap_spec(str(name))
    if spec is None:
        return None
    if spec.colors:
        table = np.asarray(spec.colors, dtype=np.float32) / 255.0
        return ListedColormap(table, name=str(name))
    positions = np.asarray([stop.position for stop in spec.stops], dtype=np.float32)
    rgb = np.asarray([stop.color for stop in spec.stops], dtype=np.float32) / 255.0
    samples = np.linspace(0.0, 1.0, max(2, int(size)), dtype=np.float32)
    table = _interpolate_colormap_table(samples, positions, rgb)
    return ListedColormap(table, name=str(name))


def _interpolate_colormap_table(samples: np.ndarray, positions: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    table = np.empty((samples.size, 3), dtype=np.float32)
    for index, sample in enumerate(samples):
        left = int(np.searchsorted(positions, sample, side="right") - 1)
        left = int(np.clip(left, 0, positions.size - 1))
        if left >= positions.size - 1:
            table[index] = rgb[-1]
            continue
        pos0 = float(positions[left])
        pos1 = float(positions[left + 1])
        if pos1 <= pos0:
            table[index] = rgb[left]
            continue
        frac = (float(sample) - pos0) / (pos1 - pos0)
        table[index] = rgb[left] + float(np.clip(frac, 0.0, 1.0)) * (rgb[left + 1] - rgb[left])
    return table


def _normalize_data_modality(name: str) -> str:
    text = str(name).strip("/")
    return text if text.startswith("data/") else f"data/{text}"
