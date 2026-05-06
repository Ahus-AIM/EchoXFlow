"""Reusable fixed-position colorbars for modality frame renderers."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize

from echoxflow.colors import data_modality_colormap_name
from echoxflow.plotting.style import PlotStyle


@dataclass(frozen=True)
class FrameColorbarSpec:
    """Fixed top-right colorbar inside a rendered modality panel."""

    label: str
    value_range: tuple[float, float] = (-1.0, 1.0)
    ticks: tuple[float, ...] = (-1.0, 0.0, 1.0)
    tick_labels: tuple[str, ...] | None = None
    bbox_axes: tuple[float, float, float, float] = (0.945, 0.66, 0.035, 0.32)
    reverse_cmap: bool = True
    font_size: float = 8.0


def colorbar_spec_for_modality(
    data_path: str,
    *,
    value_range: tuple[float, float] | None = None,
) -> FrameColorbarSpec | None:
    """Return the default colorbar spec for a Croissant data modality."""
    cmap_name = data_modality_colormap_name(data_path)
    if cmap_name == "tissue_doppler":
        if value_range is None:
            return None
        if value_range[0] < 0.0 < value_range[1]:
            display_range = (100.0 * float(value_range[0]), 100.0 * float(value_range[1]))
            display_ticks = _range_ticks(display_range)
            return FrameColorbarSpec(
                label="Velocity (cm/s)",
                value_range=display_range,
                ticks=display_ticks,
                tick_labels=_format_tick_labels(display_ticks, decimals=0),
            )
        ticks = _range_ticks(value_range)
        return FrameColorbarSpec(
            label="Encoded tissue velocity",
            value_range=value_range,
            ticks=ticks,
            tick_labels=_format_tick_labels(ticks, decimals=0),
        )
    if cmap_name == "color_doppler_velocity":
        if value_range is None:
            return None
        ticks = _range_ticks(value_range)
        return FrameColorbarSpec(
            label="Velocity (m/s)",
            value_range=value_range,
            ticks=ticks,
            tick_labels=_format_tick_labels(ticks, decimals=2),
            reverse_cmap=False,
        )
    if cmap_name == "color_doppler_power":
        return FrameColorbarSpec(
            label="Power (normalized)",
            value_range=value_range or (0.0, 1.0),
            ticks=_range_ticks(value_range or (0.0, 1.0)),
        )
    return None


def draw_top_right_colorbar(
    ax: Axes,
    cmap: Colormap,
    spec: FrameColorbarSpec,
    *,
    style: PlotStyle,
) -> None:
    """Draw a fixed inset colorbar in the top-right of an axes."""
    font_size = float(style.colorbar_fontsize)
    cbar_cmap = _copy_cmap(cmap)
    if spec.reverse_cmap:
        cbar_cmap = cbar_cmap.reversed()
    cax = ax.inset_axes(spec.bbox_axes, transform=ax.transAxes)
    cax.set_facecolor(style.panel_facecolor)
    norm = Normalize(vmin=spec.value_range[0], vmax=spec.value_range[1])
    gradient = np.linspace(spec.value_range[0], spec.value_range[1], 256, dtype=np.float32)[:, None]
    cax.imshow(
        gradient,
        cmap=cbar_cmap,
        norm=norm,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=(0.0, 1.0, spec.value_range[0], spec.value_range[1]),
    )
    cax.set_xlim(0.0, 1.0)
    cax.set_ylim(spec.value_range)
    cax.set_xticks([])
    cax.set_yticks(list(spec.ticks))
    if spec.tick_labels is not None:
        cax.set_yticklabels(list(spec.tick_labels))
    cax.set_ylabel(spec.label, fontsize=font_size, labelpad=3, color=style.text_color)
    cax.yaxis.set_label_position("left")
    cax.yaxis.set_ticks_position("left")
    cax.tick_params(axis="y", labelsize=font_size, length=2, colors=style.text_color)
    cax.tick_params(axis="x", length=0)
    cax.patch.set_edgecolor("none")
    cax.patch.set_linewidth(0)
    for spine in cax.spines.values():
        spine.set_visible(False)


def _copy_cmap(cmap: Colormap) -> Colormap:
    return copy.copy(cmap)


def _range_ticks(value_range: tuple[float, float]) -> tuple[float, float, float]:
    vmin, vmax = value_range
    return float(vmin), float((vmin + vmax) * 0.5), float(vmax)


def _format_tick_labels(ticks: tuple[float, ...], *, decimals: int) -> tuple[str, ...]:
    return tuple(_format_tick(tick, decimals=decimals) for tick in ticks)


def _format_tick(value: float, *, decimals: int) -> str:
    if decimals <= 0:
        return f"{float(value):.0f}"
    if abs(float(value)) < 0.5 * 10.0 ** (-int(decimals)):
        return f"{0.0:.{int(decimals)}f}"
    text = f"{float(value):.{int(decimals)}f}"
    return text
