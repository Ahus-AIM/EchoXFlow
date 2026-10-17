"""Reusable fixed-position colorbars for modality frame renderers."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize

from echoxflow.colors import data_modality_colormap_name
from echoxflow.plotting.style import PlotStyle
from echoxflow.schema import color_flow_velocity_codes

# The scanner's bar: 30 x 184 px in a 1016 x 708 frame, top at 5.9 % of the height, right edge at 98.8 % of the width.
_SCANNER_BAR_AXES = (0.988 - 30.0 / 1016.0, 1.0 - 0.059 - 184.0 / 708.0, 30.0 / 1016.0, 184.0 / 708.0)
_SCANNER_BAR_SIZE_PX = (184, 30)


@dataclass(frozen=True)
class FrameColorbarSpec:
    """Fixed top-right colorbar inside a rendered modality panel.

    `label` names the quantity and its unit for callers; the drawn bar has no title, like the scanner, and shows only
    the upper limit above the bar and the lower limit below it.

    With a `table` (the recording's `(256, 256, 3)` FlowColorTable) the bar is the scanner's two-dimensional one:
    velocity runs down the bar through the table's phase code and the table's first axis across it (`table_kind`
    `"flow"` for turbulence, `"tissue"` for brightness). Without one it is the one-dimensional colormap ramp.
    """

    label: str
    value_range: tuple[float, float] = (-1.0, 1.0)
    ticks: tuple[float, ...] = (-1.0, 0.0, 1.0)
    tick_labels: tuple[str, ...] | None = None
    bbox_axes: tuple[float, float, float, float] = (0.95, 0.6, 0.03, 0.28)
    reverse_cmap: bool = True
    font_size: float = 8.0
    table: np.ndarray | None = field(default=None, compare=False)
    table_kind: str = "flow"


def colorbar_spec_for_modality(
    data_path: str,
    *,
    value_range: tuple[float, float] | None = None,
    table: np.ndarray | None = None,
) -> FrameColorbarSpec | None:
    """Return the default colorbar spec for a Croissant data modality.

    `table` is the recording's FlowColorTable (turbulence-style for colour flow, brightness-style for tissue); with
    it the spec describes the scanner's two-dimensional bar, limits in m/s.
    """
    cmap_name = data_modality_colormap_name(data_path)
    if table is not None and value_range is not None and cmap_name in {"color_doppler_velocity", "tissue_doppler"}:
        limit = abs(float(value_range[1]))
        return FrameColorbarSpec(
            label="Velocity (m/s)",
            value_range=(-limit, limit),
            ticks=(-limit, limit),
            tick_labels=(scanner_limit_label(-limit), scanner_limit_label(limit)),
            bbox_axes=_SCANNER_BAR_AXES,
            reverse_cmap=False,
            table=table,
            table_kind="tissue" if cmap_name == "tissue_doppler" else "flow",
        )
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
                reverse_cmap=False,
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


def scanner_limit_label(value: float) -> str:
    """Format a velocity limit like the scanner: two decimals, no leading zero (`.60`, `-.60`)."""
    text = f"{float(value):.2f}"
    if abs(float(value)) < 0.005:
        return ".00"
    return text.replace("0.", ".", 1) if text.startswith(("0.", "-0.")) else text


def scanner_colorbar_image(
    table: np.ndarray, *, kind: str, shape: tuple[int, int] = _SCANNER_BAR_SIZE_PX
) -> np.ndarray:
    """Return the scanner's two-dimensional colour bar as `uint8` `(rows, columns, 3)`.

    Velocity falls from `+limit` at the top to `-limit` at the bottom through the table's phase code; the table's
    first axis grows from left to right, `8 + 8.5 x` for colour flow turbulence and `54 + 7.8 x`, capped at 204,
    for tissue brightness (`x` is the column in the scanner's 30 px bar and is rescaled for other widths).
    """
    rows, columns = shape
    fraction = 1.0 - (np.arange(rows) + 0.5) * 2.0 / rows
    codes = color_flow_velocity_codes(fraction, velocity_limit_mps=1.0).astype(np.intp)
    x = (np.arange(columns) + 0.5) * (30.0 / columns) - 0.5
    k = np.minimum(54.0 + 7.8 * x, 204.0) if kind == "tissue" else 8.0 + 8.5 * x
    first = np.clip(np.rint(k), 0, 255).astype(np.intp)
    return np.asarray(np.asarray(table, dtype=np.uint8)[first[None, :], codes[:, None]], dtype=np.uint8)


def draw_top_right_colorbar(
    ax: Axes,
    cmap: Colormap,
    spec: FrameColorbarSpec,
    *,
    style: PlotStyle,
) -> None:
    """Draw a fixed inset colorbar in the top-right of an axes; nothing when ``style.show_colorbars`` is off."""
    if not style.show_colorbars:
        return
    if spec.table is not None:
        _draw_scanner_colorbar(ax, spec, style=style)
        return
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
    labels = spec.tick_labels if spec.tick_labels is not None else tuple(f"{tick:g}" for tick in spec.ticks)
    centre = 0.5 * (spec.value_range[0] + spec.value_range[1])
    cax.set_yticks([centre])  # a short unlabelled centre tick; the limits are written beyond the bar ends
    cax.set_yticklabels([])
    gap = 0.2 * font_size
    for text, y, va, offset in _limit_labels(labels, gap, style):
        cax.annotate(
            text,
            xy=(1.0, y),
            xytext=(0.0, offset),
            xycoords="axes fraction",
            textcoords="offset points",
            ha="right",
            va=va,
            fontsize=font_size,
            color=style.text_color,
        )
    cax.yaxis.set_ticks_position("left")
    cax.tick_params(axis="y", length=3, colors=style.text_color)
    cax.tick_params(axis="x", length=0)
    cax.patch.set_edgecolor("none")
    cax.patch.set_linewidth(0)
    for spine in cax.spines.values():
        spine.set_visible(False)


def _draw_scanner_colorbar(ax: Axes, spec: FrameColorbarSpec, *, style: PlotStyle) -> None:
    """Draw the two-dimensional bar with its limits left-aligned at the bar's left edge, above and below it."""
    font_size = float(style.colorbar_fontsize)
    assert spec.table is not None
    cax = ax.inset_axes(spec.bbox_axes, transform=ax.transAxes)
    cax.imshow(
        scanner_colorbar_image(spec.table, kind=spec.table_kind),
        aspect="auto",
        interpolation="nearest",
        extent=(0.0, 1.0, 0.0, 1.0),
    )
    cax.set_xlim(0.0, 1.0)
    cax.set_ylim(0.0, 1.0)
    cax.set_xticks([])
    cax.set_yticks([])
    labels = spec.tick_labels if spec.tick_labels is not None else tuple(f"{tick:g}" for tick in spec.ticks)
    gap = 0.2 * font_size
    for text, y, va, offset in _limit_labels(labels, gap, style):
        cax.annotate(
            text,
            xy=(0.0, y),
            xytext=(0.0, offset),
            xycoords="axes fraction",
            textcoords="offset points",
            ha="left",
            va=va,
            fontsize=font_size,
            color=style.text_color,
        )
    for spine in cax.spines.values():
        spine.set_visible(False)


def _limit_labels(labels: tuple[str, ...], gap: float, style: PlotStyle) -> tuple[tuple[str, float, str, float], ...]:
    """The upper and lower limit texts with their anchor, alignment and offset; none when labels are off."""
    if not style.show_colorbar_labels:
        return ()
    return ((labels[-1], 1.0, "bottom", gap), (labels[0], 0.0, "top", -gap))


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
