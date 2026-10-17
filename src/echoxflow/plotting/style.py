"""Plot styling derived from EchoXFlow color configuration."""

from __future__ import annotations

from dataclasses import dataclass

from echoxflow.colors import get_colors, rgb_to_hex
from echoxflow.plotting.gating import BloodGateConfig, TissueGateConfig


@dataclass(frozen=True)
class PlotStyle:
    """Size, colours and fonts of a rendered figure.

    `width_px` and `height_px` are the output pixel size of every still and video frame, whatever `dpi` is; `dpi`
    only scales fonts and line widths. The colour fields set the figure, panel, text, ECG and cursor colours. The
    font-size fields set tick, colour bar, panel label and annotation text. The `show_*` fields switch the ECG
    strip, panel labels, figure title, Doppler extent box and depth ruler on or off. The `annotation_*`,
    `sampling_gate_*` and `cartesian_depth_ruler_*` fields style overlays and the depth ruler. `max_fps` caps the
    frame rate of videos. `strip_height_ratio` is the height of a spectral or M-mode strip relative to an image row;
    `strip_height_ratio_static` replaces it when every image has a single frame, so the strip dominates.
    `volume_curve_height_ratio` sets the LV volume curve height relative to an image row.
    """

    width_px: int = 1440
    height_px: int = 1080
    dpi: int = 100
    figure_facecolor: str = "#222222"
    panel_facecolor: str = "#222222"
    text_color: str = "#E6E6E6"
    text_dim_color: str = "#787878"
    grid_color: str = "#303030"
    ecg_trace_color: str = "#4AC44A"
    ecg_marker_color: str = "#FF6060"
    cursor_color: str = "#FF6060"
    line_color: str = "#3CFF3C"
    max_fps: float = 60.0
    strip_height_ratio: float = 0.55
    strip_height_ratio_static: float = 1.6
    volume_curve_height_ratio: float = 0.3
    show_ecg: bool = True
    axis_tick_label_fontsize: float = 14.0
    colorbar_fontsize: float = 16.0
    panel_label_fontsize: float = 11.0
    show_panel_labels: bool = True
    show_figure_title: bool = True
    color_doppler_extent_color: str = "#FFFFFF"
    color_doppler_extent_linewidth: float = 1.2
    show_color_doppler_extent: bool = True
    blood_gate: BloodGateConfig | None = None  # colour Doppler gate; None uses the BloodGateConfig defaults
    tissue_gate: TissueGateConfig | None = None  # tissue Doppler gate; None uses the TissueGateConfig defaults
    show_colorbars: bool = True
    show_colorbar_labels: bool = True
    cartesian_max_depth_m: float | None = None  # crop scan-converted sectors below this depth
    tissue_doppler_brightness_lift: float = (
        0.0  # 0-1: moves the B-mode brightness picking the tissue colour row towards 1
    )
    annotation_color: str = "#00D5FF"
    annotation_secondary_color: str = "#FF9F0A"
    annotation_edge_color: str = "#000000"
    annotation_linewidth: float = 1.1
    annotation_markersize: float = 4.0
    annotation_label_fontsize: float = 10.0
    sampling_gate_color: str = "#4AC44A"
    sampling_gate_linewidth: float = 2.0
    sampling_gate_markersize: float = 9.0
    show_cartesian_depth_ruler: bool = True
    cartesian_depth_ruler_side: str = "left"
    cartesian_depth_ruler_tick_interval_cm: float = 1.0
    cartesian_depth_ruler_label_interval_cm: float = 5.0
    cartesian_depth_ruler_minor_tick_length_cm: float = 0.22
    cartesian_depth_ruler_major_tick_length_cm: float = 0.34
    cartesian_depth_ruler_label_pad_cm: float = 0.12
    cartesian_depth_ruler_linewidth: float = 0.75
    cartesian_depth_ruler_border_linewidth: float = 0.65
    cartesian_depth_ruler_label_fontsize: float = 14.0
    cartesian_depth_ruler_show_border: bool = False
    show_mosaic_levels: bool = True
    mosaic_level_colors: tuple[str, str, str] = ("#FFD23F", "#FF6F91", "#8AFF80")

    @property
    def figsize(self) -> tuple[float, float]:
        return self.width_px / self.dpi, self.height_px / self.dpi

    @classmethod
    def from_config(cls) -> "PlotStyle":
        colors = get_colors()
        return cls(
            figure_facecolor=rgb_to_hex(colors.theme["background_dark"]),
            panel_facecolor=rgb_to_hex(colors.theme["panel_bg"]),
            text_color=rgb_to_hex(colors.theme["text_primary"]),
            text_dim_color=rgb_to_hex(colors.theme["text_dim"]),
            grid_color=rgb_to_hex(colors.ecg["grid"]),
            ecg_trace_color=rgb_to_hex(colors.ecg["trace"]),
            ecg_marker_color=rgb_to_hex(colors.ecg["marker"]),
            cursor_color=rgb_to_hex(colors.ecg["marker"]),
            line_color=rgb_to_hex(colors.doppler["ruler"]),
            sampling_gate_color=rgb_to_hex(colors.ecg["trace"]),
        )
