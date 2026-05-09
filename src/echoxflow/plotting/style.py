"""Plot styling derived from EchoXFlow color configuration."""

from __future__ import annotations

from dataclasses import dataclass

from echoxflow.colors import get_colors, rgb_to_hex


@dataclass(frozen=True)
class PlotStyle:
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
    show_ecg: bool = True
    axis_tick_label_fontsize: float = 14.0
    colorbar_fontsize: float = 16.0
    color_doppler_extent_color: str = "#FFFFFF"
    color_doppler_extent_linewidth: float = 1.2
    show_color_doppler_extent: bool = True
    annotation_color: str = "#00D5FF"
    annotation_secondary_color: str = "#FF9F0A"
    annotation_edge_color: str = "#000000"
    annotation_linewidth: float = 1.1
    annotation_markersize: float = 4.0
    annotation_label_fontsize: float = 10.0
    sampling_gate_color: str = "#4AC44A"
    sampling_gate_linewidth: float = 2.0
    sampling_gate_markersize: float = 9.0
    show_clinical_depth_ruler: bool = True
    clinical_depth_ruler_side: str = "left"
    clinical_depth_ruler_tick_interval_cm: float = 1.0
    clinical_depth_ruler_label_interval_cm: float = 5.0
    clinical_depth_ruler_minor_tick_length_cm: float = 0.22
    clinical_depth_ruler_major_tick_length_cm: float = 0.34
    clinical_depth_ruler_label_pad_cm: float = 0.12
    clinical_depth_ruler_linewidth: float = 0.75
    clinical_depth_ruler_border_linewidth: float = 0.65
    clinical_depth_ruler_label_fontsize: float = 14.0
    clinical_depth_ruler_show_border: bool = False

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
