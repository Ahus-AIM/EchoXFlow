"""Matplotlib plotting utilities for EchoXFlow recordings."""

from echoxflow.plotting.api import (
    plot_recording,
    render_loaded_arrays_video,
    render_panel_video,
    render_recording_frame,
    render_recording_video,
    save_recording_plot,
)
from echoxflow.plotting.gating import (
    BloodGateConfig,
    TissueGateConfig,
    blood_gate,
    normalize_bmode_intensity,
    normalize_doppler_power,
    tissue_gate,
)
from echoxflow.plotting.overlay import blend_segmentation_rgb, normalized_bmode_rgb
from echoxflow.plotting.renderer import RecordingPlotRenderer
from echoxflow.plotting.specs import FrameRequest, PanelSpec, PlotViewMode, RenderedFrame, TraceSpec, VideoRequest
from echoxflow.plotting.style import PlotStyle
from echoxflow.plotting.writers import write_video

__all__ = [
    "FrameRequest",
    "BloodGateConfig",
    "PanelSpec",
    "PlotViewMode",
    "PlotStyle",
    "RecordingPlotRenderer",
    "RenderedFrame",
    "TissueGateConfig",
    "TraceSpec",
    "VideoRequest",
    "blood_gate",
    "blend_segmentation_rgb",
    "normalize_bmode_intensity",
    "normalized_bmode_rgb",
    "normalize_doppler_power",
    "plot_recording",
    "render_loaded_arrays_video",
    "render_panel_video",
    "render_recording_frame",
    "render_recording_video",
    "save_recording_plot",
    "tissue_gate",
    "write_video",
]
