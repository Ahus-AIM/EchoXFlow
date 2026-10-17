"""Tools for reading and loading EchoXFlow echocardiographic data."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from echoxflow.aha_strain import AHAStrain, AHAStrainContributor, AHAStrainSegment, load_aha_strain
from echoxflow.config import data_root
from echoxflow.croissant import (
    CroissantArrayRecord,
    CroissantCatalog,
    RecordingRecord,
    RecordingRelationship,
    StreamSummary,
    find_derived_recordings,
    find_linked_recordings,
    find_recordings,
    find_source_recordings,
    load_croissant,
    load_exam_folds,
)
from echoxflow.export import (
    RecordingArray,
    upsert_croissant_recording,
    write_recording,
)
from echoxflow.loading import (
    LoadedArray,
    MeshFrame,
    PackedMeshAnnotation,
    RecordingStore,
    StreamHeader,
    open_recording,
)
from echoxflow.objects import (
    AnnotationRef,
    ArrayRef,
    LinkedArray,
    MeshSequenceRef,
    RecordingObject,
    RecordingObjectKind,
    RecordingRef,
    StrainPanel,
    load_recording_object,
)
from echoxflow.schema import ArrayInfo
from echoxflow.spectral import SpectralMetadata
from echoxflow.strain import SegmentalStrain, SegmentalStrainPanel, SourceStrainMetadata, load_segmental_strain
from echoxflow.streams import (
    AnnotationMaskStream,
    BrightnessModeStream,
    ColorDopplerPowerStream,
    ColorDopplerVelocityStream,
    EcgStream,
    EchoStream,
    FloatImageStream,
    GenericStream,
    MeshFaceStream,
    MeshFrameOffsetsStream,
    MeshPointStream,
    SpectralDopplerStream,
    StreamMetadata,
    TissueDopplerFloatStream,
    TissueDopplerRawStream,
)

_LAZY_EXPORTS = {
    "CartesianGrid": ("echoxflow.scan.geometry", "CartesianGrid"),
    "CartesianImage": ("echoxflow.scan.conversion", "CartesianImage"),
    "ImageLayer": ("echoxflow.scan.layers", "ImageLayer"),
    "SectorGeometry": ("echoxflow.scan.geometry", "SectorGeometry"),
    "SlicePlane": ("echoxflow.scan.geometry", "SlicePlane"),
    "VolumeGrid": ("echoxflow.scan.geometry", "VolumeGrid"),
    "VolumeSlice": ("echoxflow.scan.conversion", "VolumeSlice"),
    "PlotStyle": ("echoxflow.plotting", "PlotStyle"),
    "RecordingPlotRenderer": ("echoxflow.plotting", "RecordingPlotRenderer"),
    "compose_layers": ("echoxflow.scan.layers", "compose_layers"),
    "opacity_from_values": ("echoxflow.scan.layers", "opacity_from_values"),
    "plot_recording": ("echoxflow.plotting", "plot_recording"),
    "rasterize_beamspace_mask": ("echoxflow.scan.rasterization", "rasterize_beamspace_mask"),
    "rasterize_beamspace_volume_mask": ("echoxflow.scan.rasterization", "rasterize_beamspace_volume_mask"),
    "rasterize_packed_mesh_volume_masks": ("echoxflow.scan.rasterization", "rasterize_packed_mesh_volume_masks"),
    "render_recording_frame": ("echoxflow.plotting", "render_recording_frame"),
    "render_recording_video": ("echoxflow.plotting", "render_recording_video"),
    "resample_sector": ("echoxflow.scan.resampling", "resample_sector"),
    "resample_sector_stack": ("echoxflow.scan.resampling", "resample_sector_stack"),
    "save_recording_plot": ("echoxflow.plotting", "save_recording_plot"),
    "sector_lookup": ("echoxflow.scan.resampling", "sector_lookup"),
    "sector_to_cartesian": ("echoxflow.scan.conversion", "sector_to_cartesian"),
    "slice_volume": ("echoxflow.scan.conversion", "slice_volume"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


__all__ = [
    "AHAStrain",
    "AHAStrainContributor",
    "AHAStrainSegment",
    "AnnotationMaskStream",
    "AnnotationRef",
    "ArrayInfo",
    "ArrayRef",
    "BrightnessModeStream",
    "CartesianGrid",
    "CartesianImage",
    "ColorDopplerPowerStream",
    "ColorDopplerVelocityStream",
    "CroissantArrayRecord",
    "CroissantCatalog",
    "EcgStream",
    "EchoStream",
    "FloatImageStream",
    "GenericStream",
    "ImageLayer",
    "LinkedArray",
    "LoadedArray",
    "MeshFaceStream",
    "MeshFrame",
    "MeshFrameOffsetsStream",
    "MeshPointStream",
    "MeshSequenceRef",
    "PackedMeshAnnotation",
    "PlotStyle",
    "RecordingArray",
    "RecordingObject",
    "RecordingObjectKind",
    "RecordingPlotRenderer",
    "RecordingRecord",
    "RecordingRef",
    "RecordingRelationship",
    "RecordingStore",
    "SectorGeometry",
    "SegmentalStrain",
    "SegmentalStrainPanel",
    "SourceStrainMetadata",
    "SlicePlane",
    "SpectralDopplerStream",
    "SpectralMetadata",
    "StrainPanel",
    "StreamHeader",
    "StreamMetadata",
    "StreamSummary",
    "TissueDopplerFloatStream",
    "TissueDopplerRawStream",
    "VolumeGrid",
    "VolumeSlice",
    "compose_layers",
    "data_root",
    "find_derived_recordings",
    "find_linked_recordings",
    "find_recordings",
    "find_source_recordings",
    "load_aha_strain",
    "load_croissant",
    "load_exam_folds",
    "load_recording_object",
    "load_segmental_strain",
    "opacity_from_values",
    "open_recording",
    "plot_recording",
    "rasterize_beamspace_mask",
    "rasterize_beamspace_volume_mask",
    "rasterize_packed_mesh_volume_masks",
    "render_recording_frame",
    "render_recording_video",
    "resample_sector",
    "resample_sector_stack",
    "save_recording_plot",
    "sector_lookup",
    "sector_to_cartesian",
    "slice_volume",
    "upsert_croissant_recording",
    "write_recording",
]
