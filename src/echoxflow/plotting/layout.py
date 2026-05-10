"""Panel grid layout helpers for plotting."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from echoxflow.plotting.specs import PanelSpec


@dataclass(frozen=True)
class PanelPlacement:
    row: int
    col: int
    panel: PanelSpec
    row_span: int = 1
    col_span: int = 1


@dataclass(frozen=True)
class SpatialLayout:
    cols: int
    rows: int
    placements: tuple[PanelPlacement, ...]
    width_ratios: tuple[float, ...]


def spatial_layout(panels: tuple[PanelSpec, ...]) -> SpatialLayout:
    if not panels:
        return SpatialLayout(cols=1, rows=0, placements=(), width_ratios=(1.0,))
    beamspace = tuple(panel for panel in panels if panel.view == "beamspace")
    cartesian = tuple(panel for panel in panels if panel.view == "cartesian")
    if beamspace and cartesian:
        beamspace_layout_result = beamspace_layout(beamspace)
        rows = max(beamspace_layout_result.rows, len(cartesian))
        converted = tuple(
            PanelPlacement(
                row=placement.row,
                col=placement.col,
                panel=placement.panel,
                row_span=placement.row_span,
                col_span=placement.col_span,
            )
            for placement in beamspace_layout_result.placements
        )
        cartesian_span = rows if len(cartesian) == 1 else 1
        return SpatialLayout(
            cols=beamspace_layout_result.cols + 1,
            rows=rows,
            placements=(
                *converted,
                *(
                    PanelPlacement(row=row, col=beamspace_layout_result.cols, panel=panel, row_span=cartesian_span)
                    for row, panel in enumerate(cartesian)
                ),
            ),
            width_ratios=(
                *((1.0 / beamspace_layout_result.cols,) * beamspace_layout_result.cols),
                1.0,
            ),
        )
    if beamspace:
        return beamspace_layout(beamspace)
    cols = min(2, max(1, len(panels)))
    rows = int(math.ceil(len(panels) / cols))
    return SpatialLayout(
        cols=cols,
        rows=rows,
        placements=tuple(PanelPlacement(index // cols, index % cols, panel) for index, panel in enumerate(panels)),
        width_ratios=(1.0,) * cols,
    )


def beamspace_layout(panels: tuple[PanelSpec, ...]) -> SpatialLayout:
    if len(panels) == 3 and all(panel.kind == "image" for panel in panels):
        largest = max(panels, key=panel_spatial_area)
        others = tuple(panel for panel in panels if panel is not largest)
        return SpatialLayout(
            cols=2,
            rows=2,
            placements=(
                PanelPlacement(row=0, col=0, panel=largest, row_span=2),
                PanelPlacement(row=0, col=1, panel=others[0]),
                PanelPlacement(row=1, col=1, panel=others[1]),
            ),
            width_ratios=(1.0, 1.0),
        )
    cols = min(2, max(1, len(panels)))
    rows = int(math.ceil(len(panels) / cols))
    return SpatialLayout(
        cols=cols,
        rows=rows,
        placements=tuple(PanelPlacement(index // cols, index % cols, panel) for index, panel in enumerate(panels)),
        width_ratios=(1.0,) * cols,
    )


def panel_spatial_area(panel: PanelSpec) -> int:
    data = np.asarray(panel.loaded.data)
    if data.ndim >= 4 and data.shape[-1] in {3, 4}:
        return int(data.shape[-3]) * int(data.shape[-2])
    if data.ndim >= 3:
        return int(data.shape[-2]) * int(data.shape[-1])
    if data.ndim == 2:
        return int(data.shape[0]) * int(data.shape[1])
    return int(data.size)


def uses_ecg_timescale(panel: PanelSpec) -> bool:
    return panel.loaded.name.startswith("1d_") or panel.kind == "line"
