"""Native-clock anatomical strain curves and curve-derived AHA bullseyes."""

from __future__ import annotations

from typing import cast

import numpy as np
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle, Patch, Wedge
from matplotlib.ticker import MaxNLocator

from echoxflow.aha_strain import AHAStrain, load_aha_strain
from echoxflow.loading import LoadedArray, RecordingStore
from echoxflow.manifest import manifest_documents
from echoxflow.plotting.panels import PanelArtists
from echoxflow.plotting.specs import PanelKind, PanelSpec
from echoxflow.plotting.style import PlotStyle

AHA_MODALITIES = frozenset({"aha_strain", "aha_curves", "bullseye"})


def has_anatomical_strain(store: RecordingStore) -> bool:
    """Only automatically plot a declared supported anatomical product."""
    for metadata in manifest_documents(store.group.attrs):
        for document in metadata.get("documents", [metadata]):
            if (
                isinstance(document, dict)
                and document.get("anatomical_strain", {}).get("recovery_status") == "validated_native_tracking"
            ):
                return True
    return False


def anatomical_panel_views(store: RecordingStore) -> dict[str, str]:
    """Use validated native bindings to name linked cine panels."""
    return {
        binding["export_semantic_id"]: binding["source_view"]
        for metadata in manifest_documents(store.group.attrs)
        for document in metadata.get("documents", [metadata])
        if isinstance(document, dict)
        for binding in document.get("native_afi_state", {}).get("export_contour_bindings", [])
        if isinstance(binding.get("export_semantic_id"), str)
    }


def anatomical_specs(store: RecordingStore, *, curves: bool = True, bullseye: bool = True) -> tuple[PanelSpec, ...]:
    # Fail closed if a declared anatomical product is damaged; do not silently
    # replace it with an unrelated native summary curve.
    strain = load_aha_strain(store)
    specs = []
    items: list[tuple[PanelKind, str, str]] = []
    if curves:
        items.extend(("strain_curves", view, f"{view} regional strain") for view in ("A2C", "A3C", "A4C"))
    if bullseye:
        items.append(("bullseye", "", "Curve-derived AHA peak strain"))
    for kind, view, label in items:
        loaded = LoadedArray(
            name=f"aha_{kind}_{view}",
            data_path="annotations/anatomical_strain",
            data=np.empty(0),
            timestamps=None,
            timestamps_path=None,
            sample_rate_hz=None,
            attrs={"anatomical_strain": strain, "source_view": view},
            stream=None,
        )
        specs.append(PanelSpec(loaded=loaded, kind=kind, label=label))
    return tuple(specs)


def _strain(panel: PanelSpec) -> AHAStrain:
    return cast(AHAStrain, panel.loaded.attrs["anatomical_strain"])


def _axes(ax: Axes, style: PlotStyle, label: str) -> None:
    if style.show_panel_labels:
        ax.set_title(label, color=style.text_color, fontsize=style.panel_label_fontsize, pad=16)
    ax.set_facecolor(style.panel_facecolor)
    ax.tick_params(colors=style.text_dim_color, labelsize=0.65 * style.axis_tick_label_fontsize)
    for spine in ax.spines.values():
        spine.set_color(style.grid_color)


class StrainCurvePanelRenderer:
    """Draw actual samples on each view's clock, shifted to its own ED only.

    These summaries remain static during cine playback: the existing cine QRS
    display alignment is not a native-time correspondence to strain samples.
    """

    def render(self, ax: Axes, panel: PanelSpec, *, time_s: float, frame_index: int, style: PlotStyle) -> PanelArtists:
        del time_s, frame_index
        strain, view = _strain(panel), panel.loaded.attrs["source_view"]
        _axes(ax, style, panel.label)
        all_curves = [c for segment in strain.segments for c in segment.contributors]
        low = min(float(c.values.min()) for c in all_curves)
        high = max(float(c.values.max()) for c in all_curves)
        margin = max(2.0, (high - low) * 0.08)
        ax.set_ylim(low - margin, high + margin)
        ax.set_xlim(
            min(float(c.timestamps[0] - c.native_ed_time) for c in all_curves),
            max(float(c.timestamps[-1] - c.native_ed_time) for c in all_curves),
        )
        avc = None
        for segment in strain.segments:
            for curve in segment.contributors:
                if curve.source_view != view:
                    continue
                sid = segment.aha_segment_id
                color = colormaps["tab20"]((sid - 1) % 20)
                label = f"{sid} {segment.name.removeprefix('Basal ').removeprefix('Mid ').removeprefix('Apical ')}"
                if sid == 17:
                    label = "17 Apex contributor"
                elif segment.accepted is False:
                    label += " [rejected]"
                (line,) = ax.plot(
                    curve.timestamps - curve.native_ed_time,
                    curve.values,
                    color=color,
                    linewidth=1.3,
                    linestyle="--" if segment.accepted is False and sid != 17 else "-",
                    label=label,
                )
                line.set_gid(f"aha-{sid}-{view}")
                avc = curve.native_avc_time - curve.native_ed_time
        ax.axhline(0, color=style.grid_color, linewidth=0.8)
        ax.axvline(0, color=style.text_dim_color, linewidth=0.8, linestyle=":")
        if avc is not None:
            ax.axvline(avc, color=style.text_dim_color, linewidth=0.8, linestyle=":")
            ax.text(
                avc,
                1.01,
                "AVC",
                transform=ax.get_xaxis_transform(),
                color=style.text_dim_color,
                ha="center",
                fontsize=8,
            )
        ax.set_xlabel("Time from native ED (s)", color=style.text_color, fontsize=9)
        ax.set_ylabel("Strain (%)", color=style.text_color, fontsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.legend(
            loc="lower left",
            fontsize=7,
            labelcolor=style.text_color,
            facecolor=style.panel_facecolor,
            edgecolor=style.grid_color,
            framealpha=0.85,
        )
        return PanelArtists(lambda _time, _frame: None)


class BullseyePanelRenderer:
    """Show validated signed peak scalars with explicit regional rejection."""

    def render(self, ax: Axes, panel: PanelSpec, *, time_s: float, frame_index: int, style: PlotStyle) -> PanelArtists:
        del time_s, frame_index
        _axes(ax, style, panel.label)
        cmap = LinearSegmentedColormap.from_list("strain_peak", ["#e22431", "#fafafa", "#394ff0"])
        norm = Normalize(-30, 30, clip=True)
        for segment in _strain(panel).segments:
            sid = segment.aha_segment_id
            value = segment.signed_peak_percent
            fill = "#c6c6c6" if segment.accepted is False else cmap(norm(value))
            if sid <= 6:
                outer, width, start, stop, radius = 1.0, 1 / 3, (sid - 1) * 60, sid * 60, 5 / 6
            elif sid <= 12:
                outer, width, start, stop, radius = 2 / 3, 1 / 3, (sid - 7) * 60, (sid - 6) * 60, 0.5
            elif sid <= 16:
                outer, width, start, stop, radius = 1 / 3, 1 / 6, (sid - 13) * 90, (sid - 12) * 90, 0.25
            patch: Patch
            if sid == 17:
                patch = Circle((0, 0), 1 / 6, facecolor=fill, edgecolor=style.panel_facecolor)
                x, y = 0.0, 0.0
            else:
                patch = Wedge((0, 0), outer, start, stop, width=width, facecolor=fill, edgecolor=style.panel_facecolor)
                angle = np.deg2rad((start + stop) / 2)
                x, y = radius * np.cos(angle), radius * np.sin(angle)
            patch.set_gid(f"aha-region-{sid}")
            ax.add_patch(patch)
            label = "X" if segment.accepted is False else f"{value:.0f}"
            ax.text(
                x,
                y,
                label,
                color="#101010",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                gid=f"aha-value-{sid}",
            )
            if sid != 17:
                ax.annotate(
                    str(sid),
                    (x, y),
                    xytext=(0, 8),
                    textcoords="offset points",
                    color="#353535",
                    fontsize=6,
                    ha="center",
                    va="center",
                )
        ax.set(xlim=(-1.12, 1.12), ylim=(-1.4, 1.17), aspect="equal")
        ax.set_axis_off()
        bar = ax.inset_axes((0.13, 0.03, 0.74, 0.025))
        bar.imshow(np.linspace(-30, 30, 256)[None, :], cmap=cmap, norm=norm, aspect="auto", extent=(-30, 30, 0, 1))
        bar.set_yticks([])
        bar.set_xticks([-30, 0, 30], labels=["−30%", "0", "+30%"])
        bar.tick_params(axis="x", labelsize=7, colors=style.text_color, length=0)
        ax.text(
            0.5,
            -0.09,
            "Illustrative scale · X = rejected",
            transform=ax.transAxes,
            ha="center",
            va="top",
            color=style.text_dim_color,
            fontsize=7,
        )
        return PanelArtists(lambda _time, _frame: None)
