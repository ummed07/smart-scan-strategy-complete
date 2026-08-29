"""Interactive frequency-time heatmap visualization using Plotly."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

if TYPE_CHECKING:
    from simulation.environment import RFEnvironment


@dataclass
class HeatmapViewConfig:
    """Viewport controls for slicing the environment heatmap."""

    band_start: int = 0
    band_end: int | None = None
    time_start: int = 0
    time_end: int | None = None

    def resolve(self, num_bands: int, num_time_slots: int) -> tuple[int, int, int, int]:
        """Return validated (band_start, band_end, time_start, time_end) indices."""
        band_start = max(0, min(self.band_start, num_bands - 1))
        band_end = self.band_end if self.band_end is not None else num_bands
        band_end = max(band_start + 1, min(band_end, num_bands))

        time_start = max(0, min(self.time_start, num_time_slots - 1))
        time_end = self.time_end if self.time_end is not None else num_time_slots
        time_end = max(time_start + 1, min(time_end, num_time_slots))

        return band_start, band_end, time_start, time_end


def _active_emitter_labels(environment: RFEnvironment, band: int, time_slot: int) -> str:
    """Build a comma-separated label of emitters active at one cell."""
    labels: list[str] = []
    for emitter, contribution in zip(environment.emitters, environment.emitter_contributions):
        if contribution[band, time_slot] == 1:
            labels.append(f"E{emitter.emitter_id} ({emitter.behavior_type})")
    return ", ".join(labels) if labels else "None"


def _build_hover_text(
    environment: RFEnvironment,
    matrix_slice: np.ndarray,
    band_start: int,
    time_start: int,
) -> list[list[str]]:
    """Create per-cell hover text including transmission state and emitters."""
    hover: list[list[str]] = []
    for row_idx, band in enumerate(range(band_start, band_start + matrix_slice.shape[0])):
        row_hover: list[str] = []
        for col_idx, time_slot in enumerate(
            range(time_start, time_start + matrix_slice.shape[1])
        ):
            state = int(matrix_slice[row_idx, col_idx])
            status = "Active" if state == 1 else "Inactive"
            emitters = _active_emitter_labels(environment, band, time_slot)
            row_hover.append(
                f"Band: {band}<br>"
                f"Time slot: {time_slot}<br>"
                f"Status: {status}<br>"
                f"Emitters: {emitters}"
            )
        hover.append(row_hover)
    return hover


def build_environment_heatmap(
    environment: RFEnvironment,
    view: HeatmapViewConfig | None = None,
    title: str = "Simulated Frequency-Time Environment",
) -> go.Figure:
    """
    Build an interactive Plotly heatmap of the synthetic RF environment.

    X-axis: time slot
    Y-axis: frequency band
    Cell color: transmission status (0 = inactive, 1 = active)

    Plotly provides built-in zoom, pan, and hover. Use ``HeatmapViewConfig`` to
    limit the displayed band and time ranges before rendering.
    """
    view = view or HeatmapViewConfig()
    band_start, band_end, time_start, time_end = view.resolve(
        environment.num_bands,
        environment.num_time_slots,
    )

    matrix_slice = environment.matrix[band_start:band_end, time_start:time_end]
    band_labels = [f"Band {band}" for band in range(band_start, band_end)]
    time_labels = list(range(time_start, time_end))
    hover_text = _build_hover_text(environment, matrix_slice, band_start, time_start)

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix_slice,
            x=time_labels,
            y=band_labels,
            text=hover_text,
            hoverinfo="text",
            colorscale=[
                [0.0, "#0f172a"],
                [1.0, "#22d3ee"],
            ],
            zmin=0,
            zmax=1,
            colorbar=dict(
                title="Transmission",
                tickvals=[0, 1],
                ticktext=["Inactive (0)", "Active (1)"],
            ),
        )
    )

    stats = environment.compute_stats()
    fig.update_layout(
        title=dict(
            text=(
                f"{title}<br>"
                f"<sup>Bands {band_start}–{band_end - 1} | "
                f"Time {time_start}–{time_end - 1} | "
                f"Occupancy {stats.occupancy_rate * 100:.2f}%</sup>"
            ),
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(
            title="Time Slot",
            constrain="domain",
            rangeslider=dict(visible=True),
        ),
        yaxis=dict(
            title="Frequency Band",
            autorange="reversed",
        ),
        template="plotly_dark",
        height=max(400, (band_end - band_start) * 28),
        margin=dict(l=80, r=40, t=90, b=60),
    )

    return fig


def overlay_scan_path(figure: go.Figure, records, max_points: int = 400) -> go.Figure:
    """Overlay receiver scan locations on an existing environment heatmap."""
    if not records:
        return figure

    subset = records[-max_points:]
    figure.add_trace(
        go.Scatter(
            x=[record.time for record in subset],
            y=[f"Band {record.scanned_band}" for record in subset],
            mode="markers",
            name="Receiver scan",
            marker=dict(
                size=8,
                symbol="x",
                color=["#fbbf24" if record.actual_state == 1 else "#94a3b8" for record in subset],
            ),
            hovertext=[
                f"Scan t={record.time}<br>Band {record.scanned_band}<br>"
                f"Actual: {record.actual_state}"
                for record in subset
            ],
            hoverinfo="text",
        )
    )
    return figure


def save_heatmap_html(
    figure: go.Figure,
    path: str | Path,
    auto_open: bool = False,
) -> Path:
    """Save the heatmap to an HTML file for viewing in a browser."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(output), auto_open=auto_open, include_plotlyjs="cdn")
    return output


def show_heatmap(figure: go.Figure) -> None:
    """Open the heatmap in the default web browser."""
    figure.show()
