"""Phase 2 test: build and export the frequency-time heatmap."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import SimulationConfig
from simulation.environment import RFEnvironment
from visualization.heatmap import HeatmapViewConfig, build_environment_heatmap, save_heatmap_html


def print_separator(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def run_phase2_test() -> None:
    config = SimulationConfig(
        num_bands=20,
        num_time_slots=1000,
        num_emitters=5,
        random_seed=42,
    )

    print_separator("Smart Scan Strategy Simulator — Phase 2 Test")
    environment = RFEnvironment.generate(config)

    # Default view: all bands, first 200 time slots (easier to inspect)
    view = HeatmapViewConfig(band_start=0, band_end=20, time_start=0, time_end=200)
    band_start, band_end, time_start, time_end = view.resolve(
        environment.num_bands,
        environment.num_time_slots,
    )

    print(f"Environment matrix: {environment.matrix.shape}")
    print(f"Heatmap view: bands [{band_start}, {band_end}), time [{time_start}, {time_end})")

    figure = build_environment_heatmap(environment, view=view)
    output_path = save_heatmap_html(
        figure,
        PROJECT_ROOT / "data" / "environment_heatmap.html",
        auto_open=False,
    )

    print(f"Heatmap figure traces: {len(figure.data)}")
    print(f"HTML exported to: {output_path}")

    # Verify hover/emitter lookup on one active cell
    active_band, active_time = 3, 0
    emitters = environment.get_active_emitters_at(active_band, active_time)
    print(
        f"Active emitters at band={active_band}, time={active_time}: "
        f"{[e.emitter_id for e in emitters]}"
    )

    print_separator("Phase 2 Test Complete")
    print("Open data/environment_heatmap.html in a browser to explore the chart.")
    print("Or run: python scripts/view_environment.py")


if __name__ == "__main__":
    run_phase2_test()
