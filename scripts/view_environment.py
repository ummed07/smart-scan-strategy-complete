"""Generate a synthetic environment and open an interactive heatmap in the browser."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import SimulationConfig
from simulation.environment import RFEnvironment
from visualization.heatmap import HeatmapViewConfig, build_environment_heatmap, save_heatmap_html, show_heatmap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="View the simulated RF environment as an interactive Plotly heatmap.",
    )
    parser.add_argument("--bands", type=int, default=20, help="Number of frequency bands")
    parser.add_argument("--time-slots", type=int, default=1000, help="Number of time slots")
    parser.add_argument("--emitters", type=int, default=5, help="Number of synthetic emitters")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--transmission-probability",
        type=float,
        default=0.3,
        help="Base transmission probability for random/agile emitters",
    )
    parser.add_argument("--band-start", type=int, default=0, help="First band to display")
    parser.add_argument("--band-end", type=int, default=None, help="Last band + 1 to display")
    parser.add_argument("--time-start", type=int, default=0, help="First time slot to display")
    parser.add_argument("--time-end", type=int, default=200, help="Last time slot + 1 to display")
    parser.add_argument(
        "--output",
        type=str,
        default="data/environment_heatmap.html",
        help="HTML file path for saved heatmap",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Save HTML only; do not open the browser",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = SimulationConfig(
        num_bands=args.bands,
        num_time_slots=args.time_slots,
        num_emitters=args.emitters,
        random_seed=args.seed,
        transmission_probability=args.transmission_probability,
    )
    view = HeatmapViewConfig(
        band_start=args.band_start,
        band_end=args.band_end,
        time_start=args.time_start,
        time_end=args.time_end,
    )

    print("Generating synthetic environment...")
    environment = RFEnvironment.generate(config)

    print("Building interactive heatmap...")
    figure = build_environment_heatmap(environment, view=view)

    output_path = save_heatmap_html(figure, args.output, auto_open=False)
    print(f"Saved heatmap to: {output_path.resolve()}")

    if not args.no_browser:
        print("Opening heatmap in your default browser...")
        show_heatmap(figure)
    else:
        print("Open the HTML file above in a browser to explore the heatmap.")


if __name__ == "__main__":
    main()
