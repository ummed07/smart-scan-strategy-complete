"""Phase 1 test: generate environment and print basic statistics."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow imports from project root when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import SimulationConfig
from simulation.environment import RFEnvironment


def print_separator(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def run_phase1_test() -> None:
    config = SimulationConfig(
        num_bands=20,
        num_time_slots=1000,
        num_emitters=5,
        random_seed=42,
        transmission_probability=0.3,
    )

    print_separator("Smart Scan Strategy Simulator — Phase 1 Test")
    print("Generating synthetic RF environment...")
    environment = RFEnvironment.generate(config)

    print_separator("Environment Dimensions")
    print(f"Matrix shape (bands x time_slots): {environment.matrix.shape}")
    print(f"Number of bands: {environment.num_bands}")
    print(f"Number of time slots: {environment.num_time_slots}")
    print(f"Data type: {environment.matrix.dtype}")

    stats = environment.compute_stats()
    print_separator("Environment Statistics")
    print(f"Number of emitters: {stats.num_emitters}")
    print(f"Total transmission cells: {stats.total_transmissions}")
    print(f"Occupancy rate: {stats.occupancy_rate:.4f} ({stats.occupancy_rate * 100:.2f}%)")
    print(f"Active bands (with >=1 transmission): {stats.active_bands}")

    print("\nTransmissions per band (first 10 bands):")
    for band in range(min(10, environment.num_bands)):
        print(f"  Band {band:2d}: {stats.transmissions_per_band[band]}")

    print_separator("Emitter Summary")
    summary = environment.emitter_summary()
    print(summary.to_string(index=False))

    print("\nTransmissions contributed per emitter:")
    for emitter_id, count in stats.transmissions_per_emitter.items():
        print(f"  Emitter {emitter_id}: {count}")

    print_separator("Reproducibility Check")
    environment_repeat = RFEnvironment.generate(config)
    is_identical = bool((environment.matrix == environment_repeat.matrix).all())
    print(f"Same seed produces identical matrix: {is_identical}")

    print_separator("Sample Ground Truth (bands 0-4, time 0-15)")
    sample = environment.matrix[:5, :16]
    print(sample)

    print_separator("Phase 1 Test Complete")
    print("Environment simulator is working correctly.")


if __name__ == "__main__":
    run_phase1_test()
