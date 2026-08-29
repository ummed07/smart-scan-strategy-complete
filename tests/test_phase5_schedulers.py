"""Phase 5 test: baseline Sequential and Random schedulers."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import SimulationConfig
from scheduler.random_scheduler import RandomScheduler
from scheduler.sequential import SequentialScheduler
from simulation.environment import RFEnvironment
from simulation.receiver import SimulatedReceiver


def print_separator(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def run_phase5_test() -> None:
    config = SimulationConfig(num_bands=8, num_time_slots=100, num_emitters=3, random_seed=3)
    environment = RFEnvironment.generate(config)

    print_separator("Smart Scan Strategy Simulator — Phase 5 Test")

    # --- Sequential scheduler: must visit every band in strict round-robin order.
    seq_scheduler = SequentialScheduler(num_bands=environment.num_bands)
    seq_receiver = SimulatedReceiver(num_bands=environment.num_bands, dwell_time=1)
    seq_receiver.run_until(
        environment, end_time=environment.num_time_slots, command_fn=seq_scheduler.command_fn
    )
    seq_bands = [record.scanned_band for record in seq_receiver.history]
    expected = [t % environment.num_bands for t in range(environment.num_time_slots)]
    print_separator("Sequential Scheduler")
    print(f"First 16 bands scanned: {seq_bands[:16]}")
    print(f"Matches strict round-robin order: {seq_bands == expected}")

    # reset() should restart the cycle from band 0.
    seq_scheduler.reset()
    seq_receiver_2 = SimulatedReceiver(num_bands=environment.num_bands, dwell_time=1)
    seq_receiver_2.run_until(environment, end_time=5, command_fn=seq_scheduler.command_fn)
    reset_ok = [r.scanned_band for r in seq_receiver_2.history] == [0, 1, 2, 3, 4]
    print(f"reset() restarts from band 0: {reset_ok}")

    # --- Random scheduler: must only ever select valid bands, and be reproducible with the same seed.
    rand_scheduler_a = RandomScheduler.from_seed(environment.num_bands, seed=99)
    rand_receiver_a = SimulatedReceiver(num_bands=environment.num_bands, dwell_time=1)
    rand_receiver_a.run_until(
        environment, end_time=environment.num_time_slots, command_fn=rand_scheduler_a.command_fn
    )
    rand_bands_a = [record.scanned_band for record in rand_receiver_a.history]

    rand_scheduler_b = RandomScheduler.from_seed(environment.num_bands, seed=99)
    rand_receiver_b = SimulatedReceiver(num_bands=environment.num_bands, dwell_time=1)
    rand_receiver_b.run_until(
        environment, end_time=environment.num_time_slots, command_fn=rand_scheduler_b.command_fn
    )
    rand_bands_b = [record.scanned_band for record in rand_receiver_b.history]

    print_separator("Random Scheduler")
    print(f"First 16 bands scanned: {rand_bands_a[:16]}")
    valid_range = all(0 <= band < environment.num_bands for band in rand_bands_a)
    print(f"All selected bands within valid range: {valid_range}")
    print(f"Same seed reproduces identical sequence: {rand_bands_a == rand_bands_b}")

    print_separator("Phase 5 Test Complete")
    print("Sequential scheduler cycles bands in fixed order; random scheduler is seed-reproducible.")


if __name__ == "__main__":
    run_phase5_test()
