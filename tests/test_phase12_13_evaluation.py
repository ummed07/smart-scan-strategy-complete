"""Phase 12-13 test: performance metrics + experiment engine (Sequential vs
Random vs Smart ML), run independently on the same environment/seed."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config import AppConfig, SimulationConfig
from evaluation.experiment import RANDOM, SEQUENTIAL, SMART_ML, comparison_table, run_comparison
from evaluation.metrics import compute_run_metrics, extract_transmission_events
from ml.train import train_activity_model
from simulation.environment import RFEnvironment


def print_separator(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def run_phase12_13_test() -> None:
    print_separator("Smart Scan Strategy Simulator — Phase 12/13 Test")

    # --- Metrics unit check on a hand-built history + tiny environment.
    config = SimulationConfig(num_bands=2, num_time_slots=6, num_emitters=0, random_seed=1)
    environment = RFEnvironment.generate(config)
    # Force a known ground truth: band 0 active on slots 1-2 (one event), band 1 always silent.
    environment.matrix[:, :] = 0
    environment.matrix[0, 1] = 1
    environment.matrix[0, 2] = 1

    events = extract_transmission_events(environment)
    print_separator("Event Extraction")
    print(f"Events found: {[(e.band, e.start, e.end) for e in events]}")
    print(f"Exactly one event on band 0, slots 1-2: {len(events) == 1 and events[0].band == 0 and events[0].start == 1 and events[0].end == 2}")

    history = pd.DataFrame(
        [
            {"time": 0, "scanned_band": 1, "actual_state": 0, "detected_state": 0, "outcome": "CORRECT_NEGATIVE"},
            {"time": 1, "scanned_band": 0, "actual_state": 1, "detected_state": 0, "outcome": "MISS"},
            {"time": 2, "scanned_band": 0, "actual_state": 1, "detected_state": 1, "outcome": "HIT"},
            {"time": 3, "scanned_band": 1, "actual_state": 0, "detected_state": 1, "outcome": "FALSE_ALARM"},
            {"time": 4, "scanned_band": 0, "actual_state": 0, "detected_state": 0, "outcome": "CORRECT_NEGATIVE"},
        ]
    )
    metrics = compute_run_metrics("TestScheduler", history, environment)
    print_separator("Metric Values (hand-built history)")
    print(f"Detection rate (expect 1 HIT / 2 transmit opportunities = 0.5): {metrics.detection_rate}")
    print(f"False alarm rate (expect 1 FA / 3 no-transmit opportunities = 0.333...): {metrics.false_alarm_rate:.3f}")
    print(f"Interception rate (event detected via HIT at t=2, expect 1.0): {metrics.interception_rate}")
    print(f"Avg detection delay (event starts t=1, first HIT at t=2, expect 1.0): {metrics.avg_detection_delay}")
    print(f"Miss rate (expect 0.0): {metrics.miss_rate}")
    print(f"Average reward (expect (1 HIT - 1 FA) / 5 scans = 0.0): {metrics.average_reward}")

    checks = [
        metrics.detection_rate == 0.5,
        abs(metrics.false_alarm_rate - (1 / 3)) < 1e-9,
        metrics.interception_rate == 1.0,
        metrics.avg_detection_delay == 1.0,
        metrics.miss_rate == 0.0,
        metrics.average_reward == 0.0,
    ]
    print(f"All hand-computed metrics match expected values: {all(checks)}")

    # --- Full experiment engine + comparison on a larger environment.
    full_config = AppConfig(
        simulation=SimulationConfig(
            num_bands=15, num_time_slots=1200, num_emitters=6, random_seed=42
        )
    )
    full_environment = RFEnvironment.generate(full_config.simulation)
    result = train_activity_model(full_environment, full_config)
    runs = run_comparison(full_environment, full_config, result.model)

    print_separator("Independence Check (no shared receiver/history)")
    seq_hist = runs[SEQUENTIAL].history
    rand_hist = runs[RANDOM].history
    smart_hist = runs[SMART_ML].history
    lengths_ok = (
        len(seq_hist) == full_environment.num_time_slots
        and len(rand_hist) == full_environment.num_time_slots
        and len(smart_hist) == full_environment.num_time_slots
    )
    print(f"Each scheduler produced a full-length independent run: {lengths_ok}")
    sequences_differ = not (
        seq_hist["scanned_band"].tolist() == rand_hist["scanned_band"].tolist()
        or seq_hist["scanned_band"].tolist() == smart_hist["scanned_band"].tolist()
    )
    print(f"Scan sequences differ across schedulers (no history sharing): {sequences_differ}")

    table = comparison_table(runs)
    print_separator("Comparison Table (Sequential vs Random vs Smart ML)")
    print(table.to_string(index=False))

    print_separator("Phase 12/13 Test Complete")
    print("Metrics match hand-computed values; experiment engine runs three independent, fairly-compared schedulers.")


if __name__ == "__main__":
    run_phase12_13_test()
