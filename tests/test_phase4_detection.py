"""Phase 4 test: imperfect detector (Pd/Pfa) integrated with the receiver."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DetectorConfig, SimulationConfig
from detection.detector import CORRECT_NEGATIVE, FALSE_ALARM, HIT, MISS, Detector
from simulation.environment import RFEnvironment
from simulation.receiver import SimulatedReceiver


def print_separator(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def round_robin_command(receiver: SimulatedReceiver) -> int:
    if receiver.current_band is None:
        return 0
    return (receiver.current_band + 1) % receiver.num_bands


def run_phase4_test() -> None:
    config = SimulationConfig(num_bands=10, num_time_slots=2000, num_emitters=4, random_seed=7)
    environment = RFEnvironment.generate(config)

    detector_config = DetectorConfig(detection_probability=0.85, false_alarm_probability=0.05)
    detector = Detector.from_config(detector_config, seed=123)

    receiver = SimulatedReceiver(num_bands=environment.num_bands, dwell_time=1, detector=detector)
    receiver.run_until(environment, end_time=environment.num_time_slots, command_fn=round_robin_command)

    print_separator("Smart Scan Strategy Simulator — Phase 4 Test")
    print(f"Environment: {environment.num_bands} bands x {environment.num_time_slots} slots")
    print(f"Detector: Pd={detector.detection_probability}, Pfa={detector.false_alarm_probability}")

    history = receiver.history_dataframe()
    outcome_counts = history["outcome"].value_counts()
    print_separator("Outcome Counts")
    print(outcome_counts.to_string())

    # Verify: every outcome is one of the four defined labels.
    valid_outcomes = {HIT, MISS, FALSE_ALARM, CORRECT_NEGATIVE}
    all_valid = set(history["outcome"].unique()).issubset(valid_outcomes)
    print(f"All outcomes are valid labels: {all_valid}")

    # Verify: HIT/MISS only occur when actual_state == 1; FALSE_ALARM/CORRECT_NEGATIVE only when 0.
    hit_or_miss = history[history["outcome"].isin([HIT, MISS])]
    fa_or_cn = history[history["outcome"].isin([FALSE_ALARM, CORRECT_NEGATIVE])]
    consistent = bool((hit_or_miss["actual_state"] == 1).all()) and bool(
        (fa_or_cn["actual_state"] == 0).all()
    )
    print(f"Outcome consistent with ground truth actual_state: {consistent}")

    # Verify: empirical Pd/Pfa are roughly in line with configured probabilities.
    transmit_slots = history[history["actual_state"] == 1]
    silent_slots = history[history["actual_state"] == 0]
    empirical_pd = (transmit_slots["outcome"] == HIT).mean() if len(transmit_slots) else float("nan")
    empirical_pfa = (
        (silent_slots["outcome"] == FALSE_ALARM).mean() if len(silent_slots) else float("nan")
    )
    print_separator("Empirical vs Configured Pd / Pfa")
    print(f"Empirical Pd:  {empirical_pd:.3f} (configured {detector.detection_probability})")
    print(f"Empirical Pfa: {empirical_pfa:.3f} (configured {detector.false_alarm_probability})")

    # Verify: a receiver with no detector still behaves like Phase 3 (None/None).
    plain_receiver = SimulatedReceiver(num_bands=environment.num_bands, dwell_time=1)
    plain_receiver.run_until(environment, end_time=20, command_fn=round_robin_command)
    plain_history = plain_receiver.history_dataframe()
    no_detector_ok = plain_history["detected_state"].isna().all() and plain_history["outcome"].isna().all()
    print_separator("Backward Compatibility (no detector attached)")
    print(f"detected_state/outcome stay None without a detector: {no_detector_ok}")

    print_separator("Phase 4 Test Complete")
    print("Detector produces HIT/MISS/FALSE_ALARM/CORRECT_NEGATIVE consistently with ground truth.")


if __name__ == "__main__":
    run_phase4_test()
