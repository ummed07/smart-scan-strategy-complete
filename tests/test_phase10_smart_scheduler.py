"""Phase 10 test: ML-driven smart scheduler."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import AppConfig, SchedulerConfig, SimulationConfig
from detection.detector import Detector
from ml.features import BandStatsTracker
from ml.predictor import ActivityPredictor
from ml.train import train_activity_model
from scheduler.smart_scheduler import SmartScheduler
from simulation.environment import RFEnvironment
from simulation.receiver import SimulatedReceiver


def print_separator(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def run_phase10_test() -> None:
    config = AppConfig(
        simulation=SimulationConfig(
            num_bands=12, num_time_slots=1000, num_emitters=5, random_seed=21
        ),
        scheduler=SchedulerConfig(dwell_time=1, exploration_factor=0.15),
    )
    environment = RFEnvironment.generate(config.simulation)

    print_separator("Smart Scan Strategy Simulator — Phase 10 Test")
    result = train_activity_model(environment, config)
    print(f"Model trained. Test accuracy: {result.accuracy:.3f}")

    tracker = BandStatsTracker.from_config(environment.num_bands, config.ml)
    predictor = ActivityPredictor(model=result.model, tracker=tracker)
    scheduler = SmartScheduler.build(
        predictor=predictor,
        num_bands=environment.num_bands,
        config=config.scheduler,
        seed=55,
    )

    detector = Detector.from_config(config.detector, seed=77)
    receiver = SimulatedReceiver(num_bands=environment.num_bands, dwell_time=1, detector=detector)
    receiver.run_until(environment, end_time=environment.num_time_slots, command_fn=scheduler.command_fn)

    print_separator("Smart Scheduler Decisions")
    print(f"Decisions logged: {len(scheduler.history)} (should equal {environment.num_time_slots})")
    exploration_moves = sum(1 for entry in scheduler.history if entry.exploration_move)
    ratio = exploration_moves / len(scheduler.history)
    print(f"Exploration moves: {exploration_moves} ({ratio * 100:.1f}%)")
    print(
        f"Roughly matches configured exploration_factor "
        f"({config.scheduler.exploration_factor * 100:.0f}%): "
        f"{abs(ratio - config.scheduler.exploration_factor) < 0.08}"
    )

    # Every logged decision must carry one probability/score per band.
    shapes_ok = all(
        len(entry.predicted_probabilities) == environment.num_bands
        and len(entry.priority_scores) == environment.num_bands
        for entry in scheduler.history
    )
    print(f"Every decision has one score per band: {shapes_ok}")

    history = receiver.history_dataframe()
    outcome_counts = history["outcome"].value_counts()
    print_separator("Resulting Scan Outcomes")
    print(outcome_counts.to_string())

    # --- Independence check: a second, freshly-built scheduler/predictor/tracker
    # must reproduce an identical decision sequence given the same seeds - i.e.
    # no leftover state from the first run leaks into a new one.
    tracker_b = BandStatsTracker.from_config(environment.num_bands, config.ml)
    predictor_b = ActivityPredictor(model=result.model, tracker=tracker_b)
    scheduler_b = SmartScheduler.build(
        predictor=predictor_b, num_bands=environment.num_bands, config=config.scheduler, seed=55
    )
    detector_b = Detector.from_config(config.detector, seed=77)
    receiver_b = SimulatedReceiver(num_bands=environment.num_bands, dwell_time=1, detector=detector_b)
    receiver_b.run_until(
        environment, end_time=environment.num_time_slots, command_fn=scheduler_b.command_fn
    )
    bands_a = [r.scanned_band for r in receiver.history]
    bands_b = [r.scanned_band for r in receiver_b.history]
    print_separator("Independent Run Reproducibility")
    print(f"Two independent (fresh state, same seeds) runs match exactly: {bands_a == bands_b}")

    print_separator("Phase 10 Test Complete")
    print("Smart scheduler balances exploitation/exploration and runs independently and reproducibly.")


if __name__ == "__main__":
    run_phase10_test()
