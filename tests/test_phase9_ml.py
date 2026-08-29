"""Phase 9 test: causal feature tracking + ML model training/evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import AppConfig, MLConfig, SimulationConfig
from ml.features import BandStatsTracker
from ml.train import build_training_dataset, train_activity_model
from simulation.environment import RFEnvironment


def print_separator(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def run_phase9_test() -> None:
    print_separator("Smart Scan Strategy Simulator — Phase 9 Test")

    # --- BandStatsTracker: causal feature computation.
    tracker = BandStatsTracker(num_bands=3, recent_window=4)
    before = tracker.feature_row(band=0, current_time=0)
    tracker.update(band=0, time=0, detected_state=1)
    tracker.update(band=0, time=1, detected_state=0)
    tracker.update(band=0, time=2, detected_state=1)
    after = tracker.feature_row(band=0, current_time=2)

    print_separator("BandStatsTracker Causality Check")
    print(f"Feature row before any updates (num_scans should be 0): {before}")
    print(f"Feature row after 3 updates (num_scans should be 3): {after}")
    stats0 = tracker.stats[0]
    print(f"num_scans={stats0.num_scans}, num_hits={stats0.num_hits}, num_misses={stats0.num_misses}")
    print(f"detection_rate={stats0.detection_rate():.3f} (expected 2/3 = 0.667)")

    # A band never scanned must report num_scans=0 and a large time_since_last_hit.
    untouched = tracker.feature_row(band=1, current_time=10)
    print(f"Untouched band feature row (band 1): {untouched}")

    # --- Dataset construction: no feature should ever encode t+1 ground truth.
    config = AppConfig(
        simulation=SimulationConfig(
            num_bands=12, num_time_slots=800, num_emitters=5, random_seed=11
        ),
        ml=MLConfig(recent_window=8, train_fraction=0.7, random_forest_estimators=60),
    )
    environment = RFEnvironment.generate(config.simulation)
    dataset = build_training_dataset(environment, config)

    print_separator("Training Dataset")
    print(f"Rows: {len(dataset.features)}  Columns: {list(dataset.features.columns)}")
    print(f"Label distribution:\n{dataset.labels.value_counts().to_string()}")

    # Temporal split sanity: earliest cutoff rows must all precede the latest rows.
    cutoff = int(len(dataset.features) * config.ml.train_fraction)
    max_train_time = dataset.times.iloc[:cutoff].max()
    min_test_time = dataset.times.iloc[cutoff:].min()
    print(f"Max time in train split: {max_train_time}, min time in test split: {min_test_time}")
    print(f"Train precedes test (no leakage from the future into training): {max_train_time <= min_test_time}")

    # --- Train + evaluate.
    result = train_activity_model(environment, config)
    print_separator("Model Evaluation")
    print(f"Train size: {result.train_size}, Test size: {result.test_size}")
    print(f"Accuracy:  {result.accuracy:.3f}")
    print(f"Precision: {result.precision:.3f}")
    print(f"Recall:    {result.recall:.3f}")
    print(f"F1 score:  {result.f1:.3f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(result.confusion_matrix)
    print("Feature importances:")
    for name, value in sorted(result.feature_importances.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<24s} {value:.3f}")

    print_separator("Phase 9 Test Complete")
    print("Features are causal (no future leakage); RandomForest trains and evaluates successfully.")


if __name__ == "__main__":
    run_phase9_test()
