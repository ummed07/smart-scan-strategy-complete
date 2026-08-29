"""Experiment engine: runs Sequential, Random, and Smart ML schedulers on the
SAME environment (same random seed) so their performance can be fairly
compared. Each run gets a brand-new receiver, detector, and (for the smart
scheduler) tracker — nothing from one scheduler's run leaks into another's.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config import AppConfig
from detection.detector import Detector
from evaluation.metrics import RunMetrics, compute_run_metrics
from ml.features import BandStatsTracker
from ml.predictor import ActivityPredictor
from scheduler.random_scheduler import RandomScheduler
from scheduler.sequential import SequentialScheduler
from scheduler.smart_scheduler import SmartScheduler
from simulation.environment import RFEnvironment
from simulation.receiver import SimulatedReceiver

SEQUENTIAL = "Sequential"
RANDOM = "Random"
SMART_ML = "Smart ML"


@dataclass
class ExperimentRun:
    """One scheduler's full scan history plus its computed metrics."""

    name: str
    history: pd.DataFrame
    metrics: RunMetrics
    scheduler_history: list | None = None  # populated for the Smart ML run


def _run_scheduler(
    name: str,
    environment: RFEnvironment,
    config: AppConfig,
    command_fn,
    detector_seed: int,
    scheduler_history: list | None = None,
) -> ExperimentRun:
    detector = Detector.from_config(config.detector, seed=detector_seed)
    receiver = SimulatedReceiver(
        num_bands=environment.num_bands,
        dwell_time=config.scheduler.dwell_time,
        detector=detector,
    )
    receiver.run_until(environment, end_time=environment.num_time_slots, command_fn=command_fn)
    history = receiver.history_dataframe()
    metrics = compute_run_metrics(name, history, environment)
    return ExperimentRun(name=name, history=history, metrics=metrics, scheduler_history=scheduler_history)


def run_sequential_experiment(environment: RFEnvironment, config: AppConfig) -> ExperimentRun:
    """Run the baseline round-robin scheduler on ``environment``."""
    scheduler = SequentialScheduler(num_bands=environment.num_bands)
    return _run_scheduler(
        SEQUENTIAL,
        environment,
        config,
        scheduler.command_fn,
        detector_seed=config.simulation.random_seed + 101,
    )


def run_random_experiment(environment: RFEnvironment, config: AppConfig) -> ExperimentRun:
    """Run the random-band scheduler on ``environment``."""
    scheduler = RandomScheduler.from_seed(
        environment.num_bands, seed=config.simulation.random_seed + 202
    )
    return _run_scheduler(
        RANDOM,
        environment,
        config,
        scheduler.command_fn,
        detector_seed=config.simulation.random_seed + 102,
    )


def run_smart_experiment(
    environment: RFEnvironment,
    config: AppConfig,
    trained_model,
) -> ExperimentRun:
    """Run the ML-driven smart scheduler on ``environment`` using a trained model."""
    tracker = BandStatsTracker.from_config(environment.num_bands, config.ml)
    predictor = ActivityPredictor(model=trained_model, tracker=tracker)
    scheduler = SmartScheduler.build(
        predictor=predictor,
        num_bands=environment.num_bands,
        config=config.scheduler,
        seed=config.simulation.random_seed + 303,
    )
    run = _run_scheduler(
        SMART_ML,
        environment,
        config,
        scheduler.command_fn,
        detector_seed=config.simulation.random_seed + 103,
    )
    run.scheduler_history = scheduler.history
    return run


def run_comparison(
    environment: RFEnvironment,
    config: AppConfig,
    trained_model,
) -> dict[str, ExperimentRun]:
    """Run all three schedulers independently and return results by name."""
    return {
        SEQUENTIAL: run_sequential_experiment(environment, config),
        RANDOM: run_random_experiment(environment, config),
        SMART_ML: run_smart_experiment(environment, config, trained_model),
    }


def comparison_table(runs: dict[str, ExperimentRun]) -> pd.DataFrame:
    """Build the Sequential vs Random vs Smart ML comparison table (Section 14)."""
    rows = [run.metrics.as_dict() for run in runs.values()]
    return pd.DataFrame(rows)
