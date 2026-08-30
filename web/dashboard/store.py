"""Session-backed simulation state using Django cache for heavy objects."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Any

from django.core.cache import cache
from django.http import HttpRequest

from config import AppConfig
from evaluation.experiment import ExperimentRun
from ml.train import TrainingResult
from simulation.environment import RFEnvironment
from simulation.receiver import SimulatedReceiver


@dataclass
class SimulationState:
    """In-memory simulation state persisted per browser session."""

    config: AppConfig | None = None
    environment: RFEnvironment | None = None
    sequential_run: ExperimentRun | None = None
    random_run: ExperimentRun | None = None
    smart_run: ExperimentRun | None = None
    training_result: TrainingResult | None = None
    comparison_runs: dict[str, ExperimentRun] | None = None
    phase3_receiver: SimulatedReceiver | None = None
    viewport: dict[str, int] = field(
        default_factory=lambda: {
            "band_start": 0,
            "band_end": 20,
            "time_start": 0,
            "time_end": 200,
        }
    )


def _cache_key(request: HttpRequest) -> str:
    if not request.session.session_key:
        request.session.save()
    return f"smartscan:{request.session.session_key}"


def load_state(request: HttpRequest) -> SimulationState:
    raw = cache.get(_cache_key(request))
    if raw is None:
        return SimulationState()
    return pickle.loads(raw)


def save_state(request: HttpRequest, state: SimulationState) -> None:
    cache.set(_cache_key(request), pickle.dumps(state), timeout=7200)


def clear_state(request: HttpRequest) -> None:
    cache.delete(_cache_key(request))
    request.session.flush()


def config_to_dict(config: AppConfig) -> dict[str, Any]:
    sim = config.simulation
    return {
        "simulation": {
            "num_bands": sim.num_bands,
            "num_time_slots": sim.num_time_slots,
            "num_emitters": sim.num_emitters,
            "random_seed": sim.random_seed,
            "noise_level": sim.noise_level,
            "transmission_probability": sim.transmission_probability,
        },
        "detector": config.detector.__dict__.copy(),
        "scheduler": config.scheduler.__dict__.copy(),
        "ml": config.ml.__dict__.copy(),
    }
