"""Central configuration for the Smart Scan Strategy Simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


EmitterBehavior = Literal["periodic", "random", "bursty", "frequency_agile"]


@dataclass
class SimulationConfig:
    """Parameters for synthetic RF environment generation."""

    num_bands: int = 20
    num_time_slots: int = 1000
    num_emitters: int = 5
    random_seed: int = 42
    noise_level: float = 0.0
    transmission_probability: float = 0.3

    # Emitter behavior mix (used when assigning types to emitters)
    emitter_behaviors: tuple[EmitterBehavior, ...] = (
        "periodic",
        "random",
        "bursty",
        "frequency_agile",
    )


@dataclass
class ReceiverConfig:
    """Parameters for the simulated single-band receiver."""

    dwell_time: int = 1


@dataclass
class DetectorConfig:
    """Parameters for the imperfect simulated detector (used in later phases)."""

    detection_probability: float = 0.85
    false_alarm_probability: float = 0.05


@dataclass
class SchedulerConfig:
    """Parameters for scan schedulers (used in later phases)."""

    dwell_time: int = 1
    exploration_factor: float = 0.2


@dataclass
class MLConfig:
    """Parameters for ML training (used in later phases)."""

    recent_window: int = 10
    train_fraction: float = 0.7
    random_forest_estimators: int = 100


@dataclass
class AppConfig:
    """Top-level application configuration."""

    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    receiver: ReceiverConfig = field(default_factory=ReceiverConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    ml: MLConfig = field(default_factory=MLConfig)

    data_dir: str = "data"
    environment_csv: str = "data/generated_environment.csv"
    observations_csv: str = "data/observations.csv"


DEFAULT_CONFIG = AppConfig()
