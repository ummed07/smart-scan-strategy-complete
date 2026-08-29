"""Synthetic frequency-time RF environment generator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from config import SimulationConfig
from simulation.emitters import Emitter, create_emitters


@dataclass
class EnvironmentStats:
    """Summary statistics for a generated environment."""

    num_bands: int
    num_time_slots: int
    num_emitters: int
    total_transmissions: int
    occupancy_rate: float
    active_bands: int
    transmissions_per_band: dict[int, int]
    transmissions_per_emitter: dict[int, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "num_bands": self.num_bands,
            "num_time_slots": self.num_time_slots,
            "num_emitters": self.num_emitters,
            "total_transmissions": self.total_transmissions,
            "occupancy_rate": self.occupancy_rate,
            "active_bands": self.active_bands,
            "transmissions_per_band": self.transmissions_per_band,
            "transmissions_per_emitter": self.transmissions_per_emitter,
        }


@dataclass
class RFEnvironment:
    """
    Synthetic RF environment represented as environment[band][time_slot].

    Values are 0 (no simulated transmission) or 1 (simulated transmission).
    """

    config: SimulationConfig
    matrix: np.ndarray
    emitters: list[Emitter] = field(default_factory=list)
    emitter_contributions: list[np.ndarray] = field(default_factory=list)
    rng: np.random.Generator | None = None

    @property
    def num_bands(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def num_time_slots(self) -> int:
        return int(self.matrix.shape[1])

    @classmethod
    def generate(cls, config: SimulationConfig) -> RFEnvironment:
        """Generate a reproducible synthetic environment from configuration."""
        rng = np.random.default_rng(config.random_seed)
        emitters = create_emitters(config, rng)

        matrix = np.zeros((config.num_bands, config.num_time_slots), dtype=np.int8)
        contributions: list[np.ndarray] = []

        for emitter in emitters:
            contribution = emitter.generate_schedule(
                config.num_bands,
                config.num_time_slots,
                rng,
            )
            contributions.append(contribution)
            matrix = np.maximum(matrix, contribution)

        return cls(
            config=config,
            matrix=matrix,
            emitters=emitters,
            emitter_contributions=contributions,
            rng=rng,
        )

    def get_transmission(self, band: int, time_slot: int) -> int:
        """Return ground-truth transmission state for one band/time pair."""
        return int(self.matrix[band, time_slot])

    def get_active_emitters_at(self, band: int, time_slot: int) -> list[Emitter]:
        """Return emitters transmitting on the given band at the given time slot."""
        active: list[Emitter] = []
        for emitter, contribution in zip(self.emitters, self.emitter_contributions):
            if contribution[band, time_slot] == 1:
                active.append(emitter)
        return active

    def compute_stats(self) -> EnvironmentStats:
        """Compute occupancy and activity statistics for the environment."""
        transmissions_per_band = {
            band: int(self.matrix[band, :].sum()) for band in range(self.num_bands)
        }
        transmissions_per_emitter = {
            emitter.emitter_id: int(contribution.sum())
            for emitter, contribution in zip(self.emitters, self.emitter_contributions)
        }

        total_cells = self.num_bands * self.num_time_slots
        total_transmissions = int(self.matrix.sum())
        active_bands = sum(1 for count in transmissions_per_band.values() if count > 0)

        return EnvironmentStats(
            num_bands=self.num_bands,
            num_time_slots=self.num_time_slots,
            num_emitters=len(self.emitters),
            total_transmissions=total_transmissions,
            occupancy_rate=total_transmissions / total_cells if total_cells else 0.0,
            active_bands=active_bands,
            transmissions_per_band=transmissions_per_band,
            transmissions_per_emitter=transmissions_per_emitter,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Export the environment matrix as a long-form DataFrame."""
        rows: list[dict[str, int]] = []
        for band in range(self.num_bands):
            for time_slot in range(self.num_time_slots):
                rows.append(
                    {
                        "band": band,
                        "time_slot": time_slot,
                        "transmission": int(self.matrix[band, time_slot]),
                    }
                )
        return pd.DataFrame(rows)

    def save_csv(self, path: str) -> None:
        """Persist generated environment data to CSV."""
        self.to_dataframe().to_csv(path, index=False)

    def emitter_summary(self) -> pd.DataFrame:
        """Return a tabular summary of configured emitters."""
        records = [emitter.to_dict() for emitter in self.emitters]
        return pd.DataFrame(records)
