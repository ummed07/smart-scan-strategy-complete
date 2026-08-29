"""Synthetic emitter models for the simulated RF environment."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from config import EmitterBehavior, SimulationConfig


@dataclass
class Emitter(ABC):
    """Base class for synthetic emitters."""

    emitter_id: int
    behavior_type: EmitterBehavior
    importance: float
    primary_bands: list[int] = field(default_factory=list)

    @abstractmethod
    def generate_schedule(
        self,
        num_bands: int,
        num_time_slots: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Build a band-by-time transmission matrix for this emitter.

        Returns:
            Array of shape (num_bands, num_time_slots) with values 0 or 1.
        """

    def to_dict(self) -> dict[str, Any]:
        """Serialize emitter metadata for display and export."""
        return {
            "emitter_id": self.emitter_id,
            "behavior_type": self.behavior_type,
            "importance": self.importance,
            "primary_bands": self.primary_bands,
        }


@dataclass
class PeriodicEmitter(Emitter):
    """
    Transmits on a fixed band with a repeating on/off pattern.

    Example pattern over one period: 0 0 1 0 0 1
    """

    period: int = 6
    active_slots: int = 2
    phase_offset: int = 0

    def generate_schedule(
        self,
        num_bands: int,
        num_time_slots: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        schedule = np.zeros((num_bands, num_time_slots), dtype=np.int8)
        band = self.primary_bands[0] if self.primary_bands else 0

        for t in range(num_time_slots):
            position = (t + self.phase_offset) % self.period
            if position < self.active_slots:
                schedule[band, t] = 1

        return schedule

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data.update(
            {
                "period": self.period,
                "active_slots": self.active_slots,
                "phase_offset": self.phase_offset,
            }
        )
        return data


@dataclass
class RandomEmitter(Emitter):
    """Transmits randomly on assigned band(s) with a fixed probability per slot."""

    slot_probability: float = 0.3

    def generate_schedule(
        self,
        num_bands: int,
        num_time_slots: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        schedule = np.zeros((num_bands, num_time_slots), dtype=np.int8)
        band = self.primary_bands[0] if self.primary_bands else 0
        draws = rng.random(num_time_slots) < self.slot_probability
        schedule[band, draws] = 1
        return schedule

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["slot_probability"] = self.slot_probability
        return data


@dataclass
class BurstyEmitter(Emitter):
    """Emits short bursts separated by inactive gaps."""

    burst_length: int = 4
    gap_length: int = 12

    def generate_schedule(
        self,
        num_bands: int,
        num_time_slots: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        schedule = np.zeros((num_bands, num_time_slots), dtype=np.int8)
        band = self.primary_bands[0] if self.primary_bands else 0

        t = rng.integers(0, max(1, self.gap_length))
        while t < num_time_slots:
            end = min(num_time_slots, t + self.burst_length)
            schedule[band, t:end] = 1
            t += self.burst_length + self.gap_length

        return schedule

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data.update({"burst_length": self.burst_length, "gap_length": self.gap_length})
        return data


@dataclass
class FrequencyAgileEmitter(Emitter):
    """Hops across multiple simulated bands while transmitting."""

    hop_bands: list[int] = field(default_factory=list)
    slot_probability: float = 0.5
    hop_mode: str = "sequential"

    def generate_schedule(
        self,
        num_bands: int,
        num_time_slots: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        schedule = np.zeros((num_bands, num_time_slots), dtype=np.int8)
        bands = self.hop_bands or self.primary_bands or [0]
        hop_index = 0

        for t in range(num_time_slots):
            if self.hop_mode == "random":
                band = int(rng.choice(bands))
            else:
                band = bands[hop_index % len(bands)]
                hop_index += 1

            if rng.random() < self.slot_probability:
                schedule[band, t] = 1

        return schedule

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data.update(
            {
                "hop_bands": self.hop_bands,
                "slot_probability": self.slot_probability,
                "hop_mode": self.hop_mode,
            }
        )
        return data


def _assign_unique_band(
    emitter_index: int,
    num_bands: int,
    used_bands: set[int],
    rng: np.random.Generator,
) -> int:
    """Pick a primary band, preferring unused bands when possible."""
    if len(used_bands) < num_bands:
        available = [band for band in range(num_bands) if band not in used_bands]
        if available:
            return int(rng.choice(available))
    return emitter_index % num_bands


def create_emitters(config: SimulationConfig, rng: np.random.Generator) -> list[Emitter]:
    """
    Create a heterogeneous set of synthetic emitters from simulation config.

    Emitter types cycle through periodic, random, bursty, and frequency-agile.
    """
    emitters: list[Emitter] = []
    behaviors = list(config.emitter_behaviors)
    used_primary_bands: set[int] = set()

    for idx in range(config.num_emitters):
        behavior = behaviors[idx % len(behaviors)]
        importance = float(rng.uniform(0.4, 1.0))
        primary_band = _assign_unique_band(idx, config.num_bands, used_primary_bands, rng)
        used_primary_bands.add(primary_band)

        if behavior == "periodic":
            period = int(rng.integers(4, 10))
            active_slots = int(rng.integers(1, max(2, period // 2 + 1)))
            emitter: Emitter = PeriodicEmitter(
                emitter_id=idx,
                behavior_type="periodic",
                importance=importance,
                primary_bands=[primary_band],
                period=period,
                active_slots=active_slots,
                phase_offset=int(rng.integers(0, period)),
            )
        elif behavior == "random":
            emitter = RandomEmitter(
                emitter_id=idx,
                behavior_type="random",
                importance=importance,
                primary_bands=[primary_band],
                slot_probability=config.transmission_probability,
            )
        elif behavior == "bursty":
            emitter = BurstyEmitter(
                emitter_id=idx,
                behavior_type="bursty",
                importance=importance,
                primary_bands=[primary_band],
                burst_length=int(rng.integers(2, 6)),
                gap_length=int(rng.integers(8, 20)),
            )
        else:
            hop_count = int(rng.integers(2, min(5, config.num_bands + 1)))
            hop_bands = sorted(
                rng.choice(config.num_bands, size=hop_count, replace=False).tolist()
            )
            emitter = FrequencyAgileEmitter(
                emitter_id=idx,
                behavior_type="frequency_agile",
                importance=importance,
                primary_bands=hop_bands,
                hop_bands=hop_bands,
                slot_probability=float(
                    rng.uniform(
                        max(0.1, config.transmission_probability - 0.1),
                        min(0.9, config.transmission_probability + 0.2),
                    )
                ),
                hop_mode=str(rng.choice(["sequential", "random"])),
            )

        emitters.append(emitter)

    return emitters
