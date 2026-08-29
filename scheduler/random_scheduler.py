"""Random baseline scheduler, used alongside Sequential as a comparison point
for the ML-based smart scheduler (see Section 13/14 of the project brief).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from simulation.receiver import SimulatedReceiver


@dataclass
class RandomScheduler:
    """Selects a uniformly random band on every command, ignoring history."""

    num_bands: int
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())

    @classmethod
    def from_seed(cls, num_bands: int, seed: int | None = None) -> "RandomScheduler":
        """Build a scheduler with a reproducible RNG."""
        return cls(num_bands=num_bands, rng=np.random.default_rng(seed))

    def select_band(self, receiver: "SimulatedReceiver") -> int:
        """Return a uniformly random band ID."""
        return int(self.rng.integers(0, self.num_bands))

    def command_fn(self, receiver: "SimulatedReceiver") -> int:
        """Alias of ``select_band`` for use as a ``receiver.run_until`` callback."""
        return self.select_band(receiver)
