"""Baseline sequential (round-robin) frequency scanner.

Scans Band 0 -> Band 1 -> ... -> Band N -> repeat, ignoring all observation
history. This is the conventional strategy the ML-based smart scheduler is
evaluated against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulation.receiver import SimulatedReceiver


@dataclass
class SequentialScheduler:
    """Cycles through every band in a fixed, repeating order."""

    num_bands: int
    _next_band: int = 0

    def select_band(self, receiver: "SimulatedReceiver") -> int:
        """Return the next band in round-robin order."""
        band = self._next_band % self.num_bands
        self._next_band += 1
        return band

    def command_fn(self, receiver: "SimulatedReceiver") -> int:
        """Alias of ``select_band`` for use as a ``receiver.run_until`` callback."""
        return self.select_band(receiver)

    def reset(self) -> None:
        """Restart the cycle from band 0."""
        self._next_band = 0
