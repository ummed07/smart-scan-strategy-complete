"""ML-driven smart scheduler balancing exploitation and exploration.

Priority score
--------------
For every band ``b``, the scheduler computes:

    score(b) = weight_prediction   * predicted_probability(b)
             + weight_recent_activity * recent_activity(b)
             + weight_exploration  * exploration_bonus(b)

- ``predicted_probability(b)``: the trained model's P(transmission at the
  next slot) for band b, from ``ActivityPredictor``.
- ``recent_activity(b)``: fraction of recent scans on b that were hits —
  rewards bands that have been consistently active lately.
- ``exploration_bonus(b)``: grows with time-since-last-scan (normalized by
  ``exploration_time_norm`` and clipped to [0, 1]) — bands the scheduler has
  neglected slowly become attractive again even with a low predicted score,
  so the system keeps discovering activity the model hasn't learned yet.

On top of the weighted score, the scheduler does NOT always take the
argmax: with probability ``exploration_factor`` (from ``SchedulerConfig``)
it instead picks a uniformly random band (epsilon-greedy). This guarantees
a baseline exploration rate independent of the score formula above.

The scheduler updates its internal tracker from every HIT/MISS/FALSE_ALARM/
CORRECT_NEGATIVE observation it receives, so predictions improve as the run
progresses (the feedback loop described in Section 11 of the brief).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from config import SchedulerConfig
from ml.predictor import ActivityPredictor

if TYPE_CHECKING:
    from simulation.receiver import SimulatedReceiver


@dataclass
class SmartSchedulerHistoryEntry:
    """One scheduling decision, kept for the "Smart Scheduler" dashboard tab."""

    time: int
    band: int
    predicted_probabilities: list[float]
    priority_scores: list[float]
    exploration_move: bool


@dataclass
class SmartScheduler:
    """Selects the next band using ML predictions plus an exploration bonus."""

    predictor: ActivityPredictor
    num_bands: int
    config: SchedulerConfig = field(default_factory=SchedulerConfig)
    weight_prediction: float = 0.6
    weight_recent_activity: float = 0.2
    weight_exploration: float = 0.2
    exploration_time_norm: float = 20.0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())
    history: list[SmartSchedulerHistoryEntry] = field(default_factory=list)
    _synced_records: int = 0

    @classmethod
    def build(
        cls,
        predictor: ActivityPredictor,
        num_bands: int,
        config: SchedulerConfig,
        seed: int | None = None,
        weight_prediction: float = 0.6,
        weight_recent_activity: float = 0.2,
        weight_exploration: float = 0.2,
        exploration_time_norm: float = 20.0,
    ) -> "SmartScheduler":
        """Convenience constructor with a seeded RNG."""
        return cls(
            predictor=predictor,
            num_bands=num_bands,
            config=config,
            weight_prediction=weight_prediction,
            weight_recent_activity=weight_recent_activity,
            weight_exploration=weight_exploration,
            exploration_time_norm=exploration_time_norm,
            rng=np.random.default_rng(seed),
        )

    def _sync_from_receiver(self, receiver: "SimulatedReceiver") -> None:
        """Feed any receiver observations not yet seen into the predictor's tracker."""
        new_records = receiver.history[self._synced_records :]
        for record in new_records:
            detected = (
                record.detected_state
                if record.detected_state is not None
                else record.actual_state
            )
            self.predictor.observe(record.scanned_band, record.time, detected)
        self._synced_records = len(receiver.history)

    def _exploration_bonus(self, current_time: int) -> np.ndarray:
        tracker = self.predictor.tracker
        gaps = np.array(
            [
                tracker.stats[band].time_since_last_scan(current_time)
                for band in range(self.num_bands)
            ]
        )
        return np.clip(gaps / self.exploration_time_norm, 0.0, 1.0)

    def _recent_activity(self) -> np.ndarray:
        tracker = self.predictor.tracker
        return np.array(
            [tracker.stats[band].recent_activity() for band in range(self.num_bands)]
        )

    def select_band(self, receiver: "SimulatedReceiver") -> int:
        """Return the next band to scan; safe to use as a receiver command_fn."""
        self._sync_from_receiver(receiver)
        current_time = receiver.current_time

        predicted = self.predictor.predict_probabilities(current_time)
        recent_activity = self._recent_activity()
        exploration_bonus = self._exploration_bonus(current_time)

        scores = (
            self.weight_prediction * predicted
            + self.weight_recent_activity * recent_activity
            + self.weight_exploration * exploration_bonus
        )

        exploration_move = bool(self.rng.random() < self.config.exploration_factor)
        if exploration_move:
            band = int(self.rng.integers(0, self.num_bands))
        else:
            band = int(np.argmax(scores))

        self.history.append(
            SmartSchedulerHistoryEntry(
                time=current_time,
                band=band,
                predicted_probabilities=predicted.tolist(),
                priority_scores=scores.tolist(),
                exploration_move=exploration_move,
            )
        )
        return band

    def command_fn(self, receiver: "SimulatedReceiver") -> int:
        """Alias of ``select_band`` for use as a ``receiver.run_until`` callback."""
        return self.select_band(receiver)

    def reset(self) -> None:
        """Clear tracker state and decision history (independent experiment runs)."""
        self.predictor.reset()
        self.history.clear()
        self._synced_records = 0
