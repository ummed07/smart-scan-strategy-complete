"""Imperfect simulated detector applying Pd / Pfa to receiver observations.

The detector never changes ground truth. It only decides, probabilistically,
what the receiver *reports* having seen on the band/time it is currently
dwelling on:

    If a transmission is actually present (actual_state == 1):
        - detected with probability ``detection_probability``  -> HIT
        - otherwise                                             -> MISS

    If no transmission is present (actual_state == 0):
        - a false alarm is raised with probability
          ``false_alarm_probability``                           -> FALSE_ALARM
        - otherwise                                              -> CORRECT_NEGATIVE
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import DetectorConfig

HIT = "HIT"
MISS = "MISS"
FALSE_ALARM = "FALSE_ALARM"
CORRECT_NEGATIVE = "CORRECT_NEGATIVE"

OUTCOMES = (HIT, MISS, FALSE_ALARM, CORRECT_NEGATIVE)


@dataclass
class Detector:
    """Simulated imperfect detector with configurable Pd / Pfa."""

    detection_probability: float = 0.85
    false_alarm_probability: float = 0.05
    rng: np.random.Generator | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.detection_probability <= 1.0:
            raise ValueError("detection_probability must be in [0, 1]")
        if not 0.0 <= self.false_alarm_probability <= 1.0:
            raise ValueError("false_alarm_probability must be in [0, 1]")
        if self.rng is None:
            self.rng = np.random.default_rng()

    @classmethod
    def from_config(cls, config: DetectorConfig, seed: int | None = None) -> "Detector":
        """Build a detector from a ``DetectorConfig`` with its own seeded RNG."""
        return cls(
            detection_probability=config.detection_probability,
            false_alarm_probability=config.false_alarm_probability,
            rng=np.random.default_rng(seed),
        )

    def classify(self, actual_state: int) -> tuple[int, str]:
        """
        Classify one observation.

        Args:
            actual_state: Ground-truth transmission state (0 or 1) for the
                band/time slot currently being dwelled on.

        Returns:
            (detected_state, outcome) where detected_state is 0 or 1 and
            outcome is one of HIT, MISS, FALSE_ALARM, CORRECT_NEGATIVE.
        """
        if actual_state == 1:
            if self.rng.random() < self.detection_probability:
                return 1, HIT
            return 0, MISS

        if self.rng.random() < self.false_alarm_probability:
            return 1, FALSE_ALARM
        return 0, CORRECT_NEGATIVE
