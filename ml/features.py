"""Causal, per-band feature engineering for the activity-prediction model.

Every feature is computed strictly from observations already made on a band
(scans that have actually happened, up to and including the current time).
Nothing here ever reads ground truth beyond what a real receiver could have
observed, which is what keeps the trained model's *predictions* free of data
leakage. (The training *labels* still use ground truth — that is normal
supervised learning, not leakage; see ``ml/train.py``.)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from config import MLConfig

FEATURE_COLUMNS: list[str] = [
    "band",
    "num_scans",
    "num_hits",
    "num_misses",
    "time_since_last_hit",
    "detection_rate",
    "recent_activity",
    "previous_state",
    "recent_activity_trend",
]


@dataclass
class BandStats:
    """Running, causally-updated statistics for a single frequency band."""

    band: int
    recent_window: int = 10
    num_scans: int = 0
    num_hits: int = 0
    num_misses: int = 0
    last_hit_time: int | None = None
    last_scan_time: int | None = None
    previous_state: int = 0
    recent_states: deque[int] = field(default_factory=deque)

    def update(self, time: int, detected_state: int) -> None:
        """Record one new (causal) observation of this band."""
        self.num_scans += 1
        if detected_state == 1:
            self.num_hits += 1
            self.last_hit_time = time
        else:
            self.num_misses += 1

        self.recent_states.append(detected_state)
        while len(self.recent_states) > self.recent_window:
            self.recent_states.popleft()

        self.previous_state = detected_state
        self.last_scan_time = time

    def time_since_last_hit(self, current_time: int) -> float:
        """Slots elapsed since the last HIT/FALSE_ALARM detection (large if never)."""
        if self.last_hit_time is None:
            return float(current_time + 1)
        return float(current_time - self.last_hit_time)

    def time_since_last_scan(self, current_time: int) -> float:
        """Slots elapsed since this band was last scanned at all (large if never)."""
        if self.last_scan_time is None:
            return float(current_time + 1)
        return float(current_time - self.last_scan_time)

    def detection_rate(self) -> float:
        """Fraction of all scans on this band that were detections."""
        return self.num_hits / self.num_scans if self.num_scans else 0.0

    def recent_activity(self) -> float:
        """Fraction of the last ``recent_window`` scans that were detections."""
        if not self.recent_states:
            return 0.0
        return sum(self.recent_states) / len(self.recent_states)

    def recent_activity_trend(self) -> float:
        """Second-half minus first-half activity within the recent window.

        Positive values mean activity is picking up; negative means it is
        tapering off. Zero when there is not enough history yet.
        """
        n = len(self.recent_states)
        if n < 2:
            return 0.0
        states = list(self.recent_states)
        mid = n // 2
        first_half = states[:mid] or [0]
        second_half = states[mid:] or [0]
        return (sum(second_half) / len(second_half)) - (sum(first_half) / len(first_half))

    def feature_vector(self, current_time: int) -> list[float]:
        """Return this band's full feature vector as of ``current_time``."""
        return [
            float(self.band),
            float(self.num_scans),
            float(self.num_hits),
            float(self.num_misses),
            self.time_since_last_hit(current_time),
            self.detection_rate(),
            self.recent_activity(),
            float(self.previous_state),
            self.recent_activity_trend(),
        ]


@dataclass
class BandStatsTracker:
    """Maintains causal per-band statistics for every band in an environment."""

    num_bands: int
    recent_window: int = 10
    stats: dict[int, BandStats] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for band in range(self.num_bands):
            self.stats.setdefault(
                band, BandStats(band=band, recent_window=self.recent_window)
            )

    @classmethod
    def from_config(cls, num_bands: int, config: MLConfig) -> "BandStatsTracker":
        """Build a tracker using the recent-window size from ``MLConfig``."""
        return cls(num_bands=num_bands, recent_window=config.recent_window)

    def update(self, band: int, time: int, detected_state: int) -> None:
        """Feed one new (causal) observation into the tracker."""
        self.stats[band].update(time, detected_state)

    def feature_row(self, band: int, current_time: int) -> list[float]:
        """Return the current feature vector for a single band."""
        return self.stats[band].feature_vector(current_time)

    def feature_matrix(self, current_time: int) -> np.ndarray:
        """Return a (num_bands, num_features) matrix of current features."""
        return np.array(
            [self.stats[band].feature_vector(current_time) for band in range(self.num_bands)],
            dtype=float,
        )

    def reset(self) -> None:
        """Clear all accumulated statistics (used between independent experiment runs)."""
        self.stats = {
            band: BandStats(band=band, recent_window=self.recent_window)
            for band in range(self.num_bands)
        }
