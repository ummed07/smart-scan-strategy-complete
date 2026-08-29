"""Wraps a trained model with a live, causally-updated ``BandStatsTracker`` so
the smart scheduler can query "probability of activity" per band during a
run, using only observations that have actually happened so far.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ml.features import FEATURE_COLUMNS, BandStatsTracker


@dataclass
class ActivityPredictor:
    """Online predictor: trained model + a tracker fed by real observations."""

    model: RandomForestClassifier
    tracker: BandStatsTracker

    def observe(self, band: int, time: int, detected_state: int) -> None:
        """Feed one real (causal) observation into the tracker."""
        self.tracker.update(band, time, detected_state)

    def predict_probabilities(self, current_time: int) -> np.ndarray:
        """
        Return P(transmission at the next slot) for every band, using only
        information observed so far. Never touches ground truth.
        """
        feature_matrix = self.tracker.feature_matrix(current_time)
        feature_df = pd.DataFrame(feature_matrix, columns=FEATURE_COLUMNS)
        classes = list(self.model.classes_)
        proba = self.model.predict_proba(feature_df)
        if 1 in classes:
            return proba[:, classes.index(1)]
        return np.zeros(feature_matrix.shape[0])

    def reset(self) -> None:
        """Clear the live tracker (used between independent experiment runs)."""
        self.tracker.reset()
