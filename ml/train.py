"""Training pipeline for the next-slot band-activity predictor.

Dataset construction
--------------------
A ``SequentialScheduler`` with dwell_time=1 drives the receiver through an
exhaustive, single-band-at-a-time sweep of the whole environment (exactly
the physical constraint the real receiver has: one band observed per time
slot). As each scan comes in, a ``BandStatsTracker`` is updated causally.
Immediately after that update, the tracker's feature vector for the scanned
band already reflects everything knowable up to and including time ``t`` —
and nothing about time ``t + 1`` or later.

The training *label* for that row is the ground-truth transmission state of
the SAME band at time ``t + 1``, taken directly from the environment. Using
ground truth as a training label is standard supervised learning, not
leakage — leakage would mean the *feature* itself peeking at time ``t + 1``,
which never happens here.

Rows are then split by time: the earliest ``train_fraction`` of scans train
the model, the remaining later scans test it — never the reverse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config import AppConfig
from detection.detector import Detector
from ml.features import FEATURE_COLUMNS, BandStatsTracker
from scheduler.sequential import SequentialScheduler
from simulation.environment import RFEnvironment
from simulation.receiver import SimulatedReceiver


@dataclass
class TrainingDataset:
    """Feature/label rows produced by an exhaustive causal sweep."""

    features: pd.DataFrame
    labels: pd.Series
    times: pd.Series


@dataclass
class TrainingResult:
    """Trained model plus evaluation artifacts for display in the dashboard."""

    model: RandomForestClassifier
    feature_columns: list[str]
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: np.ndarray
    feature_importances: dict[str, float]
    train_size: int
    test_size: int
    test_predictions: np.ndarray
    test_probabilities: np.ndarray
    test_labels: np.ndarray


def build_training_dataset(environment: RFEnvironment, config: AppConfig) -> TrainingDataset:
    """Run an exhaustive causal sweep and build a labelled feature dataset."""
    detector = Detector.from_config(config.detector, seed=config.simulation.random_seed + 1)
    tracker = BandStatsTracker.from_config(environment.num_bands, config.ml)
    receiver = SimulatedReceiver(
        num_bands=environment.num_bands,
        dwell_time=1,
        detector=detector,
    )
    scheduler = SequentialScheduler(num_bands=environment.num_bands)

    receiver.run_until(
        environment,
        end_time=environment.num_time_slots,
        command_fn=scheduler.command_fn,
    )

    rows: list[list[float]] = []
    labels: list[int] = []
    times: list[int] = []
    last_time = environment.num_time_slots - 1

    for record in receiver.history:
        detected = (
            record.detected_state if record.detected_state is not None else record.actual_state
        )
        tracker.update(record.scanned_band, record.time, detected)

        if record.time < last_time:
            next_state = environment.get_transmission(record.scanned_band, record.time + 1)
            rows.append(tracker.feature_row(record.scanned_band, record.time))
            labels.append(next_state)
            times.append(record.time)

    features = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    return TrainingDataset(
        features=features,
        labels=pd.Series(labels, name="label"),
        times=pd.Series(times, name="time"),
    )


def train_activity_model(environment: RFEnvironment, config: AppConfig) -> TrainingResult:
    """Build the dataset, split it by time, and train/evaluate a RandomForest."""
    dataset = build_training_dataset(environment, config)

    if len(dataset.features) < 20:
        raise ValueError(
            "Not enough observations to train a model. Increase the number of "
            "time slots or bands and regenerate the environment."
        )

    cutoff = int(len(dataset.features) * config.ml.train_fraction)
    cutoff = max(1, min(cutoff, len(dataset.features) - 1))

    X_train = dataset.features.iloc[:cutoff]
    y_train = dataset.labels.iloc[:cutoff]
    X_test = dataset.features.iloc[cutoff:]
    y_test = dataset.labels.iloc[cutoff:]

    if X_train.empty or X_test.empty:
        raise ValueError(
            "Train/test split produced an empty split. Adjust train_fraction "
            "or the number of time slots."
        )
    if y_train.nunique() < 2:
        raise ValueError(
            "Training data contains only one class (all activity or all "
            "silence). Increase transmission probability, emitters, or "
            "time slots so both outcomes occur."
        )

    model = RandomForestClassifier(
        n_estimators=config.ml.random_forest_estimators,
        random_state=config.simulation.random_seed,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    classes = list(model.classes_)
    if 1 in classes:
        proba_positive = y_proba[:, classes.index(1)]
    else:
        proba_positive = np.zeros(len(y_test))

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    return TrainingResult(
        model=model,
        feature_columns=FEATURE_COLUMNS,
        accuracy=float(accuracy_score(y_test, y_pred)),
        precision=float(precision_score(y_test, y_pred, zero_division=0)),
        recall=float(recall_score(y_test, y_pred, zero_division=0)),
        f1=float(f1_score(y_test, y_pred, zero_division=0)),
        confusion_matrix=cm,
        feature_importances=dict(zip(FEATURE_COLUMNS, model.feature_importances_)),
        train_size=len(X_train),
        test_size=len(X_test),
        test_predictions=y_pred,
        test_probabilities=proba_positive,
        test_labels=y_test.to_numpy(),
    )
