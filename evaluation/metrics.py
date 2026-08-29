"""Performance metrics for comparing scan scheduling strategies.

Definitions
-----------
Probability of Detection (Pd):
    HIT count / (HIT + MISS) — i.e. successful detections divided by every
    scan opportunity where a transmission was truly present.

False Alarm Rate (Pfa):
    FALSE_ALARM count / (FALSE_ALARM + CORRECT_NEGATIVE) — false alarms
    divided by every scan opportunity where no transmission was present.

Interception Rate:
    Percentage of ground-truth transmission *events* (a contiguous run of
    activity on one band) that were detected (at least one HIT) at any
    point while the event was active.

Average Detection Delay:
    For each intercepted event, the time between the event's start and the
    first HIT recorded during it, averaged over all intercepted events.

Miss Rate:
    1 - Interception Rate: percentage of transmission events never
    detected at all during their active window.

Average Reward:
    A simple research-oriented reward: +1 per HIT, -1 per FALSE_ALARM, 0
    for MISS/CORRECT_NEGATIVE, averaged over every scan in the run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from simulation.environment import RFEnvironment


@dataclass
class TransmissionEvent:
    """A contiguous run of ground-truth activity on one band."""

    band: int
    start: int
    end: int  # inclusive


@dataclass
class RunMetrics:
    """All performance metrics computed for one scheduler's scan history."""

    scheduler_name: str
    num_scans: int
    detection_rate: float
    false_alarm_rate: float
    interception_rate: float
    avg_detection_delay: float
    miss_rate: float
    average_reward: float
    num_hits: int
    num_misses: int
    num_false_alarms: int
    num_correct_negatives: int
    num_events: int
    num_events_detected: int

    def as_dict(self) -> dict[str, float | str | int]:
        """Flat dict suitable for building the Section 14 comparison table."""
        return {
            "Scheduler": self.scheduler_name,
            "Detection Rate (Pd)": round(self.detection_rate, 4),
            "False Alarm Rate (Pfa)": round(self.false_alarm_rate, 4),
            "Interception Rate": round(self.interception_rate, 4),
            "Avg Detection Delay": round(self.avg_detection_delay, 3),
            "Miss Rate": round(self.miss_rate, 4),
            "Average Reward": round(self.average_reward, 4),
            "Scans": self.num_scans,
            "Hits": self.num_hits,
            "False Alarms": self.num_false_alarms,
            "Events": self.num_events,
            "Events Detected": self.num_events_detected,
        }


def extract_transmission_events(environment: RFEnvironment) -> list[TransmissionEvent]:
    """Find every contiguous active run (event) per band from ground truth."""
    events: list[TransmissionEvent] = []
    matrix = environment.matrix
    for band in range(environment.num_bands):
        row = matrix[band]
        in_event = False
        start = 0
        for t, value in enumerate(row):
            if value == 1 and not in_event:
                in_event = True
                start = t
            elif value == 0 and in_event:
                in_event = False
                events.append(TransmissionEvent(band=band, start=start, end=t - 1))
        if in_event:
            events.append(TransmissionEvent(band=band, start=start, end=len(row) - 1))
    return events


def _empty_metrics(scheduler_name: str) -> RunMetrics:
    return RunMetrics(
        scheduler_name=scheduler_name,
        num_scans=0,
        detection_rate=0.0,
        false_alarm_rate=0.0,
        interception_rate=0.0,
        avg_detection_delay=0.0,
        miss_rate=0.0,
        average_reward=0.0,
        num_hits=0,
        num_misses=0,
        num_false_alarms=0,
        num_correct_negatives=0,
        num_events=0,
        num_events_detected=0,
    )


def compute_run_metrics(
    scheduler_name: str,
    history: pd.DataFrame,
    environment: RFEnvironment,
) -> RunMetrics:
    """Compute every performance metric for one scheduler's scan history."""
    if history.empty or "outcome" not in history.columns:
        return _empty_metrics(scheduler_name)

    outcome_counts = history["outcome"].value_counts()
    num_hits = int(outcome_counts.get("HIT", 0))
    num_misses = int(outcome_counts.get("MISS", 0))
    num_false_alarms = int(outcome_counts.get("FALSE_ALARM", 0))
    num_correct_negatives = int(outcome_counts.get("CORRECT_NEGATIVE", 0))

    transmit_opportunities = num_hits + num_misses
    no_transmit_opportunities = num_false_alarms + num_correct_negatives

    detection_rate = num_hits / transmit_opportunities if transmit_opportunities else 0.0
    false_alarm_rate = (
        num_false_alarms / no_transmit_opportunities if no_transmit_opportunities else 0.0
    )
    average_reward = (num_hits - num_false_alarms) / len(history) if len(history) else 0.0

    hits_df = history[history["outcome"] == "HIT"]
    hit_times_by_band: dict[int, np.ndarray] = {
        band: np.sort(group["time"].to_numpy())
        for band, group in hits_df.groupby("scanned_band")
    }

    events = extract_transmission_events(environment)
    delays: list[float] = []
    events_detected = 0
    for event in events:
        band_hits = hit_times_by_band.get(event.band)
        if band_hits is None or len(band_hits) == 0:
            continue
        in_window = band_hits[(band_hits >= event.start) & (band_hits <= event.end)]
        if len(in_window) > 0:
            events_detected += 1
            delays.append(float(in_window.min() - event.start))

    num_events = len(events)
    interception_rate = events_detected / num_events if num_events else 0.0
    miss_rate = 1.0 - interception_rate if num_events else 0.0
    avg_detection_delay = float(np.mean(delays)) if delays else 0.0

    return RunMetrics(
        scheduler_name=scheduler_name,
        num_scans=len(history),
        detection_rate=detection_rate,
        false_alarm_rate=false_alarm_rate,
        interception_rate=interception_rate,
        avg_detection_delay=avg_detection_delay,
        miss_rate=miss_rate,
        average_reward=average_reward,
        num_hits=num_hits,
        num_misses=num_misses,
        num_false_alarms=num_false_alarms,
        num_correct_negatives=num_correct_negatives,
        num_events=num_events,
        num_events_detected=events_detected,
    )
