"""Simulated single-band receiver for the synthetic RF environment.

The receiver never inspects the full environment matrix or future time slots.
It only reads the ground-truth cell for the currently commanded band at the
current time. If a ``Detector`` (Phase 4) is attached, every observation is
passed through it to produce a realistic, imperfect HIT/MISS/FALSE_ALARM/
CORRECT_NEGATIVE outcome instead of the raw ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

from detection.detector import Detector

if TYPE_CHECKING:
    from simulation.environment import RFEnvironment


class ReceiverError(RuntimeError):
    """Raised when the receiver is used incorrectly."""


@dataclass
class ScanRecord:
    """One observation produced by a dwell slot."""

    time: int
    scanned_band: int
    actual_state: int
    detected_state: int | None = None
    outcome: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "scanned_band": self.scanned_band,
            "actual_state": self.actual_state,
            "detected_state": self.detected_state,
            "outcome": self.outcome,
        }


@dataclass
class SimulatedReceiver:
    """
    Virtual receiver that observes one simulated frequency band at a time.

    Attributes:
        num_bands: Number of bands in the simulated environment.
        dwell_time: Consecutive time slots spent on a commanded band.
        current_time: Next time slot that will be observed (starts at 0).
        current_band: Band currently commanded, or None before the first command.
        history: Chronological scan records.
        detector: Optional imperfect detector (Phase 4). If omitted,
            ``detected_state``/``outcome`` stay ``None`` (Phase 3 behavior).
    """

    num_bands: int
    dwell_time: int = 1
    current_time: int = 0
    current_band: int | None = None
    history: list[ScanRecord] = field(default_factory=list)
    detector: Detector | None = None
    _slots_on_current_band: int = 0

    def __post_init__(self) -> None:
        if self.dwell_time < 1:
            raise ValueError("dwell_time must be >= 1")
        if self.num_bands < 1:
            raise ValueError("num_bands must be >= 1")

    @property
    def needs_new_command(self) -> bool:
        """True when the scheduler should assign the next band."""
        if self.current_band is None:
            return True
        return self._slots_on_current_band >= self.dwell_time

    def command(self, band: int) -> None:
        """Accept a scheduler command to dwell on ``band``."""
        if band < 0 or band >= self.num_bands:
            raise ReceiverError(
                f"Band {band} is outside the valid range [0, {self.num_bands - 1}]."
            )
        self.current_band = band
        self._slots_on_current_band = 0

    def observe_current(self, environment: RFEnvironment) -> int:
        """
        Read only the current band/time cell from the environment.

        Does not advance time. Used internally by ``step``.
        """
        if self.current_band is None:
            raise ReceiverError("No band commanded. Call command(band) first.")
        if self.current_time >= environment.num_time_slots:
            raise ReceiverError("Simulation time has reached the end of the environment.")
        if environment.num_bands != self.num_bands:
            raise ReceiverError("Receiver num_bands does not match the environment.")

        return environment.get_transmission(self.current_band, self.current_time)

    def attach_detector(self, detector: Detector) -> None:
        """Attach (or replace) the imperfect detector used by ``step``."""
        self.detector = detector

    def step(self, environment: RFEnvironment) -> ScanRecord:
        """
        Observe the commanded band at the current time, record it, and advance time.

        If a detector is attached, ``detected_state``/``outcome`` reflect its
        imperfect Pd/Pfa classification of the ground-truth ``actual_state``.
        Otherwise they stay ``None`` (no-detector / Phase 3 behavior).
        """
        if self.needs_new_command:
            raise ReceiverError(
                "Dwell complete. Command a band before calling step() again."
            )

        actual_state = self.observe_current(environment)

        detected_state: int | None = None
        outcome: str | None = None
        if self.detector is not None:
            detected_state, outcome = self.detector.classify(actual_state)

        record = ScanRecord(
            time=self.current_time,
            scanned_band=int(self.current_band),
            actual_state=actual_state,
            detected_state=detected_state,
            outcome=outcome,
        )
        self.history.append(record)
        self.current_time += 1
        self._slots_on_current_band += 1
        return record

    def run_until(
        self,
        environment: RFEnvironment,
        end_time: int,
        command_fn,
    ) -> list[ScanRecord]:
        """
        Advance the receiver until ``end_time`` using ``command_fn`` for bands.

        ``command_fn(receiver)`` must return the next band ID whenever
        ``needs_new_command`` is True. The callback must not use future
        ground truth; this method only passes the receiver object.
        """
        records: list[ScanRecord] = []
        while self.current_time < end_time:
            if self.needs_new_command:
                self.command(int(command_fn(self)))
            records.append(self.step(environment))
        return records

    def history_dataframe(self) -> pd.DataFrame:
        """Return scan history as a DataFrame."""
        if not self.history:
            return pd.DataFrame(
                columns=[
                    "time",
                    "scanned_band",
                    "actual_state",
                    "detected_state",
                    "outcome",
                ]
            )
        return pd.DataFrame([record.to_dict() for record in self.history])

    def save_observations_csv(self, path: str) -> None:
        """Export scan history to CSV."""
        self.history_dataframe().to_csv(path, index=False)

    def reset(self) -> None:
        """Clear history and return to time 0 with no commanded band."""
        self.current_time = 0
        self.current_band = None
        self.history.clear()
        self._slots_on_current_band = 0
