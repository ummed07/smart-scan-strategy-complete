"""Phase 3 test: simulated receiver observes one band at a time."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import SimulationConfig
from simulation.environment import RFEnvironment
from simulation.receiver import ReceiverError, SimulatedReceiver


def print_separator(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def round_robin_command(receiver: SimulatedReceiver) -> int:
    """Demo policy: cycle bands 0, 1, 2, ... without using ground truth."""
    if receiver.current_band is None:
        return 0
    return (receiver.current_band + 1) % receiver.num_bands


def run_phase3_test() -> None:
    config = SimulationConfig(
        num_bands=20,
        num_time_slots=1000,
        num_emitters=5,
        random_seed=42,
    )
    environment = RFEnvironment.generate(config)
    dwell_time = 3
    receiver = SimulatedReceiver(num_bands=environment.num_bands, dwell_time=dwell_time)

    print_separator("Smart Scan Strategy Simulator — Phase 3 Test")
    print(f"Environment: {environment.num_bands} bands × {environment.num_time_slots} slots")
    print(f"Receiver dwell time: {dwell_time} slot(s)")
    print("Policy: round-robin (demo only; sequential scheduler is Phase 5)")

    # Run a short sweep so output stays readable.
    preview_slots = 24
    receiver.run_until(environment, end_time=preview_slots, command_fn=round_robin_command)

    print_separator("Current Receiver State")
    print(f"Current time: {receiver.current_time}")
    print(f"Current band: {receiver.current_band}")
    print(f"Needs new command: {receiver.needs_new_command}")
    print(f"History length: {len(receiver.history)}")

    print_separator("Scan History (first 24 observations)")
    history = receiver.history_dataframe()
    print(history.to_string(index=False))

    # Verify: receiver only saw the commanded band, matching ground truth.
    mismatches = 0
    for record in receiver.history:
        truth = environment.get_transmission(record.scanned_band, record.time)
        if record.actual_state != truth:
            mismatches += 1
    print_separator("Ground-Truth Check")
    print(f"Mismatches vs environment.get_transmission: {mismatches}")

    # Verify dwell: each command should occupy `dwell_time` consecutive slots.
    bands = [record.scanned_band for record in receiver.history]
    dwell_ok = all(
        bands[i : i + dwell_time] == [bands[i]] * dwell_time
        for i in range(0, len(bands) - dwell_time + 1, dwell_time)
    )
    print(f"Dwell grouping correct: {dwell_ok}")

    # Receiver must refuse to step without a command after dwell completes.
    stepped_without_command = False
    try:
        receiver.step(environment)
        stepped_without_command = True
    except ReceiverError:
        pass
    print(f"Step without command correctly blocked: {not stepped_without_command}")

    # Finish remaining slots so observations.csv is a full run.
    receiver.run_until(
        environment,
        end_time=environment.num_time_slots,
        command_fn=round_robin_command,
    )
    output_path = PROJECT_ROOT / "data" / "observations.csv"
    receiver.save_observations_csv(str(output_path))

    occupied = sum(1 for record in receiver.history if record.actual_state == 1)
    print_separator("Full Sweep Summary")
    print(f"Total observations: {len(receiver.history)}")
    print(f"Observed transmissions (actual_state=1): {occupied}")
    print(f"Coverage of time axis: {len(receiver.history)} / {environment.num_time_slots}")
    print(f"Wrote: {output_path}")

    print_separator("Phase 3 Test Complete")
    print("Receiver observes one band per slot, respects dwell time, and does not scan ahead.")


if __name__ == "__main__":
    run_phase3_test()
