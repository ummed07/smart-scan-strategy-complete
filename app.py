"""Streamlit dashboard for the Smart Scan Strategy Simulator.

All phases: synthetic environment, imperfect detector, sequential/random/
smart-ML schedulers, model training, and Sequential vs Random vs Smart ML
comparison. Educational simulation only — no real RF hardware is used.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    AppConfig,
    DetectorConfig,
    MLConfig,
    SchedulerConfig,
    SimulationConfig,
)
from detection.detector import Detector
from evaluation.experiment import (
    RANDOM,
    SEQUENTIAL,
    SMART_ML,
    comparison_table,
    run_random_experiment,
    run_sequential_experiment,
    run_smart_experiment,
)
from evaluation.metrics import extract_transmission_events
from ml.features import BandStatsTracker
from ml.predictor import ActivityPredictor
from ml.train import train_activity_model
from scheduler.sequential import SequentialScheduler
from simulation.environment import RFEnvironment
from simulation.receiver import SimulatedReceiver
from visualization.heatmap import (
    HeatmapViewConfig,
    build_environment_heatmap,
    overlay_scan_path,
    save_heatmap_html,
)

st.set_page_config(
    page_title="Smart Scan Strategy Simulator",
    page_icon="📡",
    layout="wide",
)

METRIC_INFO = {
    "Detection Rate (Pd)": "HIT / (HIT + MISS) — fraction of true transmissions the receiver detected when scanning that band/time.",
    "False Alarm Rate (Pfa)": "FALSE_ALARM / (FALSE_ALARM + CORRECT_NEGATIVE) — fraction of silent slots wrongly flagged as active.",
    "Interception Rate": "Percentage of ground-truth transmission EVENTS (contiguous active runs) detected at least once while active.",
    "Avg Detection Delay": "Average slots between an event's start and the first HIT recorded during it.",
    "Miss Rate": "1 - Interception Rate: events never detected during their active window.",
    "Average Reward": "+1 per HIT, -1 per FALSE_ALARM, 0 otherwise, averaged over every scan (simple research reward).",
}


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def _init_state() -> None:
    defaults: dict[str, Any] = {
        "environment": None,
        "app_config": None,
        "sequential_run": None,
        "random_run": None,
        "smart_run": None,
        "training_result": None,
        "comparison_runs": None,
        "comparison_table": None,
        "phase3_receiver": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _reset_all() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    _init_state()


def _build_config(
    num_bands: int,
    num_time_slots: int,
    num_emitters: int,
    random_seed: int,
    transmission_probability: float,
    noise_level: float,
    detection_probability: float,
    false_alarm_probability: float,
    dwell_time: int,
    exploration_factor: float,
    recent_window: int,
    train_fraction: float,
    random_forest_estimators: int,
) -> AppConfig:
    return AppConfig(
        simulation=SimulationConfig(
            num_bands=int(num_bands),
            num_time_slots=int(num_time_slots),
            num_emitters=int(num_emitters),
            random_seed=int(random_seed),
            noise_level=float(noise_level),
            transmission_probability=float(transmission_probability),
        ),
        detector=DetectorConfig(
            detection_probability=float(detection_probability),
            false_alarm_probability=float(false_alarm_probability),
        ),
        scheduler=SchedulerConfig(
            dwell_time=int(dwell_time),
            exploration_factor=float(exploration_factor),
        ),
        ml=MLConfig(
            recent_window=int(recent_window),
            train_fraction=float(train_fraction),
            random_forest_estimators=int(random_forest_estimators),
        ),
    )


def _generate_environment(config: AppConfig) -> None:
    st.session_state.environment = RFEnvironment.generate(config.simulation)
    st.session_state.app_config = config
    st.session_state.sequential_run = None
    st.session_state.random_run = None
    st.session_state.smart_run = None
    st.session_state.training_result = None
    st.session_state.comparison_runs = None
    st.session_state.comparison_table = None
    st.session_state.phase3_receiver = None


# --------------------------------------------------------------------------- #
# Tab 1: Overview
# --------------------------------------------------------------------------- #
def _render_overview_tab(environment: RFEnvironment | None, config: AppConfig | None) -> None:
    st.subheader("Project overview")
    st.markdown(
        """
**Smart Scan Strategy for Electronic Warfare** — an educational, fully
synthetic simulation comparing a conventional sequential frequency scanner
against a machine-learning-driven "smart" scheduler, under the constraint
that a receiver can only observe **one frequency band at a time**, with no
prior reliable intelligence about emitters or their operating
characteristics.

No real RF hardware, SDR hardware, or real-world communications are used
anywhere in this project — every emitter, transmission, and detection is
synthetic data generated for research/demonstration purposes.
        """
    )

    st.subheader("System architecture")
    st.markdown(
        """
```
Simulated Environment  ->  Receiver (1 band/slot)  ->  Detector (Pd/Pfa)
        |                                                   |
   Ground truth                                HIT / MISS / FALSE_ALARM / CORRECT_NEGATIVE
                                                             |
                                              Feature Update (BandStatsTracker)
                                                             |
                                         ML Prediction (RandomForestClassifier)
                                                             |
                                      Smart Scheduler (exploit + explore) -> next band
```
        """
    )

    st.subheader("Current simulation status")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Environment", "Generated" if environment is not None else "Not generated")
    c2.metric(
        "Bands x Time",
        f"{environment.num_bands} x {environment.num_time_slots}" if environment else "-",
    )
    c3.metric(
        "Model trained",
        "Yes" if st.session_state.training_result is not None else "No",
    )
    c4.metric(
        "Comparison run",
        "Yes" if st.session_state.comparison_runs is not None else "No",
    )

    if config is not None:
        with st.expander("Active configuration"):
            st.json(
                {
                    "simulation": config.simulation.__dict__,
                    "detector": config.detector.__dict__,
                    "scheduler": config.scheduler.__dict__,
                    "ml": config.ml.__dict__,
                }
            )


# --------------------------------------------------------------------------- #
# Tab 2: Environment
# --------------------------------------------------------------------------- #
def _render_environment_tab(
    environment: RFEnvironment,
    band_start: int,
    band_end: int,
    time_start: int,
    time_end: int,
) -> None:
    stats = environment.compute_stats()
    events = extract_transmission_events(environment)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Bands x Time", f"{stats.num_bands} x {stats.num_time_slots}")
    col2.metric("Emitters", stats.num_emitters)
    col3.metric("Occupancy", f"{stats.occupancy_rate * 100:.2f}%")
    col4.metric("Active bands", stats.active_bands)
    col5.metric("Transmission events", len(events))

    view = HeatmapViewConfig(
        band_start=int(band_start),
        band_end=int(band_end),
        time_start=int(time_start),
        time_end=int(time_end),
    )
    figure = build_environment_heatmap(environment, view=view)
    st.plotly_chart(figure, use_container_width=True)

    st.subheader("Emitter information")
    st.dataframe(environment.emitter_summary(), use_container_width=True, hide_index=True)

    csv_bytes = environment.to_dataframe().to_csv(index=False).encode("utf-8")
    html_path = save_heatmap_html(figure, "data/environment_heatmap.html", auto_open=False)

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "Download environment CSV",
            data=csv_bytes,
            file_name="generated_environment.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "Download heatmap HTML",
            data=html_path.read_bytes(),
            file_name="environment_heatmap.html",
            mime="text/html",
            use_container_width=True,
        )


# --------------------------------------------------------------------------- #
# Tab 3: Receiver
# --------------------------------------------------------------------------- #
def _round_robin_command(receiver: SimulatedReceiver) -> int:
    if receiver.current_band is None:
        return 0
    return (receiver.current_band + 1) % receiver.num_bands


class _ScanShim:
    """Lightweight ScanRecord-like shim so overlay_scan_path can render a
    DataFrame-backed history (from an ExperimentRun) the same way it renders
    a live SimulatedReceiver's history."""

    __slots__ = ("time", "scanned_band", "actual_state")

    def __init__(self, time: int, scanned_band: int, actual_state: int) -> None:
        self.time = time
        self.scanned_band = scanned_band
        self.actual_state = actual_state


def _render_receiver_tab(
    environment: RFEnvironment,
    config: AppConfig,
    band_start: int,
    band_end: int,
    time_start: int,
    time_end: int,
) -> None:
    st.markdown(
        "The simulated receiver observes **one band at a time** and never reads "
        "future time slots. Use the sidebar **Run Sequential Simulation** button "
        "(or the demo sweep below) to populate scan history with realistic "
        "HIT/MISS/FALSE_ALARM/CORRECT_NEGATIVE outcomes from the Phase 4 detector."
    )

    demo = st.button("Run short demo sweep (this tab only)")
    if demo:
        detector = Detector.from_config(config.detector, seed=config.simulation.random_seed + 999)
        receiver = SimulatedReceiver(
            num_bands=environment.num_bands,
            dwell_time=int(config.scheduler.dwell_time),
            detector=detector,
        )
        end_time = min(300, environment.num_time_slots)
        receiver.run_until(environment, end_time=end_time, command_fn=_round_robin_command)
        st.session_state.phase3_receiver = receiver

    active_history: pd.DataFrame | None = None
    source_label = None

    if st.session_state.phase3_receiver is not None:
        active_history = st.session_state.phase3_receiver.history_dataframe()
        source_label = "Demo sweep"
    elif st.session_state.sequential_run is not None:
        active_history = st.session_state.sequential_run.history
        source_label = "Sequential Simulation (sidebar)"

    if active_history is None or active_history.empty:
        st.info(
            "No scan history yet. Click **Run short demo sweep** above, or "
            "**Run Sequential Simulation** in the sidebar."
        )
        return

    st.caption(f"Showing scan history from: **{source_label}**")

    last = active_history.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Scans recorded", len(active_history))
    c2.metric("Last band scanned", int(last["scanned_band"]))
    c3.metric("Last time", int(last["time"]))
    c4.metric("Last outcome", str(last["outcome"]))

    view = HeatmapViewConfig(
        band_start=int(band_start),
        band_end=int(band_end),
        time_start=int(time_start),
        time_end=int(time_end),
    )
    figure = build_environment_heatmap(
        environment, view=view, title="Receiver scan path on synthetic environment"
    )

    records = [
        _ScanShim(int(r.time), int(r.scanned_band), int(r.actual_state))
        for r in active_history.itertuples()
    ]
    overlay_scan_path(figure, records)
    st.plotly_chart(figure, use_container_width=True)

    st.subheader("Scan history")
    st.dataframe(active_history, use_container_width=True, hide_index=True, height=360)

    st.subheader("Outcome breakdown")
    outcome_counts = active_history["outcome"].value_counts().reset_index()
    outcome_counts.columns = ["outcome", "count"]
    st.bar_chart(outcome_counts.set_index("outcome"))

    st.download_button(
        "Download observations CSV",
        data=active_history.to_csv(index=False).encode("utf-8"),
        file_name="observations.csv",
        mime="text/csv",
        use_container_width=True,
    )


# --------------------------------------------------------------------------- #
# Tab 4: ML Prediction
# --------------------------------------------------------------------------- #
def _render_ml_tab(environment: RFEnvironment, config: AppConfig) -> None:
    result = st.session_state.training_result
    if result is None:
        st.info(
            "No model trained yet. Click **Train ML Model** in the sidebar. "
            "Training runs an exhaustive causal sequential sweep, builds a "
            "labelled dataset (features from the past, label from the ground "
            "truth at the NEXT slot), and fits a RandomForestClassifier."
        )
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{result.accuracy * 100:.1f}%")
    c2.metric("Precision", f"{result.precision * 100:.1f}%")
    c3.metric("Recall", f"{result.recall * 100:.1f}%")
    c4.metric("F1 score", f"{result.f1 * 100:.1f}%")

    st.caption(f"Trained on {result.train_size} scans, tested on {result.test_size} later scans.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Confusion matrix (test set)")
        cm = result.confusion_matrix
        cm_fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=["Predicted: No transmission", "Predicted: Transmission"],
                y=["Actual: No transmission", "Actual: Transmission"],
                text=cm,
                texttemplate="%{text}",
                colorscale="Blues",
            )
        )
        cm_fig.update_layout(height=350, margin=dict(l=40, r=40, t=30, b=40))
        st.plotly_chart(cm_fig, use_container_width=True)

    with col_b:
        st.subheader("Feature importance")
        importance_df = pd.DataFrame(
            {
                "feature": list(result.feature_importances.keys()),
                "importance": list(result.feature_importances.values()),
            }
        ).sort_values("importance", ascending=True)
        imp_fig = px.bar(importance_df, x="importance", y="feature", orientation="h")
        imp_fig.update_layout(height=350, margin=dict(l=40, r=40, t=30, b=40))
        st.plotly_chart(imp_fig, use_container_width=True)

    st.subheader("Live per-band prediction (current model state)")
    st.markdown(
        "Rebuilds a fresh, causally-updated tracker by replaying an exhaustive "
        "sequential sweep, then asks the trained model for each band's "
        "probability of activity at the next slot — exactly what the Smart "
        "Scheduler consumes at decision time."
    )
    tracker = BandStatsTracker.from_config(environment.num_bands, config.ml)
    predictor = ActivityPredictor(model=result.model, tracker=tracker)
    seed_receiver = SimulatedReceiver(num_bands=environment.num_bands, dwell_time=1)
    seed_scheduler = SequentialScheduler(num_bands=environment.num_bands)
    seed_receiver.run_until(
        environment, end_time=environment.num_time_slots, command_fn=seed_scheduler.command_fn
    )
    for record in seed_receiver.history:
        detected = (
            record.detected_state if record.detected_state is not None else record.actual_state
        )
        predictor.observe(record.scanned_band, record.time, detected)

    probabilities = predictor.predict_probabilities(environment.num_time_slots - 1)
    prob_df = pd.DataFrame(
        {"band": list(range(environment.num_bands)), "predicted_probability": probabilities}
    )
    prob_fig = px.bar(prob_df, x="band", y="predicted_probability", range_y=[0, 1])
    prob_fig.update_layout(height=320, margin=dict(l=40, r=40, t=30, b=40))
    st.plotly_chart(prob_fig, use_container_width=True)


# --------------------------------------------------------------------------- #
# Tab 5: Smart Scheduler
# --------------------------------------------------------------------------- #
def _render_smart_scheduler_tab() -> None:
    run = st.session_state.smart_run
    if run is None:
        st.info(
            "No smart-scheduler run yet. Train a model first, then click "
            "**Run Smart Simulation** in the sidebar."
        )
        return

    history = run.scheduler_history or []
    if not history:
        st.warning("Smart scheduler produced no decision history.")
        return

    exploration_moves = sum(1 for entry in history if entry.exploration_move)
    c1, c2, c3 = st.columns(3)
    c1.metric("Decisions logged", len(history))
    c2.metric("Exploration moves", exploration_moves)
    c3.metric("Exploration ratio", f"{exploration_moves / len(history) * 100:.1f}%")

    st.subheader("Recent scheduling decisions")
    tail = history[-200:]
    decisions_df = pd.DataFrame(
        {
            "time": [entry.time for entry in tail],
            "selected_band": [entry.band for entry in tail],
            "exploration_move": [entry.exploration_move for entry in tail],
        }
    )
    fig = px.scatter(
        decisions_df,
        x="time",
        y="selected_band",
        color="exploration_move",
        color_discrete_map={True: "#fbbf24", False: "#22d3ee"},
        labels={"exploration_move": "Exploration move"},
    )
    fig.update_layout(height=380, margin=dict(l=40, r=40, t=30, b=40))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Last decision — predicted priorities")
    last_entry = history[-1]
    priority_df = pd.DataFrame(
        {
            "band": list(range(len(last_entry.priority_scores))),
            "predicted_probability": last_entry.predicted_probabilities,
            "priority_score": last_entry.priority_scores,
        }
    )
    st.dataframe(priority_df, use_container_width=True, hide_index=True)
    st.caption(f"Band selected at that step: **{last_entry.band}**")

    st.subheader("Scheduler scan history")
    st.dataframe(run.history, use_container_width=True, hide_index=True, height=320)


# --------------------------------------------------------------------------- #
# Tab 6: Performance
# --------------------------------------------------------------------------- #
def _render_performance_tab() -> None:
    available = {
        SEQUENTIAL: st.session_state.sequential_run,
        RANDOM: st.session_state.random_run,
        SMART_ML: st.session_state.smart_run,
    }
    available = {name: run for name, run in available.items() if run is not None}

    if not available:
        st.info(
            "No simulation results yet. Use the sidebar buttons: "
            "**Run Sequential Simulation**, **Run Smart Simulation** "
            "(after training a model), or **Run Comparison**."
        )
        return

    choice = st.selectbox("Scheduler", list(available.keys()))
    metrics = available[choice].metrics

    c1, c2, c3 = st.columns(3)
    c1.metric("Detection Rate (Pd)", f"{metrics.detection_rate * 100:.2f}%")
    c2.metric("False Alarm Rate (Pfa)", f"{metrics.false_alarm_rate * 100:.2f}%")
    c3.metric("Interception Rate", f"{metrics.interception_rate * 100:.2f}%")

    c4, c5, c6 = st.columns(3)
    c4.metric("Avg Detection Delay", f"{metrics.avg_detection_delay:.2f} slots")
    c5.metric("Miss Rate", f"{metrics.miss_rate * 100:.2f}%")
    c6.metric("Average Reward", f"{metrics.average_reward:.3f}")

    with st.expander("What do these metrics mean?"):
        for name, description in METRIC_INFO.items():
            st.markdown(f"**{name}** — {description}")

    st.subheader("Raw counts")
    st.json(
        {
            "scans": metrics.num_scans,
            "hits": metrics.num_hits,
            "misses": metrics.num_misses,
            "false_alarms": metrics.num_false_alarms,
            "correct_negatives": metrics.num_correct_negatives,
            "transmission_events": metrics.num_events,
            "events_detected": metrics.num_events_detected,
        }
    )


# --------------------------------------------------------------------------- #
# Tab 7: Comparison
# --------------------------------------------------------------------------- #
def _render_comparison_tab() -> None:
    table = st.session_state.comparison_table
    if table is None or table.empty:
        st.info(
            "No comparison yet. Train a model, then click **Run Comparison** "
            "in the sidebar to evaluate Sequential, Random, and Smart ML on "
            "the SAME environment and seed."
        )
        return

    st.subheader("Sequential vs Random vs Smart ML")
    st.dataframe(table, use_container_width=True, hide_index=True)

    charts_left, charts_right = st.columns(2)
    with charts_left:
        st.plotly_chart(
            px.bar(table, x="Scheduler", y="Detection Rate (Pd)", title="Detection rate"),
            use_container_width=True,
        )
        st.plotly_chart(
            px.bar(table, x="Scheduler", y="False Alarm Rate (Pfa)", title="False alarm rate"),
            use_container_width=True,
        )
        st.plotly_chart(
            px.bar(table, x="Scheduler", y="Hits", title="Successful detections (hits)"),
            use_container_width=True,
        )
    with charts_right:
        st.plotly_chart(
            px.bar(table, x="Scheduler", y="Interception Rate", title="Interception rate"),
            use_container_width=True,
        )
        st.plotly_chart(
            px.bar(
                table, x="Scheduler", y="Avg Detection Delay", title="Average detection delay"
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            px.bar(table, x="Scheduler", y="Average Reward", title="Average reward"),
            use_container_width=True,
        )

    runs = st.session_state.comparison_runs
    if runs:
        st.subheader("Scheduler scan decisions over time (first 300 slots)")
        combined_rows = []
        for name, run in runs.items():
            head = run.history.head(300)
            for _, row in head.iterrows():
                combined_rows.append(
                    {"time": row["time"], "band": row["scanned_band"], "scheduler": name}
                )
        combined_df = pd.DataFrame(combined_rows)
        fig = px.scatter(combined_df, x="time", y="band", color="scheduler", opacity=0.6)
        fig.update_layout(height=420, margin=dict(l=40, r=40, t=30, b=40))
        st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "Download comparison table CSV",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="comparison_table.csv",
        mime="text/csv",
        use_container_width=True,
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    _init_state()

    st.title("Smart Scan Strategy Simulator")
    st.caption(
        "Educational / research simulation only — all emitters, bands, and "
        "transmissions are synthetic. No real RF hardware is used."
    )

    with st.sidebar:
        st.header("Simulation Settings")
        num_bands = st.slider("Number of frequency bands", 4, 40, 20)
        num_time_slots = st.slider("Number of time slots", 50, 2000, 1000, step=50)
        num_emitters = st.slider("Number of emitters", 1, 12, 5)
        random_seed = st.number_input("Random seed", min_value=0, value=42, step=1)
        noise_level = st.slider("Noise level", 0.0, 1.0, 0.0, step=0.05)
        transmission_probability = st.slider(
            "Transmission probability", min_value=0.05, max_value=0.90, value=0.30, step=0.05
        )

        st.header("Detector Settings")
        detection_probability = st.slider("Detection probability (Pd)", 0.0, 1.0, 0.85, step=0.01)
        false_alarm_probability = st.slider(
            "False alarm probability (Pfa)", 0.0, 1.0, 0.05, step=0.01
        )

        st.header("Scheduler Settings")
        dwell_time = st.slider("Dwell time (slots per band)", 1, 10, 1)
        exploration_factor = st.slider("Exploration factor (epsilon)", 0.0, 1.0, 0.2, step=0.05)

        st.header("ML Settings")
        recent_window = st.slider("Recent activity window", 3, 50, 10)
        train_fraction = st.slider("Train fraction", 0.5, 0.95, 0.7, step=0.05)
        random_forest_estimators = st.slider("Random forest trees", 20, 300, 100, step=10)

        st.header("Heatmap Viewport")
        time_start = st.number_input("Time range start", min_value=0, value=0, step=1)
        time_end = st.number_input(
            "Time range end (exclusive)", min_value=1, value=min(200, num_time_slots), step=1
        )
        band_start = st.number_input("Band range start", min_value=0, value=0, step=1)
        band_end = st.number_input(
            "Band range end (exclusive)", min_value=1, value=num_bands, step=1
        )

        st.divider()
        generate = st.button("Generate Environment", type="primary", use_container_width=True)
        run_sequential = st.button("Run Sequential Simulation", use_container_width=True)
        run_random = st.button("Run Random Simulation", use_container_width=True)
        train_model = st.button("Train ML Model", use_container_width=True)
        run_smart = st.button("Run Smart Simulation", use_container_width=True)
        run_comparison_btn = st.button("Run Comparison", use_container_width=True)
        reset = st.button("Reset", use_container_width=True)

        if reset:
            _reset_all()
            st.rerun()

    config = _build_config(
        num_bands,
        num_time_slots,
        num_emitters,
        random_seed,
        transmission_probability,
        noise_level,
        detection_probability,
        false_alarm_probability,
        dwell_time,
        exploration_factor,
        recent_window,
        train_fraction,
        random_forest_estimators,
    )

    if generate:
        _generate_environment(config)
        st.success("Synthetic environment generated.")

    environment: RFEnvironment | None = st.session_state.environment
    active_config: AppConfig | None = st.session_state.app_config

    if environment is None:
        _render_overview_tab(None, None)
        st.info(
            "Use the sidebar and click **Generate Environment** to create a synthetic RF scene."
        )
        return

    # Sidebar actions that require an existing environment.
    if run_sequential:
        st.session_state.sequential_run = run_sequential_experiment(environment, active_config)
        st.success("Sequential simulation complete.")

    if run_random:
        st.session_state.random_run = run_random_experiment(environment, active_config)
        st.success("Random simulation complete.")

    if train_model:
        try:
            st.session_state.training_result = train_activity_model(environment, active_config)
            st.success("ML model trained.")
        except ValueError as exc:
            st.warning(f"Could not train model: {exc}")

    if run_smart:
        if st.session_state.training_result is None:
            st.warning("Train the ML model first (sidebar: **Train ML Model**).")
        else:
            st.session_state.smart_run = run_smart_experiment(
                environment, active_config, st.session_state.training_result.model
            )
            st.success("Smart ML simulation complete.")

    if run_comparison_btn:
        if st.session_state.training_result is None:
            st.warning("Train the ML model first (sidebar: **Train ML Model**).")
        else:
            seq_run = run_sequential_experiment(environment, active_config)
            rand_run = run_random_experiment(environment, active_config)
            smart_run = run_smart_experiment(
                environment, active_config, st.session_state.training_result.model
            )
            runs = {SEQUENTIAL: seq_run, RANDOM: rand_run, SMART_ML: smart_run}
            st.session_state.comparison_runs = runs
            st.session_state.comparison_table = comparison_table(runs)
            st.session_state.sequential_run = seq_run
            st.session_state.random_run = rand_run
            st.session_state.smart_run = smart_run
            st.success("Comparison complete: Sequential vs Random vs Smart ML.")

    tabs = st.tabs(
        [
            "Overview",
            "Environment",
            "Receiver",
            "ML Prediction",
            "Smart Scheduler",
            "Performance",
            "Comparison",
        ]
    )
    with tabs[0]:
        _render_overview_tab(environment, active_config)
    with tabs[1]:
        _render_environment_tab(environment, band_start, band_end, time_start, time_end)
    with tabs[2]:
        _render_receiver_tab(environment, active_config, band_start, band_end, time_start, time_end)
    with tabs[3]:
        _render_ml_tab(environment, active_config)
    with tabs[4]:
        _render_smart_scheduler_tab()
    with tabs[5]:
        _render_performance_tab()
    with tabs[6]:
        _render_comparison_tab()


if __name__ == "__main__":
    main()
