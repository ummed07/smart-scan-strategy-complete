"""Business logic bridging Django views and the simulation engine."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from django.http import HttpRequest, HttpResponse

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
from visualization.heatmap import HeatmapViewConfig, build_environment_heatmap, overlay_scan_path

from .store import SimulationState, clear_state, config_to_dict, load_state, save_state


class _ScanShim:
    __slots__ = ("time", "scanned_band", "actual_state")

    def __init__(self, time: int, scanned_band: int, actual_state: int) -> None:
        self.time = time
        self.scanned_band = scanned_band
        self.actual_state = actual_state


def _round_robin_command(receiver: SimulatedReceiver) -> int:
    if receiver.current_band is None:
        return 0
    return (receiver.current_band + 1) % receiver.num_bands


def build_config_from_payload(payload: dict[str, Any]) -> AppConfig:
    return AppConfig(
        simulation=SimulationConfig(
            num_bands=int(payload.get("num_bands", 20)),
            num_time_slots=int(payload.get("num_time_slots", 1000)),
            num_emitters=int(payload.get("num_emitters", 5)),
            random_seed=int(payload.get("random_seed", 42)),
            noise_level=float(payload.get("noise_level", 0.0)),
            transmission_probability=float(payload.get("transmission_probability", 0.3)),
        ),
        detector=DetectorConfig(
            detection_probability=float(payload.get("detection_probability", 0.85)),
            false_alarm_probability=float(payload.get("false_alarm_probability", 0.05)),
        ),
        scheduler=SchedulerConfig(
            dwell_time=int(payload.get("dwell_time", 1)),
            exploration_factor=float(payload.get("exploration_factor", 0.2)),
        ),
        ml=MLConfig(
            recent_window=int(payload.get("recent_window", 10)),
            train_fraction=float(payload.get("train_fraction", 0.7)),
            random_forest_estimators=int(payload.get("random_forest_estimators", 100)),
        ),
    )


def _viewport_from_payload(payload: dict[str, Any], config: AppConfig) -> dict[str, int]:
    num_bands = config.simulation.num_bands
    num_slots = config.simulation.num_time_slots
    return {
        "band_start": max(0, int(payload.get("band_start", 0))),
        "band_end": min(num_bands, int(payload.get("band_end", num_bands))),
        "time_start": max(0, int(payload.get("time_start", 0))),
        "time_end": min(num_slots, int(payload.get("time_end", min(200, num_slots)))),
    }


def _fig_json(fig: go.Figure) -> str:
    return pio.to_json(fig)


def _metrics_dict(run) -> dict[str, Any]:
    m = run.metrics
    return {
        "scheduler": m.scheduler_name,
        "detection_rate": m.detection_rate,
        "false_alarm_rate": m.false_alarm_rate,
        "interception_rate": m.interception_rate,
        "avg_detection_delay": m.avg_detection_delay,
        "miss_rate": m.miss_rate,
        "average_reward": m.average_reward,
        "num_scans": m.num_scans,
        "num_hits": m.num_hits,
        "num_misses": m.num_misses,
        "num_false_alarms": m.num_false_alarms,
        "num_correct_negatives": m.num_correct_negatives,
        "num_events": m.num_events,
        "num_events_detected": m.num_events_detected,
    }


def get_status(request: HttpRequest) -> dict[str, Any]:
    state = load_state(request)
    env = state.environment
    stats = None
    if env is not None:
        s = env.compute_stats()
        stats = {
            "num_bands": s.num_bands,
            "num_time_slots": s.num_time_slots,
            "num_emitters": s.num_emitters,
            "occupancy_rate": s.occupancy_rate,
            "active_bands": s.active_bands,
            "transmission_events": len(extract_transmission_events(env)),
        }
    return {
        "has_environment": env is not None,
        "has_model": state.training_result is not None,
        "has_comparison": state.comparison_runs is not None,
        "has_sequential": state.sequential_run is not None,
        "has_random": state.random_run is not None,
        "has_smart": state.smart_run is not None,
        "config": config_to_dict(state.config) if state.config else None,
        "viewport": state.viewport,
        "stats": stats,
        "training": _training_summary(state.training_result) if state.training_result else None,
        "metrics": {
            SEQUENTIAL: _metrics_dict(state.sequential_run) if state.sequential_run else None,
            RANDOM: _metrics_dict(state.random_run) if state.random_run else None,
            SMART_ML: _metrics_dict(state.smart_run) if state.smart_run else None,
        },
    }


def _training_summary(result) -> dict[str, Any]:
    return {
        "accuracy": result.accuracy,
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
        "train_size": result.train_size,
        "test_size": result.test_size,
    }


def generate_environment(request: HttpRequest, payload: dict[str, Any]) -> dict[str, Any]:
    state = load_state(request)
    config = build_config_from_payload(payload)
    state.config = config
    state.viewport = _viewport_from_payload(payload, config)
    state.environment = RFEnvironment.generate(config.simulation)
    state.sequential_run = None
    state.random_run = None
    state.smart_run = None
    state.training_result = None
    state.comparison_runs = None
    state.phase3_receiver = None
    save_state(request, state)
    return {"ok": True, "message": "Synthetic environment generated.", "status": get_status(request)}


def run_sequential(request: HttpRequest) -> dict[str, Any]:
    state = load_state(request)
    _require_environment(state)
    state.sequential_run = run_sequential_experiment(state.environment, state.config)
    save_state(request, state)
    return {"ok": True, "message": "Sequential simulation complete.", "status": get_status(request)}


def run_random(request: HttpRequest) -> dict[str, Any]:
    state = load_state(request)
    _require_environment(state)
    state.random_run = run_random_experiment(state.environment, state.config)
    save_state(request, state)
    return {"ok": True, "message": "Random simulation complete.", "status": get_status(request)}


def train_model(request: HttpRequest) -> dict[str, Any]:
    state = load_state(request)
    _require_environment(state)
    try:
        state.training_result = train_activity_model(state.environment, state.config)
    except ValueError as exc:
        return {"ok": False, "message": str(exc), "status": get_status(request)}
    save_state(request, state)
    return {"ok": True, "message": "ML model trained.", "status": get_status(request)}


def run_smart(request: HttpRequest) -> dict[str, Any]:
    state = load_state(request)
    _require_environment(state)
    if state.training_result is None:
        return {"ok": False, "message": "Train the ML model first.", "status": get_status(request)}
    state.smart_run = run_smart_experiment(
        state.environment, state.config, state.training_result.model
    )
    save_state(request, state)
    return {"ok": True, "message": "Smart ML simulation complete.", "status": get_status(request)}


def run_comparison(request: HttpRequest) -> dict[str, Any]:
    state = load_state(request)
    _require_environment(state)
    if state.training_result is None:
        return {"ok": False, "message": "Train the ML model first.", "status": get_status(request)}
    seq_run = run_sequential_experiment(state.environment, state.config)
    rand_run = run_random_experiment(state.environment, state.config)
    smart_run = run_smart_experiment(
        state.environment, state.config, state.training_result.model
    )
    runs = {SEQUENTIAL: seq_run, RANDOM: rand_run, SMART_ML: smart_run}
    state.comparison_runs = runs
    state.sequential_run = seq_run
    state.random_run = rand_run
    state.smart_run = smart_run
    save_state(request, state)
    return {"ok": True, "message": "Comparison complete.", "status": get_status(request)}


def run_demo_sweep(request: HttpRequest) -> dict[str, Any]:
    state = load_state(request)
    _require_environment(state)
    detector = Detector.from_config(
        state.config.detector, seed=state.config.simulation.random_seed + 999
    )
    receiver = SimulatedReceiver(
        num_bands=state.environment.num_bands,
        dwell_time=state.config.scheduler.dwell_time,
        detector=detector,
    )
    end_time = min(300, state.environment.num_time_slots)
    receiver.run_until(state.environment, end_time=end_time, command_fn=_round_robin_command)
    state.phase3_receiver = receiver
    save_state(request, state)
    return {"ok": True, "message": "Demo sweep complete.", "status": get_status(request)}


def reset_session(request: HttpRequest) -> dict[str, Any]:
    clear_state(request)
    return {"ok": True, "message": "Session reset.", "status": get_status(request)}


def update_viewport(request: HttpRequest, payload: dict[str, Any]) -> dict[str, Any]:
    state = load_state(request)
    if state.config:
        state.viewport = _viewport_from_payload(payload, state.config)
        save_state(request, state)
    return {"ok": True, "viewport": state.viewport}


def _require_environment(state: SimulationState) -> None:
    if state.environment is None or state.config is None:
        raise ValueError("Generate an environment first.")


def _heatmap_view(state: SimulationState) -> HeatmapViewConfig:
    v = state.viewport
    return HeatmapViewConfig(
        band_start=v["band_start"],
        band_end=v["band_end"],
        time_start=v["time_start"],
        time_end=v["time_end"],
    )


def chart_environment(state: SimulationState) -> str | None:
    if state.environment is None:
        return None
    fig = build_environment_heatmap(state.environment, view=_heatmap_view(state))
    return _fig_json(fig)


def chart_receiver(state: SimulationState) -> str | None:
    if state.environment is None:
        return None
    history = _active_history(state)
    if history is None or history.empty:
        return None
    fig = build_environment_heatmap(
        state.environment,
        view=_heatmap_view(state),
        title="Receiver scan path on synthetic environment",
    )
    records = [
        _ScanShim(int(r.time), int(r.scanned_band), int(r.actual_state))
        for r in history.itertuples()
    ]
    overlay_scan_path(fig, records)
    return _fig_json(fig)


def _active_history(state: SimulationState) -> pd.DataFrame | None:
    if state.phase3_receiver is not None:
        return state.phase3_receiver.history_dataframe()
    if state.sequential_run is not None:
        return state.sequential_run.history
    return None


def chart_ml(state: SimulationState) -> dict[str, Any] | None:
    result = state.training_result
    if result is None or state.environment is None or state.config is None:
        return None

    cm = result.confusion_matrix
    cm_fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Predicted: No transmission", "Predicted: Transmission"],
            y=["Actual: No transmission", "Actual: Transmission"],
            text=cm,
            texttemplate="%{text}",
            colorscale=[[0, "#0f172a"], [1, "#22d3ee"]],
        )
    )
    cm_fig.update_layout(
        height=350,
        margin=dict(l=40, r=40, t=30, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
    )

    importance_df = pd.DataFrame(
        {
            "feature": list(result.feature_importances.keys()),
            "importance": list(result.feature_importances.values()),
        }
    ).sort_values("importance", ascending=True)
    imp_fig = px.bar(importance_df, x="importance", y="feature", orientation="h")
    imp_fig.update_layout(
        height=350,
        margin=dict(l=40, r=40, t=30, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
    )
    imp_fig.update_traces(marker_color="#22d3ee")

    tracker = BandStatsTracker.from_config(state.environment.num_bands, state.config.ml)
    predictor = ActivityPredictor(model=result.model, tracker=tracker)
    seed_receiver = SimulatedReceiver(num_bands=state.environment.num_bands, dwell_time=1)
    seed_scheduler = SequentialScheduler(num_bands=state.environment.num_bands)
    seed_receiver.run_until(
        state.environment,
        end_time=state.environment.num_time_slots,
        command_fn=seed_scheduler.command_fn,
    )
    for record in seed_receiver.history:
        detected = (
            record.detected_state if record.detected_state is not None else record.actual_state
        )
        predictor.observe(record.scanned_band, record.time, detected)

    probabilities = predictor.predict_probabilities(state.environment.num_time_slots - 1)
    prob_df = pd.DataFrame(
        {"band": list(range(state.environment.num_bands)), "predicted_probability": probabilities}
    )
    prob_fig = px.bar(prob_df, x="band", y="predicted_probability", range_y=[0, 1])
    prob_fig.update_layout(
        height=320,
        margin=dict(l=40, r=40, t=30, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
    )
    prob_fig.update_traces(marker_color="#a78bfa")

    return {
        "confusion_matrix": _fig_json(cm_fig),
        "feature_importance": _fig_json(imp_fig),
        "band_probabilities": _fig_json(prob_fig),
    }


def chart_smart(state: SimulationState) -> dict[str, Any] | None:
    run = state.smart_run
    if run is None or not run.scheduler_history:
        return None
    history = run.scheduler_history
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
    fig.update_layout(
        height=380,
        margin=dict(l=40, r=40, t=30, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
    )

    last_entry = history[-1]
    priority_df = pd.DataFrame(
        {
            "band": list(range(len(last_entry.priority_scores))),
            "predicted_probability": last_entry.predicted_probabilities,
            "priority_score": last_entry.priority_scores,
        }
    )
    priority_fig = px.bar(priority_df, x="band", y="priority_score")
    priority_fig.update_layout(
        height=300,
        margin=dict(l=40, r=40, t=30, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
    )
    priority_fig.update_traces(marker_color="#34d399")

    exploration_moves = sum(1 for entry in history if entry.exploration_move)
    return {
        "decisions": _fig_json(fig),
        "priorities": _fig_json(priority_fig),
        "exploration_ratio": exploration_moves / len(history) if history else 0,
        "decisions_logged": len(history),
        "exploration_moves": exploration_moves,
        "last_band": last_entry.band,
        "priority_table": priority_df.to_dict(orient="records"),
    }


def chart_comparison(state: SimulationState) -> dict[str, Any] | None:
    if state.comparison_runs is None:
        return None
    table = comparison_table(state.comparison_runs)
    charts = {}
    for metric, title in [
        ("Detection Rate (Pd)", "Detection rate"),
        ("False Alarm Rate (Pfa)", "False alarm rate"),
        ("Hits", "Successful detections"),
        ("Interception Rate", "Interception rate"),
        ("Avg Detection Delay", "Average detection delay"),
        ("Average Reward", "Average reward"),
    ]:
        fig = px.bar(table, x="Scheduler", y=metric, title=title, color="Scheduler")
        fig.update_layout(
            height=280,
            margin=dict(l=40, r=40, t=40, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            showlegend=False,
        )
        charts[metric] = _fig_json(fig)

    combined_rows = []
    for name, run in state.comparison_runs.items():
        head = run.history.head(300)
        for _, row in head.iterrows():
            combined_rows.append(
                {"time": row["time"], "band": row["scanned_band"], "scheduler": name}
            )
    combined_df = pd.DataFrame(combined_rows)
    scatter_fig = px.scatter(combined_df, x="time", y="band", color="scheduler", opacity=0.6)
    scatter_fig.update_layout(
        height=420,
        margin=dict(l=40, r=40, t=30, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
    )

    return {
        "table": table.to_dict(orient="records"),
        "charts": charts,
        "scatter": _fig_json(scatter_fig),
    }


def chart_outcomes(state: SimulationState) -> str | None:
    history = _active_history(state)
    if history is None or history.empty:
        return None
    outcome_counts = history["outcome"].value_counts().reset_index()
    outcome_counts.columns = ["outcome", "count"]
    fig = px.bar(outcome_counts, x="outcome", y="count")
    fig.update_layout(
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
    )
    fig.update_traces(marker_color="#22d3ee")
    return _fig_json(fig)


def emitter_table(state: SimulationState) -> list[dict] | None:
    if state.environment is None:
        return None
    return state.environment.emitter_summary().to_dict(orient="records")


def history_table(state: SimulationState, source: str = "receiver") -> list[dict] | None:
    if source == "smart" and state.smart_run is not None:
        return state.smart_run.history.head(100).to_dict(orient="records")
    history = _active_history(state)
    if history is None:
        return None
    return history.head(100).to_dict(orient="records")


def download_csv_response(content: str, filename: str) -> HttpResponse:
    response = HttpResponse(content, content_type="text/csv")
    response["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response


def download_environment_csv(state: SimulationState) -> HttpResponse:
    _require_environment(state)
    return download_csv_response(
        state.environment.to_dataframe().to_csv(index=False),
        "generated_environment.csv",
    )


def download_observations_csv(state: SimulationState) -> HttpResponse:
    history = _active_history(state)
    if history is None:
        raise ValueError("No scan history available.")
    return download_csv_response(history.to_csv(index=False), "observations.csv")


def download_comparison_csv(state: SimulationState) -> HttpResponse:
    if state.comparison_runs is None:
        raise ValueError("No comparison data available.")
    table = comparison_table(state.comparison_runs)
    return download_csv_response(table.to_csv(index=False), "comparison_table.csv")
