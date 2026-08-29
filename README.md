# Smart Scan Strategy Simulator

**Educational / research simulation only.** This project does not interface
with real RF hardware, SDR devices, or live communications. Every emitter,
frequency, transmission, and receiver behavior in this project is
synthetic data generated purely for demonstration and research purposes.

## 1. Project Overview

This project simulates **Smart Scan Strategy for Electronic Warfare in the
absence of prior reliable intelligence of emitters and their operating
characteristics**. It compares three scanning strategies on the exact same
synthetic RF environment:

1. **Sequential scanner** — the conventional baseline (Band 0 → Band 1 →
   ... → repeat).
2. **Random scanner** — a second baseline for context.
3. **Smart ML scheduler** — uses a RandomForestClassifier trained on
   historical scan observations to prioritize which band to scan next,
   balancing exploitation (bands predicted to be active) against
   exploration (bands neglected for a while).

The whole system runs under one hard physical constraint: **the receiver
can only observe one frequency band at a time.**

## 2. Problem Statement

> Smart Scan Strategy for Electronic Warfare in the absence of prior
> reliable intelligence of emitters and their operating characteristics.

Given no prior knowledge of how many emitters exist, where they transmit,
or how they behave, can a machine-learning-driven scheduler learn — purely
from its own accumulating scan history — to intercept more transmissions,
faster, than a naive round-robin scan?

## 3. Architecture

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
                                                             |
                                                    (loop back to Receiver)
```

### Project Structure

```
smart_scan_strategy/
├── app.py                         # Streamlit dashboard (7 tabs, all phases)
├── config.py                      # Centralized configuration dataclasses
├── requirements.txt
├── README.md
│
├── data/
│   ├── generated_environment.csv  # Exported ground-truth environment
│   ├── observations.csv           # Exported scan history
│   └── environment_heatmap.html   # Exported interactive heatmap
│
├── simulation/
│   ├── emitters.py                # Periodic / Random / Bursty / Frequency-agile emitters
│   ├── environment.py             # RFEnvironment: band x time ground-truth generator
│   └── receiver.py                # SimulatedReceiver: single-band-at-a-time receiver
│
├── detection/
│   └── detector.py                # Imperfect detector: Pd / Pfa -> HIT/MISS/FALSE_ALARM/CORRECT_NEGATIVE
│
├── scheduler/
│   ├── sequential.py              # Baseline round-robin scheduler
│   ├── random_scheduler.py        # Random baseline scheduler
│   └── smart_scheduler.py         # ML-driven exploit/explore scheduler
│
├── ml/
│   ├── features.py                # Causal per-band feature tracking (BandStatsTracker)
│   ├── train.py                   # Dataset construction + RandomForest training
│   └── predictor.py               # Online predictor wrapping a trained model
│
├── evaluation/
│   ├── metrics.py                 # Pd, Pfa, interception rate, delay, miss rate, reward
│   └── experiment.py              # Independent experiment runner + comparison table
│
├── visualization/
│   └── heatmap.py                 # Interactive Plotly frequency-time heatmap
│
├── scripts/
│   └── view_environment.py        # CLI: generate + open heatmap in a browser
│
├── utils/                         # Reserved for shared helpers
│
└── tests/
    ├── test_phase1_environment.py
    ├── test_phase2_visualization.py
    ├── test_phase3_receiver.py
    ├── test_phase4_detection.py
    ├── test_phase5_schedulers.py
    ├── test_phase9_ml.py
    ├── test_phase10_smart_scheduler.py
    └── test_phase12_13_evaluation.py
```

## 4. Installation

```bash
cd smart_scan_strategy
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## 5. Running the Application

```bash
streamlit run app.py
```

Open **http://localhost:8501**. In the sidebar:

1. Configure simulation, detector, scheduler, and ML settings.
2. Click **Generate Environment**.
3. Click **Run Sequential Simulation** and/or **Run Random Simulation** to
   populate baseline results.
4. Click **Train ML Model** to build the activity-prediction model.
5. Click **Run Smart Simulation** to run the ML-driven scheduler.
6. Click **Run Comparison** to evaluate all three schedulers on the SAME
   environment/seed and populate the comparison table and charts.

Each phase also has a standalone CLI test under `tests/` — see Section 10.

## 6. Explanation of Each Module

### `simulation/emitters.py`
Four synthetic emitter behaviors, each producing a `(bands, time_slots)`
0/1 schedule:
- **Periodic** — fixed band, repeating on/off pattern (e.g. `0 0 1 0 0 1`).
- **Random** — fixed band, independent Bernoulli draw per slot.
- **Bursty** — fixed band, short bursts separated by inactive gaps.
- **Frequency-agile** — hops across several bands over time (sequential or
  random hop order).

### `simulation/environment.py`
`RFEnvironment.generate(config)` builds the ground-truth `environment[band][time]`
matrix as the element-wise OR of all emitters' schedules. Fully
reproducible given the same `SimulationConfig.random_seed`.

### `simulation/receiver.py`
`SimulatedReceiver` observes **one band at a time**, for `dwell_time`
consecutive slots, and never reads ground truth for any band/time it
hasn't been commanded to scan. If a `Detector` (Phase 4) is attached, every
observation passes through it to produce a realistic imperfect outcome
instead of raw ground truth.

### `detection/detector.py`
Applies configurable **Pd** (detection probability) and **Pfa** (false
alarm probability) to turn ground truth into one of four outcomes: `HIT`,
`MISS`, `FALSE_ALARM`, `CORRECT_NEGATIVE`.

### `scheduler/sequential.py`, `scheduler/random_scheduler.py`
The two baselines. Sequential cycles bands in fixed order; Random selects
uniformly at random. Neither uses any observation history.

### `ml/features.py`
`BandStatsTracker` maintains **causal**, per-band running statistics (scans,
hits, misses, time-since-last-hit, detection rate, recent activity, recent
activity trend, previous state). Every feature is computed strictly from
observations already made — never from anything in the future.

### `ml/train.py`
Builds a labelled dataset by replaying an exhaustive causal sequential
sweep of the whole environment. A row's **features** reflect everything
knowable up to time `t`; its **label** is the ground-truth transmission
state of the same band at time `t + 1`. Rows are split by time (earliest
`train_fraction` train, remainder test) and a `RandomForestClassifier` is
fit and evaluated.

### `ml/predictor.py`
`ActivityPredictor` wraps a trained model with a live tracker. During a
real run it is fed only real observations (`observe(band, time, detected_state)`)
and returns per-band "probability of activity at the next slot" —
identical in spirit to what the model was trained to predict.

### `scheduler/smart_scheduler.py`
Computes a priority score per band:

```
score(b) = weight_prediction    * predicted_probability(b)
         + weight_recent_activity * recent_activity(b)
         + weight_exploration   * exploration_bonus(b)
```

`exploration_bonus` grows with time-since-last-scan, so neglected bands
regain priority even with low predicted probability. On top of the score,
an epsilon-greedy step (probability = `exploration_factor`) occasionally
forces a uniformly random band — guaranteeing a baseline exploration rate
independent of the score formula.

### `evaluation/metrics.py`
Computes Probability of Detection, False Alarm Rate, Interception Rate,
Average Detection Delay, Miss Rate, and Average Reward — see Section 9 for
exact definitions.

### `evaluation/experiment.py`
Runs Sequential, Random, and Smart ML **independently** (fresh receiver,
detector, and tracker per run — nothing leaks between them) on the exact
same environment and seed, then builds the Section 14 comparison table.

### `visualization/heatmap.py`
Interactive Plotly frequency-time heatmap with hover (band, time, status,
active emitters), zoom, pan, and a time range slider. Can overlay a
scheduler's scan path as markers.

## 7. ML Methodology

- **Model**: `RandomForestClassifier` (scikit-learn), `class_weight="balanced"`.
- **Features** (9, all causal — see `ml/features.py`): band ID, number of
  scans, number of hits, number of misses, time since last hit, historical
  detection rate, recent activity (fraction of last N scans that hit),
  previous observed state, recent activity trend.
- **Target**: 1 if the same band has a transmission at the *next* time
  slot, else 0.
- **Leakage avoidance**: features are computed strictly from a band's own
  past observations; the temporal train/test split always trains on
  earlier scans and tests on strictly later ones.
- **Evaluation**: accuracy, precision, recall, F1, confusion matrix, and
  per-band prediction probabilities — all shown in the **ML Prediction**
  dashboard tab.

## 8. Scheduler Methodology

| Scheduler | Uses history? | Strategy |
|---|---|---|
| Sequential | No | Fixed round-robin: Band 0 → 1 → ... → N → repeat |
| Random | No | Uniformly random band every command |
| Smart ML | Yes | Weighted score (prediction + recent activity + exploration bonus), plus epsilon-greedy random moves |

The Smart ML scheduler's own tracker is updated from every real
observation it receives (the Section 11 feedback loop), so its predictions
improve as the run progresses — no future ground truth is ever consulted.

## 9. Metrics — Exact Definitions

| Metric | Formula |
|---|---|
| **Probability of Detection (Pd)** | HIT / (HIT + MISS) |
| **False Alarm Rate (Pfa)** | FALSE_ALARM / (FALSE_ALARM + CORRECT_NEGATIVE) |
| **Interception Rate** | (transmission events with ≥1 HIT while active) / (total transmission events) |
| **Average Detection Delay** | mean(first HIT time − event start time) over intercepted events |
| **Miss Rate** | 1 − Interception Rate |
| **Average Reward** | mean(+1 per HIT, −1 per FALSE_ALARM, 0 otherwise) over all scans |

A "transmission event" is one contiguous run of ground-truth activity on a
single band (see `evaluation.metrics.extract_transmission_events`).

## 10. Experimental Methodology

`evaluation/experiment.py` generates one environment (fixed seed), then
runs each scheduler on a **fresh** `SimulatedReceiver` + `Detector` (with
their own seeded RNGs) and, for the smart scheduler, a fresh
`BandStatsTracker`. No scheduler's scan history or trained state ever
leaks into another's run — this is what makes the comparison fair.

Run the standalone phase tests to see each stage validated independently:

```bash
python tests/test_phase1_environment.py       # Environment generation + reproducibility
python tests/test_phase2_visualization.py     # Heatmap export
python tests/test_phase3_receiver.py          # Single-band receiver, dwell time
python tests/test_phase4_detection.py         # Detector Pd/Pfa correctness
python tests/test_phase5_schedulers.py        # Sequential + Random schedulers
python tests/test_phase9_ml.py                # Causal features + model training/eval
python tests/test_phase10_smart_scheduler.py  # Smart scheduler explore/exploit + reproducibility
python tests/test_phase12_13_evaluation.py    # Metric correctness + independent comparison
```

## 11. Limitations

- This is a **synthetic simulation**. It does not interface with real RF
  hardware, SDR hardware, or real-world communications, and its results
  say nothing about real electromagnetic environments.
- The RandomForest model is trained on one exhaustive calibration sweep of
  the same environment it is then evaluated on; a production system would
  need to validate generalization across many independently generated
  environments.
- The smart scheduler's exploration/exploitation weights are fixed
  constants (documented in `scheduler/smart_scheduler.py`); they are not
  themselves learned or tuned automatically.
- Detection is modeled as a simple Bernoulli process (Pd/Pfa); it does not
  model SNR, propagation, or hardware-specific effects.

## 12. Future Improvements

- Reinforcement learning scheduler (Gymnasium + Stable-Baselines3) as an
  alternative to the hand-tuned priority-score smart scheduler.
- Multi-environment training/evaluation to test generalization.
- Online/incremental model retraining during a live run, instead of a
  single offline training phase.
- Richer emitter models (variable power, SNR-dependent detection).

## Configuration Reference

Edit defaults in `config.py` or pass custom dataclasses to `AppConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_bands` | 20 | Number of simulated frequency bands |
| `num_time_slots` | 1000 | Number of discrete time slots |
| `num_emitters` | 5 | Number of synthetic emitters |
| `random_seed` | 42 | Seed for reproducible generation |
| `transmission_probability` | 0.3 | Base probability for random/agile emitters |
| `dwell_time` | 1 | Receiver slots spent on each commanded band |
| `detection_probability` | 0.85 | Detector Pd |
| `false_alarm_probability` | 0.05 | Detector Pfa |
| `exploration_factor` | 0.2 | Smart scheduler epsilon-greedy exploration rate |
| `recent_window` | 10 | Scans considered for "recent activity" features |
| `train_fraction` | 0.7 | Fraction of (time-ordered) data used for ML training |
| `random_forest_estimators` | 100 | Number of trees in the RandomForestClassifier |

## Disclaimer

This software is a synthetic educational simulation. It must not be used
to scan, detect, or interact with real electromagnetic signals or
communication systems.
