# Smart Scan — Django Web Dashboard

Modern showcase UI for the Smart Scan Strategy Simulator, built with **Django** and a custom dark-theme dashboard.

## Quick start

```bash
cd smart-scan-strategy-complete
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cd web
python manage.py migrate
python manage.py runserver
```

Open **http://127.0.0.1:8000/** in your browser.

## What you get

- **Sidebar controls** — all simulation, detector, scheduler, and ML settings in one place
- **Seven tabs** — Overview, Environment, Receiver, ML Prediction, Smart Scheduler, Performance, Comparison
- **Interactive Plotly charts** — heatmaps, confusion matrix, feature importance, comparison bars
- **One-click workflow** — Generate → Train → Compare, same as the Streamlit app
- **CSV downloads** — environment, observations, and comparison table

## vs Streamlit

| | Streamlit (`app.py`) | Django (`web/`) |
|---|---|---|
| UI | Default Streamlit widgets | Custom dark dashboard, showcase-ready |
| Run command | `streamlit run app.py` | `python manage.py runserver` |
| Simulation engine | Shared (`simulation/`, `ml/`, etc.) | Same |

Both frontends use the same simulation code. Use Django for demos and presentations; use Streamlit for rapid prototyping.

## Project layout

```
web/
├── manage.py
├── smartscan_project/     # Django settings & URLs
└── dashboard/
    ├── views.py           # HTTP + JSON API
    ├── services.py        # Simulation orchestration
    ├── store.py           # Session/cache state
    ├── templates/         # HTML dashboard
    └── static/            # CSS + JS
```

## Notes

- State is stored per browser session (in-memory cache). Restarting the server clears active sessions.
- Educational simulation only — no real RF hardware.
