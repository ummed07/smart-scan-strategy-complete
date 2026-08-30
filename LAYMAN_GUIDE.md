# Smart Scan Strategy — Explained in Plain English

This guide explains what the project does **without** assuming you know radio engineering, machine learning, or electronic warfare.

> **Important:** This is a **computer simulation for learning and research**. It does not connect to real radios, antennas, or live signals. Everything in it is fake data on purpose.

---

## The one-sentence version

**Imagine you have one pair of eyes but twenty TV channels might show something important at any moment — can a computer learn where to look next so it catches more action, faster, than just flipping channels in order?**

That is what this project tests.

---

## The everyday analogy

Picture a security guard watching a building with **20 doors** (frequency bands). People (signals) sometimes walk through different doors at different times. The guard can only **look at one door at a time**.

The guard does not get a map of the building upfront. They only learn from what they have already seen.

The project asks:

> **Which strategy helps the guard notice more people, sooner?**

It compares three strategies:

| Strategy | Plain English | Like… |
|----------|---------------|-------|
| **Sequential** | Check door 1, then 2, then 3… and repeat forever | A rota that never changes |
| **Random** | Pick a random door each time | Closing your eyes and pointing |
| **Smart ML** | Use past experience to guess which door is worth checking next, but still peek at neglected doors sometimes | A guard who learns patterns but does not get overconfident |

---

## What problem is this solving?

In electronic warfare and signal intelligence, receivers often cannot watch the whole radio spectrum at once. They must **choose where to tune next**.

In this project, that constraint is strict:

- **20 frequency bands** (like 20 radio channels)
- **1000 time steps** (like 1000 moments in time)
- **Only one band can be watched per step**

Unknown “emitters” (fake signal sources) transmit on various bands with different habits:

- **Periodic** — like a clock: on, off, on, off on a fixed channel
- **Random** — unpredictable blips on one channel
- **Bursty** — short bursts with long quiet gaps
- **Frequency-agile** — hops between channels over time

You start with **no cheat sheet**. The system must learn only from its own scan history.

---

## What this project is **not**

To avoid confusion:

| This project | Not this |
|--------------|----------|
| Predicts “will this channel be active next?” | **Anomaly detection** (finding weird one-off events) |
| A teaching / research simulator | Real hardware or live RF scanning |
| Smart scheduling | Hacking, jamming, or intercepting real communications |

---

## How a single “scan” works

Each time step, the loop looks like this:

```
1. Scheduler picks ONE band to watch
2. Receiver looks at that band for one moment
3. Detector reports what it thinks it saw (imperfectly!)
4. Stats are updated from that report
5. (Smart mode) ML guesses which bands look promising next
6. Go back to step 1
```

### The detector is intentionally imperfect

Real sensors make mistakes. This simulation models that:

- **Hit** — signal was there, and we noticed ✓
- **Miss** — signal was there, but we missed it ✗
- **False alarm** — nothing was there, but we thought we saw something ✗
- **Correct negative** — nothing there, and we correctly saw nothing ✓

By default, the detector catches about **85%** of real signals and falsely alarms about **5%** of the time when nothing is there. That makes the problem realistic: even a smart strategy has to work with noisy information.

---

## What does the “machine learning” part do?

The ML piece is **not** magic and **not** anomaly detection.

It does one specific job:

> **Given what we have seen so far on each band, how likely is that band to be active on the next time step?**

### What the model learns from

For each band, it tracks simple history, for example:

- How many times have we scanned this band?
- How many hits and misses?
- How long since we last saw activity?
- Has activity been picking up lately?

All of this comes **only from past observations** — the model never peeks at the future.

### How it is trained

1. The simulator builds a fake radio environment (who transmits when).
2. A training pass sweeps through all bands in order, collecting “what we knew then” vs “what actually happened next.”
3. A **Random Forest** classifier learns patterns from that history.
4. During a real run, the smart scheduler uses those predictions to rank bands.

Think of it like learning that “channel 7 tends to go active every few minutes” after watching for a while — except the patterns can be more complex.

---

## How the “smart” scheduler decides

For each band, it computes a **priority score** from three ingredients:

1. **Prediction** — the ML model’s guess that the band will be active soon
2. **Recent activity** — bands that have been hot lately stay interesting
3. **Exploration bonus** — bands we have ignored for a long time get another look

Most of the time it picks the highest-scoring band. But sometimes (about 20% by default) it picks **at random on purpose** — so it does not get stuck only watching channels it already “thinks” it understands.

This balance is called **exploit vs explore**:

- **Exploit** — trust what you have learned
- **Explore** — keep checking places you might have wrong

---

## How we know which strategy “wins”

After running all three strategies on the **same** fake environment (fair comparison), the project measures things like:

| Metric | What it means in plain English |
|--------|-------------------------------|
| **Interception rate** | Of all real transmission events, how many did we catch at least once? |
| **Average detection delay** | When we did catch something, how long did we take? |
| **Miss rate** | How much activity did we completely fail to notice? |
| **False alarm rate** | How often did we cry wolf when nothing was there? |

The goal of the smart strategy is to **catch more transmissions, sooner**, without going wild with false alarms.

---

## What’s inside the project (folder tour)

You do not need to read code to use the idea, but here is what each part represents:

| Folder / file | Plain English role |
|---------------|-------------------|
| `simulation/` | Creates the fake world: emitters, environment, receiver |
| `detection/` | Makes observations imperfect (hits, misses, false alarms) |
| `scheduler/` | Decides which band to scan next (sequential, random, or smart) |
| `ml/` | Builds features, trains the model, makes predictions |
| `evaluation/` | Scores how well each strategy performed |
| `visualization/` | Draws heatmaps so you can *see* activity over time |
| `app.py` | Web dashboard to run everything with buttons |

---

## Try it yourself (quick start)

```bash
cd smart-scan-strategy-complete
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then in the browser sidebar:

1. **Generate Environment** — create the fake radio world
2. **Run Sequential / Random Simulation** — run the simple baselines
3. **Train ML Model** — teach the predictor from scan history
4. **Run Smart Simulation** — run the learning-based scheduler
5. **Run Comparison** — see all three side by side

Open the heatmap tab to watch activity like a calendar: bands on one axis, time on the other, colored by whether someone was transmitting.

---

## Key takeaways

1. **One receiver, many bands** — the hard part is *where to look next*, not whether looking is possible.
2. **Three strategies** — round-robin, random, and ML-guided with exploration.
3. **Learning from history** — the smart approach improves as it accumulates scan results.
4. **Imperfect sensing** — even good plans fail sometimes because detectors miss things or false-alarm.
5. **Simulation only** — results show whether the *idea* works in a controlled fake world, not on real airwaves.

---

## Where to go next

- **Technical details:** see [README.md](README.md)
- **Phase-by-phase tests:** run scripts in `tests/` to see each piece validated step by step
- **Tune behavior:** defaults live in `config.py` (number of bands, detector accuracy, exploration rate, etc.)

---

## Glossary (jargon → plain English)

| Term | Plain English |
|------|---------------|
| **Band / frequency band** | One radio “channel” the receiver can tune to |
| **Emitter** | A fake signal source transmitting on some schedule |
| **Dwell time** | How long the receiver stays on one band before moving |
| **Pd** | Probability of detection — how often we catch a real signal |
| **Pfa** | False alarm rate — how often we report a signal when none exists |
| **Random Forest** | A common ML method that combines many simple decision trees |
| **Scheduler** | The brain that picks the next band to scan |
| **Ground truth** | What *actually* happened (known in simulation, hidden from the scheduler) |
