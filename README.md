---
title: DispatchPulse
emoji: 🚑
colorFrom: red
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
license: apache-2.0
---

# DispatchPulse

**An OpenEnv environment where an AI agent acts as a 911 emergency dispatch coordinator.**
The agent receives incoming calls, classifies their severity, and dispatches limited
emergency units (ALS / BLS ambulances, fire engines, police) under time pressure.
Patient outcomes are scored against **real clinical survival curves** — no
LLM-as-judge, just defensible math.

> Submission for the [Meta PyTorch OpenEnv Hackathon — India 2026](https://www.scaler.com/school-of-technology/meta-pytorch-hackathon).

---

## Why this environment

In India, an estimated 24,000+ people die every day because of slow emergency
response — average ambulance time is 25–35 minutes, well beyond the golden hour,
and only ~20% of ambulances carry advanced life support. DispatchPulse simulates
this crisis as an interactive RL environment where the agent has to learn the
*counter-intuitive* strategies real dispatchers use:

- **The greedy "closest unit" strategy fails.** Dispatching the only ALS to a
  sprained ankle leaves nothing for the cardiac arrest that arrives 3 minutes
  later — survival drops from 70% to 15%.
- **Triage matters more than speed.** A weighted reward (severity 1 calls
  count 3× more than severity 4) means the agent has to *prioritise*, not
  just react.
- **Hospital choice matters.** Sending a stroke patient to a hospital without
  a stroke unit, or to one on diversion, costs you score.

The reward function uses real clinical survival curves from the EMS literature
(Larsen et al. 1993 for cardiac arrest; Saver 2006 "Time is Brain" for stroke;
golden hour curves for trauma). It's deterministic, defensible, and gives a
continuous signal that an RL agent can actually learn from.

---

## OpenEnv compliance

| Requirement | Status |
|---|---|
| Real-world task (not games or toys) | ✅ Emergency dispatch — actual profession |
| Typed Pydantic models | ✅ `models.py` |
| `step()` / `reset()` / `state()` API | ✅ via `MCPEnvironment` base class |
| `openenv.yaml` manifest | ✅ |
| ≥ 3 tasks with graders, scores 0.0–1.0 | ✅ easy / medium / hard |
| Meaningful reward + partial progress | ✅ survival curves + per-step rewards |
| Baseline `inference.py` at root | ✅ heuristic & LLM modes |
| Reproducible (fixed seed) | ✅ `seed=42` default everywhere |
| Dockerfile + HF Spaces deploy | ✅ uses `openenv-base` |

---

## Action space (MCP tools)

DispatchPulse exposes its interface as MCP tools, the canonical OpenEnv pattern:

| Tool | Args | Time cost | What it does |
|---|---|---|---|
| `view_dispatch_center` | none | free | Returns the current dispatch center as text. |
| `dispatch` | `call_id`, `unit_id`, `hospital_id?` | 1 min | Send a unit to a call (optionally pre-routing to a destination hospital). |
| `classify` | `call_id`, `severity` | 1 min | Reclassify a call's severity (1–5). |
| `callback` | `call_id`, `question` | 1 min | Phone the caller back. 70% chance they clarify the true emergency type. |
| `wait` | `minutes` (default 1) | n min (≤5) | Skip ahead in the simulation when there's nothing to do. |

## Observation space

Each tool returns a structured text view of the dispatch center:
- Pending calls (truncated to top 8 by severity for context-window safety)
- Available units with closest-call ETAs
- Busy units with their status
- Hospital roster with bed availability and specialties
- Recent outcomes

The full text view is also available for inspection via `view_dispatch_center()`.

## Tasks

| Task | Calls | Units | Hospitals | Duration | Caller misreporting | What's hard about it |
|---|---|---|---|---|---|---|
| `easy` | 5 | 4 | 1 | 30 min | 0% | Basic dispatch — learn the action grammar |
| `medium` | 15 | 6 | 2 | 45 min | 20% | Mass casualty bus accident at minute 12; some callers lie |
| `hard` | 30 | 8 | 3 (1 on diversion) | 60 min | 35% | Earthquake response — extreme scarcity, panicked callers, hospital triage matters |

All three are deterministic given the seed.

---

## Reward function

Final episode score = weighted combination of four components, all in [0, 1]:

| Component | Weight | What it measures |
|---|---|---|
| `survival_score` | 0.60 | Severity-weighted average outcome across all calls (uses clinical survival curves × unit effectiveness × hospital modifier) |
| `efficiency_score` | 0.15 | Fraction of calls dispatched, penalised for wasting ALS on minor calls |
| `triage_accuracy` | 0.15 | Fraction of severity-1 calls dispatched within 25% of their timeout window |
| `penalty` | −0.10 | Deductions for timed-out criticals and wrong-unit assignments |

Severity weights inside the survival score: **3× for severity 1, 2× for 2, 1.5× for 3, 1× for 4, 0.5× for 5**.

### Survival curves (from EMS literature)

| Emergency | Curve | Notes |
|---|---|---|
| Cardiac arrest | exponential, ~10%/min decay | Larsen et al. 1993 |
| Trauma | sigmoid centred at 45 min | "golden hour" |
| Stroke | exponential decay | Saver 2006 — every minute = 1.9M neurons |
| Fire | exponential, doubles per minute | property loss |
| Breathing difficulty | gentler exponential | |
| Minor injury | nearly flat | stable patient |
| Mental health | gentler exponential | de-escalation success |

Each call's outcome is multiplied by:
- **Unit effectiveness** (e.g., ALS → cardiac = 1.0; BLS → cardiac = 0.5; fire engine → cardiac = 0.1)
- **Hospital modifier** (specialty match: +5%; on diversion or zero beds: −15%)

---

## Baseline scores

Both runs use the same fixed seed (`42`) and are reproducible.

### Heuristic agent (no LLM, just rule-based triage)

| Task | Total | Survival | Efficiency | Triage | Penalty |
|---|---|---|---|---|---|
| easy   | 0.5476 | 0.463 | 0.800 | 1.000 | −0.000 |
| medium | 0.3750 | 0.377 | 0.600 | 0.500 | −0.160 |
| hard   | 0.2183 | 0.214 | 0.433 | 0.500 | −0.500 |
| **Average** | **0.3803** | | | | |

The clean monotonic decrease across difficulty (easy > medium > hard) confirms
the env discriminates between scenarios as designed.

### LLM agent (GPT-4o-mini, default settings)

Run with: `OPENAI_API_KEY=sk-... python inference.py --agent llm --model gpt-4o-mini`

Run on your own infrastructure to compare against the heuristic baseline.

---

## Setup

### Run locally with Python

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
python inference.py --agent heuristic
```

### Run locally with Docker

```bash
docker build -t dispatchpulse .
docker run -p 8000:8000 dispatchpulse
# Then in another shell:
curl http://localhost:8000/health
```

### Use as a client (OpenEnv MCPToolClient pattern)

```python
from client import DispatchPulseEnv

with DispatchPulseEnv(base_url="http://localhost:8000") as env:
    env.reset(task_name="easy", seed=42)
    print(env.call_tool("view_dispatch_center"))
    env.call_tool("dispatch", call_id="CALL-001", unit_id="BLS-1", hospital_id="H1")
    env.call_tool("wait", minutes=1)
```

### Run on Hugging Face Spaces

This repo is auto-built as a Docker Space. Visit:
[`https://huggingface.co/spaces/Arun-Sanjay/dispatchpulse`](https://huggingface.co/spaces/Arun-Sanjay/dispatchpulse)

---

## Calibration tests

The reward function ships with calibration tests that double as documentation:

```bash
python tests/test_reward.py
python tests/test_simulation.py
```

These verify that:
- Survival curves match published clinical numbers
- A "do-nothing" agent scores below 0.15 on every task
- The heuristic agent strictly outperforms the silent agent
- Heuristic scores monotonically decrease easy → medium → hard
- ALS at cardiac arrest beats fire engine at cardiac arrest by ≥5×
- Specialty hospital match boosts outcome; diversion hurts it

---

## Project layout

```
DispatchPulse/
├── README.md                # this file
├── Dockerfile               # uses openenv-base image
├── openenv.yaml             # OpenEnv manifest
├── pyproject.toml
├── inference.py             # ROUND 1 REQUIRED: baseline runner with --agent llm/heuristic
├── client.py                # DispatchPulseEnv (extends MCPToolClient)
├── models.py                # Pydantic models (Position, EmergencyCall, ...)
├── simulation.py            # DispatchSimulation engine
├── reward.py                # Survival curves + episode reward
├── grader.py                # Programmatic 0.0–1.0 grader
├── scenario_loader.py       # YAML task loader
├── text_view.py             # LLM-friendly dispatch center renderer
├── utils.py                 # Distance/ETA/templates
├── server/
│   ├── app.py               # FastAPI app via openenv create_app
│   └── dispatchpulse_environment.py  # MCPEnvironment subclass
├── tasks/
│   ├── easy.yaml
│   ├── medium.yaml
│   └── hard.yaml
└── tests/
    ├── test_reward.py
    └── test_simulation.py
```

---

## License

Apache 2.0. Built for the Meta PyTorch OpenEnv Hackathon — India 2026.
