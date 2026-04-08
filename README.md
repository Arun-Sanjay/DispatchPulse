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
continuous signal an RL agent can actually learn from.

---

## OpenEnv compliance

| Requirement | Status |
|---|---|
| Real-world task (not games or toys) | ✅ Emergency dispatch — actual profession |
| Typed Pydantic models inheriting from OpenEnv `Action` / `Observation` / `State` | ✅ `models.py` |
| `Environment` base-class subclass with `reset()` / `step()` / `state` | ✅ `server/environment.py` |
| FastAPI server via `create_fastapi_app(...)` | ✅ `server/app.py` |
| `EnvClient` client with `_step_payload` / `_parse_result` / `_parse_state` | ✅ `client.py` |
| `openenv.yaml` manifest | ✅ |
| ≥ 3 tasks with graders, scores 0.0–1.0 | ✅ easy / medium / hard |
| Meaningful reward + partial progress | ✅ survival curves + per-step rewards |
| `inference.py` at root, OpenAI client, mandatory env vars, `[START]/[STEP]/[END]` format | ✅ |
| Reproducible (fixed seed) | ✅ `seed=42` default everywhere |
| Pre-submission validator script | ✅ `scripts/validate-submission.sh` |
| Dockerfile + HF Spaces deploy | ✅ uses `openenv-base` |
| Runs on 2 vCPU / 8 GB RAM | ✅ pure Python math, no ML inference |

---

## Project layout (canonical OpenEnv structure)

```
DispatchPulse/
├── README.md
├── Dockerfile               # uses ghcr.io/meta-pytorch/openenv-base
├── openenv.yaml             # OpenEnv manifest
├── pyproject.toml
├── inference.py             # ROUND 1 ENTRY POINT — must be in root
├── client.py                # DispatchPulseEnv (subclass of EnvClient)
├── models.py                # DispatchPulseAction / Observation / State
│                            # plus internal sim models
├── simulation.py            # DispatchSimulation engine
├── reward.py                # Survival curves + episode reward
├── grader.py                # Programmatic 0.0–1.0 grader
├── scenario_loader.py       # YAML task loader
├── text_view.py             # LLM-friendly dispatch center renderer
├── utils.py                 # Distance / ETA / templates
├── server/
│   ├── __init__.py
│   ├── app.py               # FastAPI app via create_fastapi_app(...)
│   └── environment.py       # DispatchPulseEnvironment(Environment)
├── tasks/
│   ├── easy.yaml
│   ├── medium.yaml
│   └── hard.yaml
├── scripts/
│   └── validate-submission.sh   # runs the 3 grader checks locally
└── tests/
    ├── test_reward.py
    └── test_simulation.py
```

---

## Action space (typed Pydantic)

`DispatchPulseAction` has these `action_type` values:

| `action_type` | Required fields | Time cost | What it does |
|---|---|---|---|
| `dispatch` | `call_id`, `unit_id`, `hospital_id?` | 1 min | Send a unit to a call (optionally pre-routing to a hospital). |
| `classify` | `call_id`, `severity` (1-5) | 1 min | Reclassify a call's severity. |
| `callback` | `call_id`, `message` | 1 min | Phone the caller back. 70% chance they clarify the true emergency type. |
| `wait` | `minutes` (default 1, max 5) | n min | Skip ahead in the simulation when there's nothing to do. |
| `view` | — | free | Re-fetch the dispatch center text without advancing time. |

The action also has a free-text `text` field — the server parses lines like
`dispatch CALL-001 ALS-1 H1` so an LLM can produce them directly.

## Observation space

`DispatchPulseObservation` has:

- `text` — formatted dispatch center view (the field the LLM reads)
- `current_time`, `time_limit`
- `calls_pending`, `units_available`, `calls_completed`, `calls_timed_out`, `total_calls`
- `last_action_error` — error string from the previous action, or `None`
- `info_message` — what just happened
- inherited `done`, `reward`, `metadata`

## Tasks

| Task | Calls | Units | Hospitals | Duration | Caller misreporting | What's hard about it |
|---|---|---|---|---|---|---|
| `easy` | 5 | 4 | 1 | 30 min | 0% | Basic dispatch — learn the action grammar |
| `medium` | 15 | 6 | 2 | 45 min | 20% | Mass casualty bus accident at minute 12; some callers lie |
| `hard` | 30 | 8 | 3 (1 on diversion) | 60 min | 35% | Earthquake — extreme scarcity, panicked callers, hospital triage matters |

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

| Emergency | Curve | Source / notes |
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

## Baseline scores (heuristic agent, seed=42)

A simple rule-based heuristic (always pick the most-critical call, send the
most effective available unit, reserve ALS for high-severity calls) produces
the following calibrated scores:

| Task | Total | Survival | Efficiency | Triage | Penalty | Completed/Total |
|---|---|---|---|---|---|---|
| easy   | 0.5476 | 0.463 | 0.800 | 1.000 | −0.000 | 4/5 |
| medium | 0.3750 | 0.377 | 0.600 | 0.500 | −0.160 | 9/15 |
| hard   | 0.2183 | 0.214 | 0.433 | 0.500 | −0.500 | 13/30 |
| **Average** | **0.3803** | | | | | |

The clean monotonic decrease across difficulty (easy > medium > hard) confirms
the env discriminates between scenarios as designed.

---

## Inference script — `inference.py`

Per the hackathon spec, `inference.py` is in the **project root** and follows
the mandatory contract:

### Required environment variables

| Variable | Purpose | Default in script |
|---|---|---|
| `API_BASE_URL` | LLM endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Which model to call | `Qwen/Qwen2.5-72B-Instruct` |
| `HF_TOKEN` | API key for the LLM | (no default) |
| `LOCAL_IMAGE_NAME` | Docker image for `from_docker_image()` | (no default) |
| `DISPATCHPULSE_TASK` | Which task to run (`easy`/`medium`/`hard`) | `easy` |

### Stdout format (verbatim)

```
[START] task=<task_name> env=dispatchpulse model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
```

- One `[START]` line at episode begin
- One `[STEP]` line per step, immediately after `env.step()` returns
- One `[END]` line after `env.close()`, ALWAYS emitted (even on exception)
- `reward` and `rewards` to 2 decimal places; `score` to 3 decimal places
- `done` and `success` are lowercase booleans

### Connection logic

1. If `LOCAL_IMAGE_NAME` is set → `await DispatchPulseEnv.from_docker_image(LOCAL_IMAGE_NAME)`
2. Else if `ENV_BASE_URL` is set → connect directly to a running env server
3. Otherwise → spin up an in-process simulation as a fallback (for offline runs)

### Run it

```bash
# Against the live HF Space
ENV_BASE_URL=https://arun-sanjay-dispatchpulse.hf.space \
HF_TOKEN=$HF_TOKEN \
python inference.py

# Against a local Docker image
LOCAL_IMAGE_NAME=dispatchpulse:latest \
HF_TOKEN=$HF_TOKEN \
python inference.py

# In-process fallback (no network, no Docker)
python inference.py
```

---

## Setup

### Run locally with Python

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
python inference.py
```

### Run locally with Docker

```bash
docker build -t dispatchpulse .
docker run -p 8000:8000 dispatchpulse
# Then in another shell:
curl http://localhost:8000/health
```

### Use as a client (OpenEnv `EnvClient` pattern)

```python
import asyncio
from client import DispatchPulseEnv
from models import DispatchPulseAction

async def main():
    async with DispatchPulseEnv(base_url="https://arun-sanjay-dispatchpulse.hf.space") as env:
        result = await env.reset(task_name="easy", seed=42)
        while not result.done:
            action = DispatchPulseAction(action_type="wait", minutes=1, text="wait 1")
            result = await env.step(action)
            print(result.observation.text[:200])
        print(f"Final score: {result.reward}")

asyncio.run(main())
```

### Run on Hugging Face Spaces

Auto-built as a Docker Space:
[`https://huggingface.co/spaces/Arun-Sanjay/dispatchpulse`](https://huggingface.co/spaces/Arun-Sanjay/dispatchpulse)

---

## Pre-submission validator

Run the same three checks the hackathon's automated grader runs:

```bash
./scripts/validate-submission.sh https://arun-sanjay-dispatchpulse.hf.space .
```

It checks:
1. **HF Space deploys** — `POST /reset` returns HTTP 200
2. **Docker build** — `docker build .` succeeds (≤ 10 min)
3. **OpenEnv compliance** — `openenv validate` passes

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
- A simple heuristic strictly outperforms the silent agent
- Heuristic scores monotonically decrease easy → medium → hard
- ALS at cardiac arrest beats fire engine at cardiac arrest by ≥5×
- Specialty hospital match boosts outcome; diversion hurts it

---

## License

Apache 2.0. Built for the Meta PyTorch OpenEnv Hackathon — India 2026.
