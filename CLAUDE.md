# CLAUDE.md — DispatchPulse Project Context

**Read this file before doing anything in this repo.** It is the single source of truth for what DispatchPulse is, what state it's in, and what the rules of engagement are for any Claude session working on it.

---

## 1. What this project is

**DispatchPulse** is a submission for the **Meta PyTorch OpenEnv Hackathon — India 2026**, hosted by Scaler School of Technology in Bangalore in partnership with Meta, Hugging Face, and the PyTorch Foundation.

It is an **OpenEnv reinforcement-learning environment** where an LLM agent plays the role of a 911 emergency dispatch coordinator. The agent receives incoming emergency calls, triages them, and dispatches limited units (ALS/BLS ambulances, fire engines, police) under time pressure. Patient outcomes are scored using **real clinical survival curves** from EMS literature — no LLM-as-judge, deterministic math only.

The pitch: in India, 24,000+ people die daily from slow emergency response. Average ambulance time is 25–35 minutes, well beyond the golden hour. DispatchPulse simulates this crisis as a learnable RL environment where greedy "closest-unit" strategies provably fail.

### Hackathon details
- **Owner:** Arun Sanjay (Bangalore, India). HF username: `Arun-Sanjay`. GitHub username: `Arun-Sanjay`.
- **Round 1 deadline:** April 8, 2026 (already submitted, automated grading in progress at time of writing)
- **Round 2 finale:** April 25-26, 2026 — 48-hour in-person hackathon at Scaler campus, Bangalore
- **Prize pool:** $30,000 total (top 3: $10K each + Meta/HF interview; top 8: $2K each; top 15: $650 each)
- **Problem statement:** "Build a real-world OpenEnv environment that an AI agent can learn from through the standard step()/reset()/state() API. Minimum 3 tasks with graders, scores 0.0–1.0, deployed to HF Spaces + Docker."

---

## 2. Where the project lives

| Location | URL / Path |
|---|---|
| Local working dir | `/Users/arunsanjay/Documents/Projects/DispatchPulse` |
| GitHub repo (public) | https://github.com/Arun-Sanjay/DispatchPulse |
| HF Space (live) | https://huggingface.co/spaces/Arun-Sanjay/dispatchpulse |
| HF Space API base | https://arun-sanjay-dispatchpulse.hf.space |
| Python venv | `.venv/` (Python 3.11.15, uv-managed) |
| Two git remotes | `origin` → HF Space, `github` → GitHub |

---

## 3. Submission status (as of end of Round 1 day)

**Round 1:** Submission #3 submitted on 8 April 2026 at 20:02 IST.
- ✅ **Phase 1: PASSED** (OpenEnv Reset 200, Dockerfile at root, inference.py at root, openenv validate green — `uv.lock` commit fixed this)
- 🟡 **Phase 2: IN PROGRESS** (30-40 min wait, email notification when done)
- 5 Phase 2 checks: Docker Build Creation, inference.py Execution, Output Parsing, Task Validation, LLM Criteria Check

**If Phase 2 passes** — we're done. Wait for Round 2 invites on April 10.
**If Phase 2 fails** — the email will say which of the 5 checks failed. Fix only the specific failure. Do NOT speculatively refactor.

### Known fixes already applied (do not re-apply)
1. `uv.lock` added to fix "Missing uv.lock" on `openenv validate`
2. `EXPOSE 8000` + `app_port: 8000` in README frontmatter to route HF Space traffic correctly
3. `GET /tasks`, `GET /tasks/{task_id}`, `POST /grader` endpoints added to `server/app.py` (fix for "Not enough tasks with graders")
4. `openenv.yaml` declares all 3 tasks explicitly with `has_grader: true`
5. MCP-based environment (original design) fully replaced with canonical `Environment` base-class subclass per OpenEnv spec — see commit `64d56f9`
6. `inference.py` rewritten to match sample exactly: reads `API_BASE_URL` / `MODEL_NAME` / `HF_TOKEN` / `LOCAL_IMAGE_NAME`, uses `from openai import OpenAI`, emits `[START]/[STEP]/[END]` with 2-decimal rewards and 3-decimal score

---

## 4. Project architecture

```
DispatchPulse/
├── CLAUDE.md               ← THIS FILE — read first
├── README.md               ← HF Space landing page (has sdk/app_port frontmatter)
├── Dockerfile              ← Multi-stage build on ghcr.io/meta-pytorch/openenv-base
├── openenv.yaml            ← Manifest + tasks[] with has_grader: true
├── pyproject.toml          ← uv-managed deps
├── uv.lock                 ← REQUIRED — do not delete
├── .python-version         ← 3.11
│
├── inference.py            ← ROUND 1 ENTRY POINT — at project root, exact spec format
│
├── models.py               ← DispatchPulseAction / Observation / State inherit from
│                             openenv.core.env_server.types base classes
│                             + internal sim models (EmergencyCall, Unit, Hospital, etc.)
│
├── client.py               ← DispatchPulseEnv(EnvClient) — async WebSocket client
│                             with _step_payload / _parse_result / _parse_state
│
├── simulation.py           ← DispatchSimulation engine (pure Python, deterministic, seeded)
├── reward.py               ← 7 clinical survival curves + episode reward composer
├── grader.py               ← grade_simulation(sim) -> Reward
├── scenario_loader.py      ← YAML → scenario dict loader
├── text_view.py            ← Renders dispatch center as LLM-readable text
├── utils.py                ← Distance, ETA, caller templates, unit lookups
├── __init__.py             ← Re-exports DispatchPulseEnv / Action / Observation / State
│
├── server/
│   ├── __init__.py
│   ├── environment.py      ← DispatchPulseEnvironment(Environment[Action, Obs, State])
│   │                         implements reset() / step() / state property
│   └── app.py              ← create_app(...) + GET /tasks + POST /grader + GET /tasks/{id}
│
├── tasks/
│   ├── easy.yaml           ← 5 calls, 4 units, 1 hospital, 30 min, 0% inaccuracy
│   ├── medium.yaml         ← 15 calls, 6 units, 2 hospitals, 45 min, 20% inaccuracy
│   └── hard.yaml           ← 30 calls, 8 units, 3 hospitals, 60 min, 35% inaccuracy
│
├── scripts/
│   └── validate-submission.sh   ← mirrors the hackathon's 3 validator checks
│
├── tests/
│   ├── __init__.py
│   ├── test_reward.py      ← 11 calibration tests for survival curves + hospital modifier
│   └── test_simulation.py  ← 10 tests for sim engine (run with .venv/bin/python directly,
│                             not pytest — module-level entry points)
│
└── baseline_results.json   ← Heuristic baseline: easy=0.55, medium=0.38, hard=0.22
```

### Critical architecture facts
1. **Use the canonical OpenEnv pattern.** `Environment` base class, NOT `MCPEnvironment`. Typed Pydantic `Action`/`Observation`/`State` inheriting from `openenv.core.env_server.types`. `EnvClient` for the client, NOT `MCPToolClient`.
2. **`server/app.py` uses `create_app(...)`** (not `create_fastapi_app` — the former enables the Gradio UI at `/` when `ENABLE_WEB_INTERFACE=true`, which the Dockerfile sets). On top of that we add `/tasks`, `/tasks/{task_id}`, and `/grader` routes with `@app.get()` / `@app.post()`.
3. **The simulation is pure Python math.** No ML libraries. No embeddings. No GPU. Runs on 2 vCPU / 8 GB RAM. Do NOT add heavy dependencies.
4. **The reward function is the moat.** Cardiac ~10%/min decay, trauma golden hour sigmoid, stroke neural decay. Cited from real EMS literature. Do NOT tweak these values — calibration tests depend on them.
5. **Deterministic with `seed=42`.** All randomness goes through `np.random.RandomState(seed)`. Same seed + same actions → identical outcome. Grader relies on this.
6. **Three scoring tiers always stratify clean:** easy > medium > hard. Every code change must preserve this monotonic decrease. Run `.venv/bin/python tests/test_simulation.py` to verify.

---

## 5. Action / Observation / State interface

### `DispatchPulseAction`
```python
class DispatchPulseAction(Action):  # inherits openenv.core.env_server.types.Action
    action_type: str          # "dispatch" | "classify" | "callback" | "wait" | "view"
    text: str = ""            # free-text form e.g. "dispatch CALL-001 ALS-1 H1"
    call_id: Optional[str]    # for dispatch/classify/callback
    unit_id: Optional[str]    # for dispatch
    hospital_id: Optional[str]  # for dispatch (optional destination)
    severity: Optional[int]   # 1-5 for classify
    message: Optional[str]    # for callback (the question)
    minutes: Optional[int]    # 1-5 for wait
```

Server-side `DispatchPulseEnvironment.step()` accepts either structured fields OR the `text` field (parsed via `_parse_text_action()` helper — same grammar as the `inference.py` parser).

### `DispatchPulseObservation`
```python
class DispatchPulseObservation(Observation):  # inherits openenv ...types.Observation
    text: str                      # rendered dispatch center (LLM reads this)
    current_time: int              # simulation minute
    time_limit: int
    calls_pending: int
    units_available: int
    calls_completed: int
    calls_timed_out: int
    total_calls: int
    last_action_error: Optional[str]
    info_message: Optional[str]
    # inherited: done, reward, metadata
```

### `DispatchPulseState`
Episode-level snapshot: `current_time`, `episode_done`, task/seed IDs, counters, `running_reward`. Returned by `GET /state`.

---

## 6. Reward function — the science

```
total = 0.60 * survival_score
      + 0.15 * efficiency_score
      + 0.15 * triage_accuracy
      - 0.10 * penalty
```

All four components in [0, 1]. Final `total` clamped to [0, 1].

### Survival curves (`reward.py`)
| Emergency | Formula | Sanity points |
|---|---|---|
| Cardiac arrest | `max(0, 0.95 * exp(-0.10 * (t - 1)))` for t > 1 | 1min=0.95, 4min=0.70, 12min=0.31 |
| Trauma | Sigmoid centered at 45 min: `max(0.05, 0.95 / (1 + exp(0.08*(t-45))))` | 5min=0.95, 45min~0.50, 90min=0.10 |
| Stroke | `max(0.05, 0.92 * exp(-0.04 * (t - 2)))` | 2min=0.92, 20min=0.42, 60min=0.09 |
| Fire | `max(0.02, 0.90 * exp(-0.15 * (t - 2)))` | 2min=0.90, 8min=0.36 |
| Breathing | Gentler exponential, max 0.10 floor | |
| Minor injury | Nearly flat: `max(0.50, 0.98 - 0.005 * t)` | |
| Mental health | Gentler exponential, 0.20 floor | |

Each call's outcome is multiplied by:
- **Unit effectiveness multiplier** (e.g. ALS→cardiac=1.0, BLS→cardiac=0.5, fire→cardiac=0.1)
- **Hospital modifier** (specialty match: ×1.05, on diversion or zero beds: ×0.85, else 1.0)

### Severity weights (inside survival_score)
Critical (1): ×3.0, Urgent (2): ×2.0, Moderate (3): ×1.5, Low (4): ×1.0, False alarm (5): ×0.5

### Anti-loophole rules
- A silent `wait`-only agent must score <0.15 on every task. If a refactor breaks this, `tests/test_reward.py::test_silent_agent_scores_near_zero` will catch it.
- `triage_accuracy` returns 0.0 (not 1.0) if critical calls existed but none were dispatched. Closes the silent-agent loophole.

---

## 7. `inference.py` contract — DO NOT BREAK

This is the file the grader actually executes. Every rule below is load-bearing.

### Mandatory environment variables
| Variable | Default in script | Notes |
|---|---|---|
| `API_BASE_URL` | `https://router.huggingface.co/v1` | OpenAI-compatible endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | HF router model id |
| `HF_TOKEN` | **no default** | Must come from env, per spec |
| `LOCAL_IMAGE_NAME` | no default | Set by the grader for `from_docker_image()` |
| `ENV_BASE_URL` | optional override for remote URL | Used in fallback mode |
| `DISPATCHPULSE_TASK` | `easy` | Which task to run |

### Mandatory stdout format (exact)
```
[START] task=<task_name> env=dispatchpulse model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
...
[END] success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
```

Rules (violating any of these zeros the score):
- Exactly one `[START]`, one `[STEP]` per step after `env.step()` returns, one `[END]` ALWAYS (even on exception — use `try/finally`)
- `reward` / `rewards` formatted to **2 decimal places**
- `score` formatted to **3 decimal places**
- `done` / `success` are **lowercase** `true` / `false`
- `error` is the raw last action error string, or literal `null`
- All fields on a single line with no embedded newlines
- Each task must return score in [0, 1]

### Mandatory code requirements
- File must be at project root
- File must be named exactly `inference.py`
- Must use `from openai import OpenAI` for ALL LLM calls — no other client libraries
- Must call `await env.close()` before emitting `[END]`

### Connection logic (current)
1. If `LOCAL_IMAGE_NAME` is set → `await DispatchPulseEnv.from_docker_image(LOCAL_IMAGE_NAME)`
2. Else if `ENV_BASE_URL` is set → direct `DispatchPulseEnv(base_url=ENV_BASE_URL)`
3. Else → in-process `_LocalInProcessEnv` fallback (for offline dev)

---

## 8. Rules of engagement for any Claude session

### Golden rules
1. **Read this file before every substantive change.** The architecture decisions here are load-bearing and non-obvious.
2. **Read `.claude/skills/dispatchpulse/SKILL.md`** for the operational playbook (how to test, deploy, debug, push).
3. **Never push to GitHub or HF Space yourself.** Arun does all git pushes manually. Your job is to prepare commits locally and hand him the exact `git push` command with a placeholder for the token. Never suggest embedding a token you were given in chat.
4. **Never handle credentials directly.** If Arun pastes a token in chat, immediately warn him, tell him to revoke it, and refuse to use it. Tokens belong in shell prompts, not in chat.
5. **Never modify the sim engine, reward formulas, or survival curves without running `tests/test_reward.py` and `tests/test_simulation.py` first AND after.** Both must stay green.
6. **Never add heavy dependencies.** No torch, no transformers, no embedding libraries. Pure Python + numpy + pydantic + fastapi + openenv-core + openai + pyyaml. That's it.
7. **Never rename or move `inference.py`** — it must stay at the project root.
8. **Never delete `uv.lock`** — the Phase 1 validator requires it.
9. **Do NOT preemptively refactor things that are working.** The current submission passes Phase 1. If Phase 2 fails, the email will name the specific check. Fix only that check. Broad refactors risk breaking what's working.

### Commit discipline
- Commit in small, logical chunks with descriptive messages
- Each commit should compile and tests should pass
- Commit messages should describe the WHY, not just the WHAT
- Format: `<Verb> <thing>: <short why>` on line 1, blank line, then details
- Never use `git push --force` without an explicit reason
- Never commit tokens, API keys, or `.env` files

### Testing
```bash
# Unit tests — run these after ANY code change
.venv/bin/python tests/test_reward.py
.venv/bin/python tests/test_simulation.py

# FastAPI server smoke test
.venv/bin/uvicorn server.app:app --host 127.0.0.1 --port 8765 &
curl -sf http://127.0.0.1:8765/health
curl -sf http://127.0.0.1:8765/tasks | python3 -m json.tool
curl -sf -X POST http://127.0.0.1:8765/reset -H "Content-Type: application/json" -d '{"task_name":"easy","seed":42}'

# Validator (mirrors grader's 3 checks)
./scripts/validate-submission.sh https://arun-sanjay-dispatchpulse.hf.space .

# inference.py smoke test (uses in-process fallback, no API key needed)
DISPATCHPULSE_TASK=easy .venv/bin/python inference.py 2>&1 | head -20
```

### Git remotes (already configured)
```bash
origin  https://huggingface.co/spaces/Arun-Sanjay/dispatchpulse   # HF Space
github  https://github.com/Arun-Sanjay/DispatchPulse.git           # GitHub
```

### Push commands to hand to Arun (he runs them, you don't)
```bash
# HF Space push
git push https://Arun-Sanjay:PASTE_HF_TOKEN@huggingface.co/spaces/Arun-Sanjay/dispatchpulse main

# GitHub push
git push https://Arun-Sanjay:PASTE_GH_TOKEN@github.com/Arun-Sanjay/DispatchPulse.git main
```

Always remind him to **revoke both tokens immediately after** at:
- https://huggingface.co/settings/tokens
- https://github.com/settings/tokens

---

## 9. Known risks and failure modes (ranked)

1. **LLM action parse failures.** If the judge's LLM wraps actions in markdown fences or prefixes ("Action:"), our `parse_action_text()` in `inference.py` falls back to `wait 1`. Mitigation: if Phase 2 fails on output parsing or low LLM score, add lenient parsing that strips markdown, handles prefixes, and tolerates trailing punctuation.
2. **`from_docker_image` never tested end-to-end locally.** I don't have Docker. The fallback path works. If the grader's docker hand-off fails, we'll see it in the Phase 2 email and can target a fix.
3. **Task discovery mechanism is HTTP-based (`GET /tasks`).** If the validator parses source code instead of hitting HTTP, it might not find our registry. Mitigation: add a Python-level `TASKS = {...}` dict in a `task_definitions.py` module that mirrors the passing Calendar Scheduling submission.
4. **Context window on hard task.** 30 calls × text view could balloon the LLM's prompt. `text_view.py` already truncates to top 8 pending calls. Don't remove this cap.
5. **Concurrency.** `SUPPORTS_CONCURRENT_SESSIONS = True` is set on the Environment class, and `max_concurrent_envs=8` on the app. Good for parallel graders.

---

## 10. What "done" looks like

- Phase 1 GREEN ✅ (already done)
- Phase 2 GREEN ✅ (in progress, waiting for email)
- Email from Scaler confirming Round 1 submission accepted
- Round 1 results announced April 10, 2026
- If selected: Round 2 finale April 25-26 in Bangalore (home turf for Arun)

If Phase 2 fails, apply targeted fix, resubmit, iterate until green. The form explicitly allows re-submissions: *"You can re-submit until your submission passes all automated checks. We always evaluate your latest submission."*

---

## 11. Quick reference — commands I reach for most

```bash
# Where am I?
cd /Users/arunsanjay/Documents/Projects/DispatchPulse

# What's the state?
git log --oneline | head -10
git status
git remote -v

# Test everything locally
.venv/bin/python tests/test_reward.py && .venv/bin/python tests/test_simulation.py

# Check the live Space
curl -sf https://arun-sanjay-dispatchpulse.hf.space/health
curl -sf https://arun-sanjay-dispatchpulse.hf.space/tasks | python3 -m json.tool
curl -sf -X POST https://arun-sanjay-dispatchpulse.hf.space/grader \
  -H "Content-Type: application/json" -d '{"task_id":"easy","seed":42}' | python3 -m json.tool

# Check HF Space runtime state
curl -sf https://huggingface.co/api/spaces/Arun-Sanjay/dispatchpulse \
  | python3 -c "import sys,json; d=json.load(sys.stdin); rt=d.get('runtime',{}); print('stage:', rt.get('stage'), 'sha:', rt.get('sha'))"

# Run the pre-submission validator
./scripts/validate-submission.sh https://arun-sanjay-dispatchpulse.hf.space .
```

---

**End of CLAUDE.md.** If anything in this file is stale, update it before doing other work.
