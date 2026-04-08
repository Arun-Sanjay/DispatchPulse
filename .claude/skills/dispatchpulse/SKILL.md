---
name: dispatchpulse
description: Operational playbook for the DispatchPulse Meta PyTorch OpenEnv Hackathon submission. Load this when working in /Users/arunsanjay/Documents/Projects/DispatchPulse or when Arun mentions the hackathon, DispatchPulse, Round 1/Round 2, HF Space Arun-Sanjay/dispatchpulse, or the GitHub repo Arun-Sanjay/DispatchPulse. Covers how to test, debug, deploy, and respond to Phase 1/Phase 2 validator failures without breaking the existing passing submission.
---

# DispatchPulse Operations Skill

**Scope:** This skill is the *how-to* companion to `CLAUDE.md`. CLAUDE.md tells you WHAT the project is. This skill tells you WHAT TO DO — step-by-step playbooks for every common scenario.

**First action in any session:** read `CLAUDE.md` at the project root. Then find your scenario in section 2 below and follow it. Do not improvise — every command here is tested.

---

## 1. Session bootstrap (do this every time)

```bash
cd /Users/arunsanjay/Documents/Projects/DispatchPulse

# Sanity check: working tree clean, on main, venv exists
git status
git log --oneline | head -5
git remote -v
ls .venv/bin/python
```

Expected state:
- `On branch main`, `nothing to commit, working tree clean`
- Latest commit should be `82ce364 Fix Phase 2: add GET /tasks and POST /grader endpoints` (or whatever the user has pushed since)
- Two remotes: `origin` (HF Space) and `github` (GitHub)
- `.venv/bin/python` exists and is Python 3.11

If any of those is wrong, stop and ask Arun what state he's expecting.

Quick smoke test that nothing is broken:
```bash
.venv/bin/python tests/test_reward.py && .venv/bin/python tests/test_simulation.py
# Expected: "All reward tests passed!" and "All simulation tests passed!"
```

If tests don't pass, do NOT start work. Ask Arun what changed.

---

## 2. Scenario playbooks

### 2.1 "Phase 2 passed — we're done"

Actions:
1. Congratulate Arun. Don't push any code.
2. Explicitly tell him: *"Do not push any more commits to either remote until Round 1 results on April 10. The submitted version is frozen and working."*
3. Offer to stash any uncommitted experiments on a branch for Round 2 prep.
4. Remind him to take a screenshot of the success page for his records.

### 2.2 "Phase 2 failed — which check?"

First step: get the exact failure reason from Arun. The Scaler email names one of these 5 checks:
- **Docker Build Creation** → go to 2.3
- **inference.py Execution** → go to 2.4
- **Output Parsing** → go to 2.5
- **Task Validation** → go to 2.6
- **LLM Criteria Check** → go to 2.7

Do NOT speculatively fix multiple checks at once. Fix exactly the one that failed, verify locally, push, resubmit. If a second check fails after the first is fixed, handle it then.

### 2.3 Docker Build Creation failed

Most likely cause: the grader's `docker build .` ran out of time, couldn't pull the base image, or hit a `pip install` error with the git-installed `openenv-core`.

**Debug checklist:**
```bash
# 1. Does the Dockerfile still exist at the repo root?
ls -la Dockerfile

# 2. Is the base image pullable from GHCR?
#    (only do this if Arun has Docker — skip otherwise)
docker pull ghcr.io/meta-pytorch/openenv-base:latest 2>&1 | tail -5

# 3. Is pyproject.toml's git-based openenv-core dep still working?
#    Check by letting uv re-resolve:
.venv/bin/uv lock --upgrade-package openenv-core 2>&1 | tail -20
```

**Possible fixes (in order of safety):**
1. **Pin a specific openenv-core tag** in `pyproject.toml` if the grader can't reach `@v0.2.3`. Look up the latest stable tag on GitHub first.
2. **Add `.dockerignore`** to exclude `.venv/`, `__pycache__/`, `.git/`, `tests/` from the Docker context — faster builds, smaller context.
3. **Switch the `RUN uv sync` step to use `--frozen`** to force uv to use `uv.lock` exactly without re-resolving.

Do NOT change the base image. Do NOT switch to a plain `python:3.11-slim` base — the grader expects openenv-base.

### 2.4 inference.py Execution failed

Most likely causes:
- `from_docker_image()` call hangs or errors out
- Script takes longer than 20 minutes (grader timeout)
- LLM call raises an unhandled exception that bypasses the `[END]` emission

**Debug checklist:**
```bash
# 1. Does inference.py run locally in the in-process fallback?
DISPATCHPULSE_TASK=easy .venv/bin/python inference.py 2>&1 | head -20
# Must print [START], [STEP]s, [END] with valid format

# 2. Does it handle missing HF_TOKEN gracefully?
unset HF_TOKEN API_KEY
DISPATCHPULSE_TASK=easy .venv/bin/python inference.py 2>&1 | grep -E '^\[(START|STEP|END)\]' | tail -5

# 3. Does it still emit [END] on exception?
DISPATCHPULSE_TASK=easy MODEL_NAME="definitely-not-a-real-model" .venv/bin/python inference.py 2>&1 | grep -E '^\[END\]'
```

**Possible fixes:**
1. **Add a hard timeout** around the `env.step()` call using `asyncio.wait_for(..., timeout=30)`
2. **Wrap the entire episode loop** in try/except so the `[END]` line always fires (the current code already does this via `finally` — verify it's still intact)
3. **Shorten `MAX_STEPS`** from 60 to 40 to ensure the script always finishes in under 20 min even with slow LLMs
4. If the issue is `from_docker_image()` specifically: look at the openenv-core `LocalDockerProvider` source in `.venv/lib/python3.11/site-packages/openenv/core/containers/runtime/providers/` and mimic what a passing submission does

### 2.5 Output Parsing failed

The grader couldn't parse our stdout. Check every byte:

```bash
# Capture a real run and inspect line by line
DISPATCHPULSE_TASK=easy .venv/bin/python inference.py > /tmp/out.log 2>&1
grep -E '^\[(START|STEP|END)\]' /tmp/out.log | cat -A  # -A shows hidden chars
```

Common format bugs:
- Trailing whitespace on a `[STEP]` line (shouldn't matter but sometimes graders are strict)
- `True`/`False` instead of `true`/`false`
- `reward=0.5` instead of `reward=0.50` (missing 2nd decimal)
- `score=0.5` instead of `score=0.500` (missing 3rd decimal)
- Extra newlines inside an action field (if an LLM returns multi-line text)
- Missing `[END]` on exception paths

Check `inference.py` log_start / log_step / log_end functions. Keep them simple f-strings.

### 2.6 Task Validation failed

Grader couldn't find 3 graded tasks. Our submission exposes them THREE ways:
1. `GET /tasks` HTTP endpoint (in `server/app.py`)
2. `POST /grader` HTTP endpoint (in `server/app.py`)
3. `tasks:` list in `openenv.yaml` manifest

**Debug checklist:**
```bash
# 1. Does /tasks return 3 tasks?
curl -sf https://arun-sanjay-dispatchpulse.hf.space/tasks | python3 -m json.tool | grep has_grader

# 2. Does /grader work for each task?
for t in easy medium hard; do
  curl -sf -X POST https://arun-sanjay-dispatchpulse.hf.space/grader \
    -H "Content-Type: application/json" -d "{\"task_id\":\"$t\",\"seed\":42}" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{d[\"task_id\"]}: score={d[\"score\"]:.3f} passed={d[\"passed\"]}')"
done

# 3. Does openenv.yaml declare 3 tasks with has_grader: true?
grep -c "has_grader" openenv.yaml  # should output 3
```

**Fallback fix:** create `task_definitions.py` at project root with a Python-level `TASKS` dict that mirrors the Calendar Scheduling passing submission pattern:

```python
# task_definitions.py
from dataclasses import dataclass

@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    name: str
    difficulty: str  # "easy" | "medium" | "hard"
    description: str
    max_steps: int

TASKS = {
    "easy": TaskDefinition(
        task_id="easy", name="easy", difficulty="easy",
        description="...", max_steps=30,
    ),
    "medium": TaskDefinition(
        task_id="medium", name="medium", difficulty="medium",
        description="...", max_steps=45,
    ),
    "hard": TaskDefinition(
        task_id="hard", name="hard", difficulty="hard",
        description="...", max_steps=60,
    ),
}
```

Then import it in `server/app.py` so static analyzers can trace the symbol.

### 2.7 LLM Criteria Check failed

This is the vaguest failure mode. It means either:
- The LLM's actions couldn't be parsed (→ fix `inference.py::parse_action_text`)
- The LLM scored zero across all tasks (→ the env is too hard OR the prompt is bad)
- The env doesn't meet some "real-world task" quality bar (→ unlikely, we have clinical math)

**Fix priority:**
1. **Bulletproof action parsing** — add lenient parsing that strips markdown fences, handles `"Action: ..."` prefixes, tolerates trailing periods, accepts `"dispatch(CALL-001, ALS-1)"` function-call syntax, and regex-matches common variants. Here's a tested version:

```python
import re

def parse_action_text(text: str) -> DispatchPulseAction:
    """Lenient parser: tolerates markdown, prefixes, function call syntax."""
    text = (text or "").strip()

    # Strip markdown code fences
    text = re.sub(r"^```\w*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    text = text.strip()

    # Take first non-empty line
    for line in text.splitlines():
        line = line.strip().strip("`").strip()
        if line:
            text = line
            break

    # Strip common prefixes
    for prefix in ("Action:", "action:", "ACTION:", "Response:", "> "):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    # Strip trailing period / quotes
    text = text.rstrip(".\"' ")

    # Try function-call syntax: dispatch(CALL-001, ALS-1)
    match = re.match(r"(\w+)\s*\((.*)\)$", text)
    if match:
        fn = match.group(1).lower()
        args = [a.strip().strip("'\"") for a in match.group(2).split(",")]
        text = f"{fn} " + " ".join(args)

    # Now use the existing space-separated parser
    parts = text.split(maxsplit=4)
    # ... rest of existing logic
```

2. **Add 3 few-shot examples in the system prompt** showing observation→action pairs.
3. **Lower `temperature` to 0.0** (already done in current inference.py — verify).
4. **Verify the prompt includes all 5 action verbs** with exact grammar examples.

### 2.8 "Round 2 prep — make a branch for experiments"

Round 1 is submitted and frozen. For Round 2 prep:
```bash
git checkout -b round2-experiments
# Make experimental changes here
# Never merge back to main unless we know Round 1 is done with
```

Tell Arun he should not touch `main` until the Round 2 finale is over.

### 2.9 "Can we make upgrades without breaking the submission?"

If Round 1 is passing and Arun asks for upgrades, the rules:
- ✅ Safe: README polish, baseline result JSON updates, additional test cases, docstring improvements, comment clarifications
- ⚠️ Risky: any code change to `inference.py`, `models.py`, `server/`, `simulation.py`, `reward.py`, `grader.py`
- 🔴 Forbidden: changes to reward weights, survival curve parameters, task YAMLs (they're locked in), base Docker image

For risky changes: make them on a branch, run the full test suite, run the validator, run `inference.py` with the in-process fallback, AND open a new HF Space with a different name to test the Docker build in isolation. Only merge to `main` if Arun explicitly says so.

---

## 3. Testing reference

### Unit tests (must stay green on every commit)
```bash
.venv/bin/python tests/test_reward.py
# Expected: "All reward tests passed!"

.venv/bin/python tests/test_simulation.py
# Expected: "All simulation tests passed!"
```

These are NOT pytest — they're module-level `__main__` scripts. Do not run `pytest` on them.

### Local FastAPI smoke test
```bash
# Start server in background
ENABLE_WEB_INTERFACE=true .venv/bin/uvicorn server.app:app --host 127.0.0.1 --port 8765 > /tmp/dp.log 2>&1 &
SERVER_PID=$!
sleep 4

# Exercise every endpoint
curl -sf http://127.0.0.1:8765/health                    # {"status":"healthy"}
curl -sf http://127.0.0.1:8765/tasks | python3 -m json.tool
curl -sf http://127.0.0.1:8765/tasks/easy | python3 -m json.tool
curl -sf -X POST http://127.0.0.1:8765/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name":"easy","seed":42}' | python3 -m json.tool | head -20
curl -sf -X POST http://127.0.0.1:8765/step \
  -H "Content-Type: application/json" \
  -d '{"action":{"action_type":"wait","minutes":2,"text":"wait 2","metadata":{}}}' | python3 -m json.tool | head -20
curl -sf -X POST http://127.0.0.1:8765/grader \
  -H "Content-Type: application/json" \
  -d '{"task_id":"easy","seed":42}' | python3 -m json.tool

# Cleanup
kill $SERVER_PID 2>/dev/null
```

### Live Space smoke test
```bash
BASE=https://arun-sanjay-dispatchpulse.hf.space

curl -sf $BASE/health
curl -sf $BASE/tasks | python3 -m json.tool | head -30
curl -sf -X POST $BASE/reset -H "Content-Type: application/json" -d '{"task_name":"easy","seed":42}' | head -c 300
curl -sf -X POST $BASE/grader -H "Content-Type: application/json" -d '{"task_id":"easy","seed":42}' | python3 -m json.tool
```

### Validator script (mirrors the Phase 1 checks)
```bash
./scripts/validate-submission.sh https://arun-sanjay-dispatchpulse.hf.space .
```

All 3 checks must pass (or skip cleanly with WARN for docker/openenv if those CLIs aren't installed locally).

### inference.py end-to-end (in-process)
```bash
DISPATCHPULSE_TASK=easy .venv/bin/python inference.py 2>&1 | grep -E '^\[(START|STEP|END)\]'
```

Must produce exactly one `[START]` line, one or more `[STEP]` lines, and exactly one `[END]` line — all format-compliant.

### HF Space runtime status (is it healthy right now?)
```bash
curl -sf https://huggingface.co/api/spaces/Arun-Sanjay/dispatchpulse \
  | python3 -c "import sys,json; d=json.load(sys.stdin); rt=d.get('runtime',{}); print('stage:', rt.get('stage'), 'sha:', rt.get('sha')[:7], 'lastModified:', d.get('lastModified'))"
```

`stage: RUNNING` is what you want. `RUNNING_APP_STARTING` means a rebuild is in progress.

---

## 4. Deploy workflow (Arun runs the pushes, you prepare them)

### Step 1: commit locally

Small, logical chunks. Example:
```bash
git add <specific files>
git commit -m "<imperative verb> <thing>: <why>"
```

### Step 2: verify locally

```bash
.venv/bin/python tests/test_reward.py && .venv/bin/python tests/test_simulation.py
./scripts/validate-submission.sh https://arun-sanjay-dispatchpulse.hf.space .
```

Both must pass before handing off push commands.

### Step 3: hand Arun the push commands

Never run these yourself. Give him the exact commands, with `PASTE_HF_TOKEN_HERE` and `PASTE_GH_TOKEN_HERE` placeholders:

```bash
# Push to HF Space (triggers rebuild)
git push https://Arun-Sanjay:PASTE_HF_TOKEN_HERE@huggingface.co/spaces/Arun-Sanjay/dispatchpulse main

# Push to GitHub
git push https://Arun-Sanjay:PASTE_GH_TOKEN_HERE@github.com/Arun-Sanjay/DispatchPulse.git main
```

Remind him where to create fresh tokens:
- HF: https://huggingface.co/settings/tokens (role: **Write**)
- GitHub: https://github.com/settings/tokens (classic, scope: **repo**)

Remind him to **revoke both tokens immediately after the push**. Tokens in command lines are in shell history and should be treated as burned.

If Arun accidentally pastes a token in chat: stop, warn him, tell him to invalidate it, refuse to use it.

### Step 4: watch the rebuild

After Arun confirms the pushes landed:
```bash
# Poll the HF Space until stage=RUNNING with the new commit sha
for i in 1 2 3 4 5; do
  sleep 45
  curl -sf https://huggingface.co/api/spaces/Arun-Sanjay/dispatchpulse \
    | python3 -c "import sys,json; d=json.load(sys.stdin); rt=d.get('runtime',{}); print(f'poll {$i}: stage={rt.get(\"stage\")} sha={rt.get(\"sha\",\"\")[:7]}')"
  STATUS=$(curl -s -o /dev/null -w "%{http_code}" https://arun-sanjay-dispatchpulse.hf.space/tasks)
  if [ "$STATUS" = "200" ]; then
    echo "LIVE"; break
  fi
done
```

### Step 5: tell Arun to resubmit on Scaler

Give him the two URLs to paste:
- GitHub: `https://github.com/Arun-Sanjay/DispatchPulse`
- HF Space: `https://huggingface.co/spaces/Arun-Sanjay/dispatchpulse`
- Tick all 5 pre-submission checkboxes
- Click Submit Solution
- Wait for Phase 1 (~instant) → Phase 2 (30-40 min email)

---

## 5. What NOT to do (load-bearing don'ts)

1. **Don't push to GitHub or HF Space yourself.** Arun drives all git pushes. You prepare commits, he pushes.
2. **Don't accept tokens in chat.** If he pastes one, warn him, tell him to revoke, refuse to use.
3. **Don't refactor working code speculatively.** If Phase 2 is passing, don't touch anything. If it's failing, fix ONLY the specific named check.
4. **Don't change reward math, survival curves, or severity weights.** Calibration tests depend on them.
5. **Don't delete or rename `uv.lock`, `inference.py`, `openenv.yaml`, or `Dockerfile`.** All are load-bearing.
6. **Don't add heavy dependencies.** No torch, transformers, vllm, etc. Project runs on 2 vCPU / 8 GB RAM.
7. **Don't modify `tasks/*.yaml`.** The scenarios are locked in for this submission.
8. **Don't install pytest.** The tests are `__main__`-style scripts.
9. **Don't use `git push --force`** unless explicitly justified (the GitHub repo was force-pushed once to overwrite an auto-generated README — it should never happen again).
10. **Don't update this skill file or CLAUDE.md for cosmetic reasons.** Only update when project state actually changes.

---

## 6. Common mistakes from prior sessions (avoid repeating)

- **Confused `create_app` vs `create_fastapi_app`.** `create_app` is the right choice — it serves the Gradio UI at `/` when `ENABLE_WEB_INTERFACE=true`. `create_fastapi_app` is API-only and causes the "details not found" 404 at the root URL.
- **Used MCPEnvironment and MCPToolClient initially.** That was wrong. The canonical pattern is `Environment` base class + `EnvClient` base class. See commit `64d56f9`.
- **Forgot `uv.lock`.** Phase 1 validator fails with "Missing uv.lock - run 'uv lock' to generate it". Always run `.venv/bin/python -m pip install uv && .venv/bin/uv lock` if `uv.lock` is missing.
- **Forgot the `ENABLE_WEB_INTERFACE=true` env var** when testing locally. Without it, `create_app` drops the Gradio routes.
- **Pasted tokens into chat by accident.** This has happened 3+ times. Every paste requires an immediate revoke. Arun knows to be careful, but remind him up front.
- **Tried to run `pytest`.** Don't. The tests use `if __name__ == "__main__":` blocks and run as plain Python scripts.
- **Proposed preemptive refactors before Phase 2 finished.** Bad idea — wait for the concrete error, fix only that.

---

## 7. Emergency contacts & links

- **Hackathon dashboard:** https://www.scaler.com/school-of-technology/meta-pytorch-hackathon
- **OpenEnv docs:** https://meta-pytorch.org/OpenEnv/
- **OpenEnv GitHub:** https://github.com/meta-pytorch/OpenEnv
- **Reference passing submissions:**
  - Calendar Scheduling: https://github.com/Giresh-458/Calendar-Scheduling-OpenEnv
  - SQL Repair: https://github.com/WALKMAN303/openenv-project
  - Warehouse Logistics: https://huggingface.co/spaces/ArjunMadhava/meta-hackathon-2026
- **Submission help:** help_openenvhackathon@scaler.com
- **TRL OpenEnv integration docs:** https://huggingface.co/docs/trl/main/en/openenv

---

## 8. Resumption prompt — for when Arun starts a fresh Claude session

If Arun has just opened a new chat and wants you to pick up where the previous one left off, tell him to paste this into the new chat:

> I'm working on DispatchPulse at `/Users/arunsanjay/Documents/Projects/DispatchPulse` for the Meta PyTorch OpenEnv Hackathon India 2026. Round 1 submission is already in flight.
>
> Please read these two files before doing anything:
> 1. `CLAUDE.md` at the project root — full project context
> 2. `.claude/skills/dispatchpulse/SKILL.md` — operational playbook
>
> Then check the current state with `git log --oneline | head -5`, `git status`, and `curl -sf https://arun-sanjay-dispatchpulse.hf.space/health`.
>
> Here's the current situation: [briefly describe — e.g. "Phase 2 email came back with X failure" or "Phase 2 passed, I want to prep Round 2" or "I want to add LLM baseline scores to README"]
>
> What do you recommend as the next step?

That prompt plus the two files will get a new Claude session fully up to speed in under 30 seconds of reading.

---

**End of SKILL.md.** Load this alongside CLAUDE.md at the start of every session working on DispatchPulse.
