"""DispatchPulse — Round 1 inference script.

Strictly follows the Meta PyTorch OpenEnv Hackathon submission spec.

MANDATORY env vars (per the official sample):
    API_BASE_URL    LLM endpoint (default: https://router.huggingface.co/v1)
    MODEL_NAME      Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN        API key for the LLM
    LOCAL_IMAGE_NAME Local docker image name when using from_docker_image()

Stdout format (exact, three line types):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Connection logic:
    - If LOCAL_IMAGE_NAME is set: use ``from_docker_image(LOCAL_IMAGE_NAME)``
    - Else if ENV_BASE_URL is set: connect directly to that running server
    - Else: spin up an in-process simulation as a fallback (for offline runs)
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import textwrap
from typing import Any, List, Optional

# ---------------------------------------------------------------------------
# Make project modules importable when this script is run directly.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from openai import OpenAI  # noqa: E402  (per spec — must use openai client)

from client import DispatchPulseEnv  # noqa: E402
from models import DispatchPulseAction  # noqa: E402

# ---------------------------------------------------------------------------
# MANDATORY environment variables (per submission spec)
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
ENV_BASE_URL = os.getenv("ENV_BASE_URL")  # optional override for direct URL

# Task selection: grader sets DISPATCHPULSE_TASK to one of {easy, medium, hard}
TASK_NAME = os.getenv("DISPATCHPULSE_TASK")  # back-compat: if set, run only this one task
BENCHMARK = "dispatchpulse"

# All three graded tasks — the inference script iterates through this list by default,
# emitting one [START]/[STEP]*/[END] block per task, matching the pattern used by every
# passing Meta PyTorch OpenEnv Hackathon submission (SQL Repair, Calendar Scheduling,
# Warehouse Logistics). The Phase 2 grader counts task blocks in stdout, so running
# only one task produces a "Not enough tasks with graders" failure.
TASK_IDS = ["easy", "medium", "hard"]

# Episode caps — keep small enough to finish in <20 minutes total
MAX_STEPS = 60
TEMPERATURE = 0.0
MAX_TOKENS = 200
SUCCESS_SCORE_THRESHOLD = 0.20

# Hard timeouts (seconds) — guarantee script finishes under the 20 min grader cap
LLM_CALL_TIMEOUT_S = 60.0
ENV_STEP_TIMEOUT_S = 30.0
TOTAL_EPISODE_TIMEOUT_S = 900.0  # 15 min, leaves 5 min buffer under 20 min cap

VALID_TASKS = ("easy", "medium", "hard")


# ---------------------------------------------------------------------------
# Required stdout logging helpers
# ---------------------------------------------------------------------------


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_clean = " ".join(str(action).split())  # collapse whitespace
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool, steps: int, score: float, rewards: List[float]
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM dispatcher prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an experienced 911 emergency dispatch coordinator.
    You receive incoming emergency calls and must dispatch the right unit
    at the right time to maximise patient survival outcomes.

    DISPATCHER STANDARD OPERATING PROCEDURE:
    1. CRITICAL CALLS FIRST. Severity 1 (cardiac arrest, severe trauma,
       stroke) is life-threatening. Cardiac arrest survival drops ~10% per
       minute.
    2. SEND THE RIGHT UNIT. ALS ambulance for cardiac/stroke/severe trauma.
       BLS ambulance for stable patients and minor injuries. Fire engine
       only for fires. Police for mental health crises.
    3. CONSERVE ALS UNITS. Do not send your only ALS to a sprained ankle.
    4. PICK THE RIGHT HOSPITAL. Cardiac -> hospital with cardiac unit;
       stroke -> stroke unit; trauma -> trauma center. Avoid hospitals on
       diversion or with zero beds.
    5. CALLBACK WHEN UNCLEAR. If a caller's description seems wrong, use
       callback to verify the true emergency type.
    6. WAIT WHEN APPROPRIATE. If no decisions are pending, advance time.

    On each turn you receive a text view of the dispatch center. You must
    reply with EXACTLY one action, one of:

      dispatch <call_id> <unit_id> [hospital_id]
      classify <call_id> <severity 1-5>
      callback <call_id> <free-text question>
      wait <minutes 1-5>

    Examples:
      dispatch CALL-001 ALS-1 H1
      classify CALL-002 1
      callback CALL-003 Is the patient breathing?
      wait 2

    Reply with the action only. No explanation, no markdown.
    """
).strip()


def build_user_prompt(observation_text: str, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "(no prior actions)"
    return (
        f"Recent actions you took:\n{history_block}\n\n"
        f"Current dispatch center:\n{observation_text}\n\n"
        f"Reply with exactly one action."
    )


def get_model_action_text(
    client: OpenAI, observation_text: str, history: List[str]
) -> str:
    user_prompt = build_user_prompt(observation_text, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else "wait 1"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "wait 1"


# ---------------------------------------------------------------------------
# Action parsing — converts the LLM's free text into a DispatchPulseAction
# ---------------------------------------------------------------------------


_ACTION_VERBS = ("dispatch", "classify", "callback", "wait", "view", "view_dispatch_center")

_FUNC_CALL_RE = re.compile(r"^(\w+)\s*\((.*)\)\s*$")
_PREFIX_RE = re.compile(
    r"^(action|response|answer|output|result)\s*[:\-=]\s*",
    flags=re.IGNORECASE,
)


def _clean_llm_text(text: str) -> str:
    """Best-effort normalize an LLM's plain-text reply into a single action line.

    Handles markdown code fences, `Action:`/`Response:` prefixes, leading quote
    characters, trailing punctuation, and quoted strings. If none of the lines
    look like a valid action verb, returns ``""`` so the caller falls back to
    ``wait 1``.
    """
    if not text:
        return ""

    # Strip outer markdown code fences ```python ... ``` or ``` ... ```
    fenced = re.sub(r"```[a-zA-Z0-9_\-]*\n?", "", text)
    fenced = re.sub(r"\n?```\s*$", "", fenced)
    fenced = fenced.strip()

    # Walk through lines; the first one that starts with an action verb wins
    candidate = ""
    for raw_line in fenced.splitlines():
        line = raw_line.strip().strip("`").strip()
        line = _PREFIX_RE.sub("", line)  # drop "Action:" / "Response:" prefixes
        line = line.strip().strip("'\"").strip()  # drop leading/trailing quotes
        line = line.rstrip(".!?,;:")  # drop trailing punctuation
        if not line:
            continue
        first_word = line.split(maxsplit=1)[0].lower() if line else ""
        if first_word in _ACTION_VERBS:
            candidate = line
            break
        if not candidate:
            candidate = line  # keep the first non-empty line as last-resort fallback

    if not candidate:
        return ""

    # Handle function-call syntax: dispatch(CALL-001, ALS-1, H1) → dispatch CALL-001 ALS-1 H1
    match = _FUNC_CALL_RE.match(candidate)
    if match:
        verb = match.group(1).lower()
        if verb in _ACTION_VERBS:
            args_raw = match.group(2)
            args = [a.strip().strip("'\"").strip() for a in args_raw.split(",")]
            args = [a for a in args if a]
            candidate = " ".join([verb] + args)

    return candidate


def parse_action_text(text: str) -> DispatchPulseAction:
    """Parse the LLM's plain-text reply into a DispatchPulseAction.

    Lenient to common LLM output drift: markdown code fences, ``Action:`` /
    ``Response:`` prefixes, function-call syntax ``dispatch(CALL-001, ALS-1)``,
    trailing punctuation, quoted strings, and multi-line replies.
    """
    cleaned = _clean_llm_text(text)
    if not cleaned:
        return DispatchPulseAction(action_type="wait", minutes=1, text="wait 1")

    parts = cleaned.split(maxsplit=4)
    if not parts:
        return DispatchPulseAction(action_type="wait", minutes=1, text="wait 1")
    head = parts[0].lower()

    if head == "dispatch" and len(parts) >= 3:
        hospital = parts[3] if len(parts) >= 4 else None
        return DispatchPulseAction(
            action_type="dispatch",
            call_id=parts[1],
            unit_id=parts[2],
            hospital_id=hospital,
            text=cleaned,
        )
    if head == "classify" and len(parts) >= 3:
        try:
            sev = int(parts[2])
        except ValueError:
            return DispatchPulseAction(action_type="wait", minutes=1, text="wait 1")
        return DispatchPulseAction(
            action_type="classify",
            call_id=parts[1],
            severity=sev,
            text=cleaned,
        )
    if head == "callback" and len(parts) >= 2:
        question = " ".join(parts[2:]) if len(parts) > 2 else ""
        return DispatchPulseAction(
            action_type="callback",
            call_id=parts[1],
            message=question,
            text=cleaned,
        )
    if head == "wait":
        try:
            mins = int(parts[1]) if len(parts) > 1 else 1
        except ValueError:
            mins = 1
        mins = max(1, min(mins, 5))
        return DispatchPulseAction(action_type="wait", minutes=mins, text=f"wait {mins}")
    if head in ("view", "view_dispatch_center"):
        return DispatchPulseAction(action_type="view", text="view")
    return DispatchPulseAction(action_type="wait", minutes=1, text="wait 1")


# ---------------------------------------------------------------------------
# Local in-process fallback (for offline runs without Docker / network)
# ---------------------------------------------------------------------------


class _LocalInProcessEnv:
    """Minimal in-process env that mimics the OpenEnv client interface.

    Used as a fallback when neither LOCAL_IMAGE_NAME nor ENV_BASE_URL is set.
    Exposes async ``reset()`` / ``step(action)`` / ``close()`` returning
    objects shaped like ``StepResult``.
    """

    def __init__(self, task_name: str, seed: int = 42) -> None:
        from scenario_loader import load_scenario
        from simulation import DispatchSimulation
        from text_view import render_dispatch_center

        self._render = render_dispatch_center
        self._sim_cls = DispatchSimulation
        self._scenario = load_scenario(task_name)
        self._task = task_name
        self._seed = seed
        self.sim = None

    async def reset(self, **_kwargs) -> Any:
        self.sim = self._sim_cls(self._scenario, seed=self._seed)
        return _SimpleResult(
            text=self._render(self.sim, self._task), reward=0.0, done=False
        )

    async def step(self, action: DispatchPulseAction) -> Any:
        from grader import grade_simulation

        if self.sim is None:
            raise RuntimeError("Call reset() first.")
        if self.sim.episode_done:
            return _SimpleResult(
                text=self._render(self.sim, self._task),
                reward=0.0,
                done=True,
            )

        action_type = (action.action_type or "").strip().lower()
        if action_type == "dispatch":
            self.sim.dispatch(
                call_id=action.call_id or "",
                unit_id=action.unit_id or "",
                hospital_id=action.hospital_id,
            )
            self.sim.advance_time(1)
        elif action_type == "classify":
            self.sim.classify(
                call_id=action.call_id or "",
                severity=int(action.severity or 3),
            )
            self.sim.advance_time(1)
        elif action_type == "callback":
            self.sim.callback(
                call_id=action.call_id or "",
                question=action.message or "",
            )
            self.sim.advance_time(1)
        elif action_type == "wait":
            self.sim.advance_time(int(action.minutes or 1))
        elif action_type == "view":
            pass
        else:
            self.sim.advance_time(1)

        done = bool(self.sim.episode_done)
        reward = float(grade_simulation(self.sim).total) if done else 0.0
        return _SimpleResult(
            text=self._render(self.sim, self._task), reward=reward, done=done
        )

    async def close(self) -> None:
        return None


class _SimpleResult:
    def __init__(self, text: str, reward: float, done: bool) -> None:
        self.observation = _SimpleObs(text)
        self.reward = reward
        self.done = done


class _SimpleObs:
    def __init__(self, text: str) -> None:
        self.text = text


# ---------------------------------------------------------------------------
# Main async entry point
# ---------------------------------------------------------------------------


async def _connect_env(task_name: str) -> Any:
    """Open an env connection matching the grader's expectations.

    Priority order:
      1. LOCAL_IMAGE_NAME (or IMAGE_NAME) → ``DispatchPulseEnv.from_docker_image``
      2. ENV_BASE_URL → connect to a running HTTP server
      3. In-process ``_LocalInProcessEnv`` fallback for offline / tests
    """
    if LOCAL_IMAGE_NAME:
        try:
            env = await DispatchPulseEnv.from_docker_image(LOCAL_IMAGE_NAME)
            print(
                f"[DEBUG] connected via from_docker_image({LOCAL_IMAGE_NAME!r})",
                flush=True,
            )
            return env
        except Exception as exc:
            print(
                f"[DEBUG] from_docker_image failed ({exc}); falling back to in-process",
                flush=True,
            )
            return _LocalInProcessEnv(task_name=task_name, seed=42)

    if ENV_BASE_URL:
        try:
            env = DispatchPulseEnv(base_url=ENV_BASE_URL)
            await env.connect()
            print(f"[DEBUG] connected to remote env at {ENV_BASE_URL}", flush=True)
            return env
        except Exception as exc:
            print(
                f"[DEBUG] remote connect failed ({exc}); falling back to in-process",
                flush=True,
            )
            return _LocalInProcessEnv(task_name=task_name, seed=42)

    print("[DEBUG] using in-process fallback env", flush=True)
    return _LocalInProcessEnv(task_name=task_name, seed=42)


async def run_episode(env: Any, client: OpenAI, task_name: str) -> float:
    """Run ONE task episode. Emits exactly one [START], N [STEP], one [END].

    ``env`` and ``client`` are caller-owned. The episode always emits an [END]
    line (even on exception) via a try/finally, and the final score is
    computed from the terminal observation's reward (the grader uses this).

    Returns the final score in [0, 1] so the caller can aggregate a summary.
    """
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    done_seen = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await asyncio.wait_for(
            env.reset(task_name=task_name, seed=42),
            timeout=ENV_STEP_TIMEOUT_S,
        )
        obs_text = getattr(result.observation, "text", "") or ""

        for step in range(1, MAX_STEPS + 1):
            if getattr(result, "done", False):
                done_seen = True
                break

            # LLM call with its own timeout
            try:
                action_text = await asyncio.wait_for(
                    asyncio.to_thread(get_model_action_text, client, obs_text, history),
                    timeout=LLM_CALL_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                action_text = "wait 1"
                print("[DEBUG] LLM call timeout; falling back to wait 1", flush=True)

            action = parse_action_text(action_text)

            error: Optional[str] = None
            try:
                result = await asyncio.wait_for(
                    env.step(action),
                    timeout=ENV_STEP_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                error = "env.step timeout"
                rewards.append(0.0)
                steps_taken = step
                log_step(
                    step=step,
                    action=action.text or action.action_type,
                    reward=0.0,
                    done=False,
                    error=error,
                )
                continue
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                rewards.append(0.0)
                steps_taken = step
                log_step(
                    step=step,
                    action=action.text or action.action_type,
                    reward=0.0,
                    done=False,
                    error=error,
                )
                continue

            reward_value = float(getattr(result, "reward", 0.0) or 0.0)
            done = bool(getattr(result, "done", False))
            rewards.append(reward_value)
            steps_taken = step
            obs_text = getattr(result.observation, "text", "") or obs_text
            history.append(
                f"step {step}: {action.text or action.action_type} -> r={reward_value:.2f}"
            )

            log_step(
                step=step,
                action=action.text or action.action_type,
                reward=reward_value,
                done=done,
                error=getattr(result.observation, "last_action_error", None),
            )

            if done:
                done_seen = True
                # Terminal step's reward IS the final episode score [0, 1]
                score = max(0.0, min(1.0, reward_value))
                break

        # Only use the rewards-max fallback when the episode loop exhausted
        # MAX_STEPS WITHOUT ever seeing done=True. When done was reached, the
        # terminal reward is authoritative — preserving legitimate zero scores.
        if not done_seen and score == 0.0 and rewards:
            score = max(0.0, min(1.0, max(rewards)))

        success = score >= SUCCESS_SCORE_THRESHOLD

    except asyncio.TimeoutError:
        print("[DEBUG] Episode timed out at top level", flush=True)
    except Exception as exc:
        print(f"[DEBUG] Episode crashed: {type(exc).__name__}: {exc}", flush=True)
    finally:
        log_end(
            success=success, steps=steps_taken, score=score, rewards=rewards
        )

    return score


def _tasks_to_run() -> List[str]:
    """Resolve which tasks to run this invocation.

    Priority:
      1. ``TASK_IDS`` env var (comma-separated) — overrides everything
      2. ``DISPATCHPULSE_TASK`` env var — single-task back-compat
      3. Default: run ALL graded tasks (easy, medium, hard)

    The default behaviour is what the hackathon grader depends on: a single
    ``python inference.py`` invocation must produce one [START]/[END] block
    per graded task so the Phase 2 Task Validation check sees N >= 3.
    """
    raw = os.getenv("TASK_IDS")
    if raw:
        ids = [t.strip() for t in raw.split(",") if t.strip()]
        ids = [t for t in ids if t in VALID_TASKS]
        if ids:
            return ids
    if TASK_NAME and TASK_NAME in VALID_TASKS:
        return [TASK_NAME]
    return list(TASK_IDS)


async def main() -> None:
    """Run every graded task once and emit a [START]/[STEP]/[END] block per task.

    The Phase 2 grader counts task blocks in this script's stdout. Running
    one task is "Not enough tasks with graders"; running all three is what
    the spec requires.
    """
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "missing-key")

    tasks = _tasks_to_run()
    print(f"[DEBUG] running tasks: {tasks}", flush=True)

    # Use a single env connection across all tasks when possible. For the
    # in-process fallback, each task spins its own fresh sim via reset().
    # For remote / docker envs, reset() re-initializes the simulation
    # server-side with the requested task.
    env: Any = None
    try:
        env = await _connect_env(task_name=tasks[0])

        for task_name in tasks:
            try:
                await asyncio.wait_for(
                    run_episode(env, client, task_name),
                    timeout=TOTAL_EPISODE_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                print(
                    f"[DEBUG] task {task_name} exceeded TOTAL_EPISODE_TIMEOUT_S; "
                    f"emitting fallback [END]",
                    flush=True,
                )
                log_end(success=False, steps=0, score=0.0, rewards=[])
            except Exception as exc:
                print(
                    f"[DEBUG] task {task_name} crashed: {type(exc).__name__}: {exc}; "
                    f"emitting fallback [END]",
                    flush=True,
                )
                log_end(success=False, steps=0, score=0.0, rewards=[])
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception as exc:
                print(f"[DEBUG] env.close() error: {exc}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
