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
TASK_NAME = os.getenv("DISPATCHPULSE_TASK", "easy")
BENCHMARK = "dispatchpulse"

# Episode caps — keep small enough to finish in <20 minutes total
MAX_STEPS = 60
TEMPERATURE = 0.0
MAX_TOKENS = 200
SUCCESS_SCORE_THRESHOLD = 0.20

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


def parse_action_text(text: str) -> DispatchPulseAction:
    """Parse the LLM's plain-text reply into a DispatchPulseAction."""
    text = (text or "").strip()
    # Take only the first non-empty line
    for line in text.splitlines():
        line = line.strip().strip("`").strip()
        if line:
            text = line
            break
    parts = text.split(maxsplit=4)
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
            text=text,
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
            text=text,
        )
    if head == "callback" and len(parts) >= 2:
        question = " ".join(parts[2:]) if len(parts) > 2 else ""
        return DispatchPulseAction(
            action_type="callback",
            call_id=parts[1],
            message=question,
            text=text,
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


async def run_episode(task_name: str) -> None:
    """Run one episode against the configured environment.

    Emits exactly one [START] line, one [STEP] line per step, and one [END]
    line at the end (always, even on exception).
    """
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "missing-key")

    env: Any
    if LOCAL_IMAGE_NAME:
        try:
            env = await DispatchPulseEnv.from_docker_image(LOCAL_IMAGE_NAME)
            print(
                f"[DEBUG] connected via from_docker_image({LOCAL_IMAGE_NAME!r})",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[DEBUG] from_docker_image failed ({exc}); falling back to in-process",
                flush=True,
            )
            env = _LocalInProcessEnv(task_name=task_name, seed=42)
    elif ENV_BASE_URL:
        try:
            env = DispatchPulseEnv(base_url=ENV_BASE_URL)
            await env.connect()
            print(f"[DEBUG] connected to remote env at {ENV_BASE_URL}", flush=True)
        except Exception as exc:
            print(
                f"[DEBUG] remote connect failed ({exc}); falling back to in-process",
                flush=True,
            )
            env = _LocalInProcessEnv(task_name=task_name, seed=42)
    else:
        env = _LocalInProcessEnv(task_name=task_name, seed=42)
        print("[DEBUG] using in-process fallback env", flush=True)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_name=task_name, seed=42)
        obs_text = getattr(result.observation, "text", "") or ""

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action_text = get_model_action_text(client, obs_text, history)
            action = parse_action_text(action_text)

            error: Optional[str] = None
            try:
                result = await env.step(action)
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

            reward_value = float(result.reward or 0.0)
            done = bool(result.done)
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
                # The terminal step's reward IS the final episode score [0,1]
                score = max(0.0, min(1.0, reward_value))
                break

        if score == 0.0 and rewards:
            # Fallback: clamp the max observed reward
            score = max(0.0, min(1.0, max(rewards)))

        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode crashed: {exc}", flush=True)
    finally:
        try:
            await env.close()
        except Exception as exc:
            print(f"[DEBUG] env.close() error: {exc}", flush=True)
        log_end(
            success=success, steps=steps_taken, score=score, rewards=rewards
        )


async def main() -> None:
    task = TASK_NAME if TASK_NAME in VALID_TASKS else "easy"
    await run_episode(task)


if __name__ == "__main__":
    asyncio.run(main())
