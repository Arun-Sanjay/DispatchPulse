"""FastAPI application for DispatchPulse.

Uses ``create_app(...)`` from openenv-core for the standard ``/reset``,
``/step``, ``/state``, ``/health``, ``/metadata``, ``/schema``, ``/ws`` routes
plus the Gradio UI at ``/`` (when ``ENABLE_WEB_INTERFACE=true``).

On top of that baseline we add two DispatchPulse-specific endpoints the
hackathon grader discovers:

- ``GET /tasks`` — list the 3 graded tasks with metadata
- ``POST /grader`` — score an episode or explicit call log against a task
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from pydantic import BaseModel, Field

# Support both in-repo and standalone imports.
try:
    from openenv.core.env_server.http_server import create_app

    from .environment import DispatchPulseEnvironment
except ImportError:  # pragma: no cover
    from openenv.core.env_server.http_server import create_app
    from server.environment import DispatchPulseEnvironment

# Import project root modules
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from models import DispatchPulseAction, DispatchPulseObservation  # noqa: E402
from grader import grade_simulation  # noqa: E402
from reward import calculate_episode_reward  # noqa: E402
from scenario_loader import VALID_TASKS, load_scenario  # noqa: E402
from simulation import DispatchSimulation  # noqa: E402

# Create the standard OpenEnv app (Gradio UI + HTTP API routes).
app = create_app(
    DispatchPulseEnvironment,
    DispatchPulseAction,
    DispatchPulseObservation,
    env_name="dispatchpulse",
    max_concurrent_envs=8,
)


# ---------------------------------------------------------------------------
# Task catalog — 3 graded tasks with metadata for GET /tasks
# ---------------------------------------------------------------------------


class TaskInfo(BaseModel):
    """Metadata for a single graded task."""

    task_id: str
    name: str
    difficulty: str = Field(..., description="easy | medium | hard")
    description: str
    max_steps: int
    time_limit_minutes: int
    num_calls: int
    num_units: int
    num_hospitals: int
    caller_inaccuracy: float
    has_grader: bool = True


class TaskListResponse(BaseModel):
    """Response for GET /tasks."""

    tasks: List[TaskInfo]
    count: int


def _task_info(task_id: str) -> TaskInfo:
    scenario = load_scenario(task_id)
    world_cfg = scenario.get("world_config", {}) or {}
    return TaskInfo(
        task_id=task_id,
        name=scenario.get("name", task_id),
        difficulty=task_id,
        description=(scenario.get("description") or "").strip(),
        max_steps=int(world_cfg.get("time_limit_minutes", 30)),
        time_limit_minutes=int(world_cfg.get("time_limit_minutes", 30)),
        num_calls=len(scenario.get("calls", [])),
        num_units=len(scenario.get("units", [])),
        num_hospitals=len(scenario.get("hospitals", [])),
        caller_inaccuracy=float(scenario.get("caller_inaccuracy", 0.0)),
        has_grader=True,
    )


@app.get("/tasks", tags=["DispatchPulse"], response_model=TaskListResponse)
def list_tasks() -> TaskListResponse:
    """Return the full list of graded tasks.

    DispatchPulse ships with exactly three deterministic tasks — ``easy``,
    ``medium``, ``hard`` — each with its own grader that returns a score in
    [0.0, 1.0] at episode end.
    """
    infos = [_task_info(t) for t in VALID_TASKS]
    return TaskListResponse(tasks=infos, count=len(infos))


@app.get("/tasks/{task_id}", tags=["DispatchPulse"], response_model=TaskInfo)
def get_task(task_id: str) -> TaskInfo:
    """Return metadata for a single task by id."""
    if task_id not in VALID_TASKS:
        raise HTTPException(
            status_code=404,
            detail=f"unknown task_id '{task_id}' (valid: {', '.join(VALID_TASKS)})",
        )
    return _task_info(task_id)


# ---------------------------------------------------------------------------
# Grader — POST /grader
# ---------------------------------------------------------------------------


class GraderRequest(BaseModel):
    """Request body for POST /grader.

    Provide either an ``episode_id`` (to grade a live episode that's already
    been run) or an explicit ``task_id`` + action log (to re-run and grade a
    scripted episode without needing any server-side state).
    """

    task_id: Optional[str] = Field(
        default=None, description="One of: easy | medium | hard"
    )
    seed: int = Field(default=42, description="Random seed for reproducibility")
    actions: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description=(
            "Ordered list of actions to replay (each item has "
            "action_type and any required args). When omitted, the grader "
            "scores the simulation as-is at its current state."
        ),
    )


class GraderResult(BaseModel):
    """Response from POST /grader."""

    task_id: str
    score: float = Field(..., ge=0.0, le=1.0)
    passed: bool
    details: str
    survival_score: float
    efficiency_score: float
    triage_accuracy: float
    penalty: float
    completed_calls: int
    timed_out_calls: int
    total_calls: int


def _replay_actions(sim: DispatchSimulation, actions: List[Dict[str, Any]]) -> None:
    """Replay a scripted action list through a fresh simulation."""
    max_steps = 500
    for idx, act in enumerate(actions):
        if idx >= max_steps or sim.episode_done:
            break
        atype = (act.get("action_type") or "").strip().lower()
        if atype == "dispatch":
            sim.dispatch(
                call_id=str(act.get("call_id", "")),
                unit_id=str(act.get("unit_id", "")),
                hospital_id=act.get("hospital_id"),
            )
            sim.advance_time(1)
        elif atype == "classify":
            try:
                sev = int(act.get("severity", 3))
            except (TypeError, ValueError):
                sev = 3
            sim.classify(str(act.get("call_id", "")), sev)
            sim.advance_time(1)
        elif atype == "callback":
            sim.callback(
                str(act.get("call_id", "")),
                str(act.get("message", act.get("question", ""))),
            )
            sim.advance_time(1)
        elif atype == "wait":
            try:
                mins = int(act.get("minutes", 1))
            except (TypeError, ValueError):
                mins = 1
            sim.advance_time(max(1, min(mins, sim.config.max_wait_step_minutes)))
        elif atype == "view":
            continue
        else:
            sim.advance_time(1)

    # If we ran out of actions before the episode ended, fast-forward the
    # clock so all remaining calls time out and the episode terminates.
    while not sim.episode_done:
        sim.advance_time(sim.config.time_limit_minutes)


@app.post("/grader", tags=["DispatchPulse"], response_model=GraderResult)
def grade_task(payload: GraderRequest) -> GraderResult:
    """Run the grader for a task.

    Two modes:
        1. ``task_id`` only → score a silent run (all calls timeout) as a
           sanity check that the task loads and has a valid grader.
        2. ``task_id + actions`` → replay the scripted action log then score.
    """
    task_id = (payload.task_id or "easy").strip().lower()
    if task_id not in VALID_TASKS:
        raise HTTPException(
            status_code=404,
            detail=f"unknown task_id '{task_id}' (valid: {', '.join(VALID_TASKS)})",
        )

    scenario = load_scenario(task_id)
    sim = DispatchSimulation(scenario, seed=int(payload.seed))

    if payload.actions:
        _replay_actions(sim, payload.actions)
    else:
        # No actions provided: run the episode to completion with no decisions.
        while not sim.episode_done:
            sim.advance_time(sim.config.time_limit_minutes)

    reward = calculate_episode_reward(
        sim.completed_calls,
        sim.timed_out_calls,
        sim.total_calls(),
        sim.dispatches,
    )

    return GraderResult(
        task_id=task_id,
        score=reward.total,
        passed=reward.total >= 0.20,
        details=reward.details,
        survival_score=reward.survival_score,
        efficiency_score=reward.efficiency_score,
        triage_accuracy=reward.triage_accuracy,
        penalty=reward.penalty,
        completed_calls=len(sim.completed_calls),
        timed_out_calls=len(sim.timed_out_calls),
        total_calls=sim.total_calls(),
    )


def main() -> None:
    """Entry point for ``uv run server`` or direct execution."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
