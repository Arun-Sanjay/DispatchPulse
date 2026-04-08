"""FastAPI application for DispatchPulse.

Uses ``create_app(...)`` from openenv-core for the standard ``/reset``,
``/step``, ``/state``, ``/health``, ``/metadata``, ``/schema``, ``/ws`` routes
plus the Gradio UI at ``/`` (when ``ENABLE_WEB_INTERFACE=true``).

On top of that baseline we add three DispatchPulse-specific endpoints the
hackathon grader discovers:

- ``GET /tasks`` — list the 3 graded tasks with metadata
- ``GET /tasks/{task_id}`` — single-task metadata lookup
- ``POST /grader`` — score an episode (silent run or replayed action list)

All three endpoints pull from :mod:`task_definitions`, which is the canonical
task registry for the repo.
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
from task_definitions import (  # noqa: E402
    GRADER_FUNCTIONS,
    NUM_TASKS_WITH_GRADERS,
    TASK_IDS_WITH_GRADERS,
    TASKS,
    TaskDefinition,
    grade_submission,
    get_task,
    list_tasks as _list_tasks,
    run_grader,
)

# Create the standard OpenEnv app (Gradio UI + HTTP API routes).
app = create_app(
    DispatchPulseEnvironment,
    DispatchPulseAction,
    DispatchPulseObservation,
    env_name="dispatchpulse",
    max_concurrent_envs=8,
)


# ---------------------------------------------------------------------------
# GET /tasks — list all graded tasks
# ---------------------------------------------------------------------------


class TaskInfo(BaseModel):
    """HTTP-serializable view of a TaskDefinition."""

    task_id: str
    name: str
    difficulty: str
    description: str
    max_steps: int
    time_limit_minutes: int
    num_calls: int
    num_units: int
    num_hospitals: int
    caller_inaccuracy: float
    has_grader: bool
    grader_fn_name: str


class TaskListResponse(BaseModel):
    tasks: List[TaskInfo]
    count: int
    num_tasks_with_graders: int
    task_ids_with_graders: List[str]
    grader_functions: List[str]


def _task_to_info(t: TaskDefinition) -> TaskInfo:
    return TaskInfo(
        task_id=t.task_id,
        name=t.name,
        difficulty=t.difficulty,
        description=t.description,
        max_steps=t.max_steps,
        time_limit_minutes=t.time_limit_minutes,
        num_calls=t.num_calls,
        num_units=t.num_units,
        num_hospitals=t.num_hospitals,
        caller_inaccuracy=t.caller_inaccuracy,
        has_grader=t.has_grader,
        grader_fn_name=t.grader_fn_name,
    )


@app.get("/tasks", tags=["DispatchPulse"], response_model=TaskListResponse)
def list_tasks_endpoint() -> TaskListResponse:
    """Return the full list of graded tasks.

    DispatchPulse ships with exactly three deterministic tasks — ``easy``,
    ``medium``, ``hard`` — each with its own grader (``grade_submission``)
    that returns a score in [0.0, 1.0] at episode end.
    """
    task_list = _list_tasks()
    return TaskListResponse(
        tasks=[_task_to_info(t) for t in task_list],
        count=len(task_list),
        num_tasks_with_graders=NUM_TASKS_WITH_GRADERS,
        task_ids_with_graders=TASK_IDS_WITH_GRADERS,
        grader_functions=GRADER_FUNCTIONS,
    )


@app.get("/tasks/{task_id}", tags=["DispatchPulse"], response_model=TaskInfo)
def get_task_endpoint(task_id: str) -> TaskInfo:
    """Return metadata for a single task by id."""
    try:
        task = get_task(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _task_to_info(task)


# ---------------------------------------------------------------------------
# POST /grader — score a submission
# ---------------------------------------------------------------------------


class GraderRequest(BaseModel):
    """Request body for POST /grader."""

    task_id: Optional[str] = Field(
        default=None, description="One of: easy | medium | hard"
    )
    seed: int = Field(default=42, description="Random seed for reproducibility")
    actions: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description=(
            "Ordered list of actions to replay (each item has action_type "
            "and required args). When omitted, grades a silent run."
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


@app.post("/grader", tags=["DispatchPulse"], response_model=GraderResult)
def grader_endpoint(payload: GraderRequest) -> GraderResult:
    """Grade a task submission.

    Delegates to :func:`task_definitions.grade_submission` which is the
    canonical grader for DispatchPulse.
    """
    task_id = (payload.task_id or "easy").strip().lower()
    try:
        score, details = grade_submission(
            task_id=task_id,
            actions=payload.actions,
            seed=int(payload.seed),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return GraderResult(
        task_id=details["task_id"],
        score=details["score"],
        passed=details["passed"],
        details=details["details"],
        survival_score=details["survival_score"],
        efficiency_score=details["efficiency_score"],
        triage_accuracy=details["triage_accuracy"],
        penalty=details["penalty"],
        completed_calls=details["completed_calls"],
        timed_out_calls=details["timed_out_calls"],
        total_calls=details["total_calls"],
    )


def main() -> None:
    """Entry point for ``uv run server`` or direct execution."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
