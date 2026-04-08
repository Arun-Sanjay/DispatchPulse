"""FastAPI application for DispatchPulse.

Uses ``create_fastapi_app(...)`` from openenv-core for the standard ``/reset``,
``/step``, ``/state``, ``/health``, ``/metadata``, ``/schema``, ``/ws`` routes.

On top of that baseline we add four DispatchPulse-specific endpoints the
hackathon grader discovers:

- ``GET /``          — Root metadata (name, status, endpoints)
- ``GET /tasks``     — list the 3 graded tasks
- ``GET /tasks/{task_id}`` — single-task metadata lookup
- ``POST /grader``   — score an episode (silent run or replayed action list)
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from pydantic import BaseModel, Field

# Support both in-repo and standalone imports.
try:
    from openenv.core.env_server import create_fastapi_app

    from .environment import DispatchPulseEnvironment
except ImportError:  # pragma: no cover
    from openenv.core.env_server import create_fastapi_app
    from server.environment import DispatchPulseEnvironment

# Import project root modules
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from models import DispatchPulseAction, DispatchPulseObservation  # noqa: E402
from task_definitions import (  # noqa: E402
    TASKS,
    grade_submission,
    get_task,
    list_tasks as _list_tasks,
)

# Create the standard OpenEnv app (API-only — no Gradio UI).
app = create_fastapi_app(
    DispatchPulseEnvironment,
    DispatchPulseAction,
    DispatchPulseObservation,
)


# ---------------------------------------------------------------------------
# GET / — root metadata
# ---------------------------------------------------------------------------


@app.get("/", tags=["DispatchPulse"])
def root() -> Dict[str, Any]:
    """Root endpoint returning basic metadata about the environment."""
    return {
        "name": "dispatchpulse",
        "status": "ok",
        "description": "Emergency dispatch coordinator OpenEnv environment",
        "endpoints": [
            "/health",
            "/tasks",
            "/tasks/{task_id}",
            "/reset",
            "/step",
            "/state",
            "/grader",
        ],
        "tasks": ["easy", "medium", "hard"],
    }


# ---------------------------------------------------------------------------
# GET /tasks — list all graded tasks
# ---------------------------------------------------------------------------


_ACTION_SCHEMA = {
    "action_type": "string — one of: dispatch, classify, callback, wait, view",
    "text": "string — e.g. 'dispatch CALL-001 ALS-1 H1'",
    "call_id": "string (optional)",
    "unit_id": "string (optional)",
    "hospital_id": "string (optional)",
    "severity": "integer 1-5 (optional)",
    "message": "string (optional)",
    "minutes": "integer 1-5 (optional)",
}


@app.get("/tasks", tags=["DispatchPulse"])
def list_tasks_endpoint() -> Dict[str, Any]:
    """Return the list of graded tasks.

    DispatchPulse ships with exactly three deterministic tasks — ``easy``,
    ``medium``, ``hard`` — each with its own grader (``grade_submission``)
    that returns a score in [0.0, 1.0] at episode end.
    """
    task_list = _list_tasks()
    return {
        "tasks": [
            {
                "task_id": t.task_id,
                "name": t.name,
                "difficulty": t.difficulty,
                "description": t.description,
            }
            for t in task_list
        ],
        "total": len(TASKS),
        "action_schema": _ACTION_SCHEMA,
    }


@app.get("/tasks/{task_id}", tags=["DispatchPulse"])
def get_task_endpoint(task_id: str) -> Dict[str, Any]:
    """Return metadata for a single task by id."""
    try:
        task = get_task(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "task_id": task.task_id,
        "name": task.name,
        "difficulty": task.difficulty,
        "description": task.description,
    }


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


@app.post("/grader", tags=["DispatchPulse"])
def grader_endpoint(payload: GraderRequest) -> Dict[str, Any]:
    """Grade a task submission.

    Delegates to :func:`task_definitions.grade_submission` which is the
    canonical grader for DispatchPulse. Returns a dict with `task_id`,
    `score`, `passed`, and reward component breakdown.
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

    return {
        "task_id": details["task_id"],
        "score": details["score"],
        "passed": details["passed"],
        "details": details["details"],
        "survival_score": details["survival_score"],
        "efficiency_score": details["efficiency_score"],
        "triage_accuracy": details["triage_accuracy"],
        "penalty": details["penalty"],
        "completed_calls": details["completed_calls"],
        "timed_out_calls": details["timed_out_calls"],
        "total_calls": details["total_calls"],
    }


def main() -> None:
    """Entry point for ``uv run server`` or direct execution."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
