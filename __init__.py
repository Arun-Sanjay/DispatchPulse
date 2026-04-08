"""DispatchPulse — emergency-dispatch OpenEnv environment.

A real-world OpenEnv environment where an AI agent acts as a 911 emergency
dispatch coordinator. The agent triages incoming calls, dispatches limited
units (ALS / BLS ambulances, fire engines, police), and selects destination
hospitals. Patient outcomes are scored against real clinical survival
curves.

Public API:
    DispatchPulseEnv        — async client (subclass of openenv EnvClient)
    DispatchPulseAction     — typed action
    DispatchPulseObservation — typed observation
    DispatchPulseState      — typed state snapshot
    TASKS                   — registry of 3 graded tasks (easy, medium, hard)
    TaskDefinition          — frozen dataclass describing one task
    grade_submission(...)   — canonical grader function, returns (score, details)
    list_tasks()            — list all TaskDefinitions
    get_task(task_id)       — single task lookup
    run_grader              — alias for grade_submission
"""

from client import DispatchPulseEnv
from models import (
    DispatchPulseAction,
    DispatchPulseObservation,
    DispatchPulseState,
)
from task_definitions import (
    GRADER_FUNCTIONS,
    NUM_TASKS_WITH_GRADERS,
    TASK_IDS_WITH_GRADERS,
    TASKS,
    TaskDefinition,
    grade_submission,
    get_task,
    list_tasks,
    run_grader,
)

__all__ = [
    "DispatchPulseEnv",
    "DispatchPulseAction",
    "DispatchPulseObservation",
    "DispatchPulseState",
    "TASKS",
    "TaskDefinition",
    "grade_submission",
    "list_tasks",
    "get_task",
    "run_grader",
    "NUM_TASKS_WITH_GRADERS",
    "TASK_IDS_WITH_GRADERS",
    "GRADER_FUNCTIONS",
]
__version__ = "1.0.0"
