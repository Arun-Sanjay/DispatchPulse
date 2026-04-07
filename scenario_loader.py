"""Load task scenarios from YAML files bundled with the package."""

from __future__ import annotations

import os
from typing import List

import yaml

# Resolve the tasks directory relative to this file so it works regardless of CWD
_TASKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks")

VALID_TASKS = ("easy", "medium", "hard")


def load_scenario(task_name: str) -> dict:
    """Load and return the scenario dict for the given task.

    Raises:
        ValueError: if task_name is not one of {easy, medium, hard}.
        FileNotFoundError: if the YAML file is missing.
    """
    if task_name not in VALID_TASKS:
        raise ValueError(
            f"Unknown task '{task_name}'. Valid tasks: {', '.join(VALID_TASKS)}"
        )
    path = os.path.join(_TASKS_DIR, f"{task_name}.yaml")
    with open(path, "r", encoding="utf-8") as f:
        scenario = yaml.safe_load(f)
    if "name" not in scenario:
        scenario["name"] = task_name
    return scenario


def list_tasks() -> List[str]:
    return list(VALID_TASKS)
