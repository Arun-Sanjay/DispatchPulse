"""Programmatic grader. Returns a single 0.0–1.0 score for an episode."""

from __future__ import annotations

from typing import Optional

from models import Reward
from reward import calculate_episode_reward
from simulation import DispatchSimulation


def grade_simulation(sim: DispatchSimulation) -> Reward:
    """Compute the final Reward from a (finished or in-progress) simulation."""
    return calculate_episode_reward(
        sim.completed_calls,
        sim.timed_out_calls,
        sim.total_calls(),
        sim.dispatches,
    )


def grade_score(sim: DispatchSimulation) -> float:
    """Convenience: just the scalar 0.0–1.0 score."""
    return grade_simulation(sim).total
