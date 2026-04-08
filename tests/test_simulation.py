"""Tests that the DispatchSimulation engine behaves correctly end-to-end."""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from grader import grade_simulation  # noqa: E402
from scenario_loader import load_scenario, list_tasks  # noqa: E402
from simulation import DispatchSimulation  # noqa: E402


def test_load_all_scenarios():
    """All three task YAMLs load and parse correctly."""
    for name in list_tasks():
        s = load_scenario(name)
        assert s["name"] == name
        assert "calls" in s
        assert "units" in s
        assert "hospitals" in s


def test_easy_scenario_starts_with_no_pending():
    """Reset on easy: at t=0 there should be 0 pending calls (first call at t=1)."""
    sim = DispatchSimulation(load_scenario("easy"), seed=42)
    assert sim.current_time == 0
    assert len(sim.get_pending_calls()) == 0
    assert sim.total_calls() == 5


def test_easy_first_call_arrives_at_t1():
    sim = DispatchSimulation(load_scenario("easy"), seed=42)
    sim.advance_time(1)
    assert len(sim.get_pending_calls()) == 1


def test_dispatch_marks_unit_busy():
    sim = DispatchSimulation(load_scenario("easy"), seed=42)
    sim.advance_time(1)  # release CALL-001
    pending = sim.get_pending_calls()
    assert len(pending) == 1
    call = pending[0]

    # Pick any unit
    unit = next(iter(sim.units.values()))
    n_avail_before = len(sim.get_available_units())
    reward, msg = sim.dispatch(call.call_id, unit.unit_id)
    assert "Dispatched" in msg
    assert reward >= 0.0
    assert len(sim.get_available_units()) == n_avail_before - 1


def test_dispatch_unknown_call_returns_negative():
    sim = DispatchSimulation(load_scenario("easy"), seed=42)
    sim.advance_time(1)
    reward, msg = sim.dispatch("CALL-NONEXISTENT", "ALS-1")
    assert reward < 0
    assert "not found" in msg.lower()


def test_dispatch_unavailable_unit_returns_negative():
    sim = DispatchSimulation(load_scenario("easy"), seed=42)
    sim.advance_time(1)
    pending = sim.get_pending_calls()
    sim.dispatch(pending[0].call_id, "ALS-1")
    # ALS-1 is now busy. Try again with a fresh call (after time passes).
    sim.advance_time(2)
    pending2 = sim.get_pending_calls()
    if pending2:
        reward, msg = sim.dispatch(pending2[0].call_id, "ALS-1")
        assert reward < 0


def test_episode_eventually_completes():
    """Run the easy scenario forward time-only — episode should hit time limit."""
    sim = DispatchSimulation(load_scenario("easy"), seed=42)
    sim.advance_time(60)  # over the 30-min limit
    assert sim.episode_done is True


def test_silent_agent_easy_scores_low():
    """Doing nothing on easy should score below 0.15."""
    sim = DispatchSimulation(load_scenario("easy"), seed=42)
    sim.advance_time(60)
    r = grade_simulation(sim)
    assert r.total < 0.15, f"silent agent scored {r.total}"


def _run_simple_heuristic(sim: "DispatchSimulation", max_steps: int = 200) -> None:
    """Tiny rule-based agent used by tests to confirm the env discriminates.

    Picks the most-critical pending call and assigns the most effective
    available unit to it; otherwise waits one minute.
    """
    from reward import get_effectiveness
    from utils import calculate_distance

    steps = 0
    while not sim.episode_done and steps < max_steps:
        pending = sim.get_pending_calls()
        avail = sim.get_available_units()
        if not pending or not avail:
            sim.advance_time(1)
            steps += 1
            continue

        sev_w = {1: 6.0, 2: 4.0, 3: 2.0, 4: 1.0, 5: 0.5}
        best = None
        best_score = float("-inf")
        for call in pending:
            w = sev_w.get(
                call.reported_severity.value if call.reported_severity else 5, 1.0
            )
            for unit in avail:
                eff = get_effectiveness(unit.unit_type, call.true_type)
                if eff < 0.30:
                    continue
                dist = calculate_distance(unit.position, call.location)
                penalty = 0.5 if (
                    unit.unit_type.value == "als_ambulance"
                    and call.true_severity.value >= 4
                ) else 0.0
                score = w * eff - penalty - 0.05 * dist
                if score > best_score:
                    best_score = score
                    best = (call, unit)

        if best is None:
            sim.advance_time(1)
        else:
            call, unit = best
            sim.dispatch(call_id=call.call_id, unit_id=unit.unit_id)
            sim.advance_time(1)
        steps += 1


def test_heuristic_easy_beats_silent():
    """A simple heuristic agent should score noticeably above silent."""
    sim = DispatchSimulation(load_scenario("easy"), seed=42)
    _run_simple_heuristic(sim)
    r = grade_simulation(sim)
    assert r.total > 0.30, f"heuristic on easy scored only {r.total}"


def test_difficulty_progression():
    """Heuristic should score higher on easy than on hard."""
    scores = {}
    for task in ("easy", "medium", "hard"):
        sim = DispatchSimulation(load_scenario(task), seed=42)
        _run_simple_heuristic(sim)
        scores[task] = grade_simulation(sim).total
    assert scores["easy"] > scores["hard"], (
        f"heuristic easy {scores['easy']} should beat hard {scores['hard']}"
    )


if __name__ == "__main__":
    test_load_all_scenarios()
    test_easy_scenario_starts_with_no_pending()
    test_easy_first_call_arrives_at_t1()
    test_dispatch_marks_unit_busy()
    test_dispatch_unknown_call_returns_negative()
    test_dispatch_unavailable_unit_returns_negative()
    test_episode_eventually_completes()
    test_silent_agent_easy_scores_low()
    test_heuristic_easy_beats_silent()
    test_difficulty_progression()
    print("All simulation tests passed!")
