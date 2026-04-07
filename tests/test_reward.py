"""Calibration tests for the reward function — these double as documentation
of the survival curves and prove the env discriminates between good and bad
agents.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from models import EmergencyCall, EmergencyType, EmergencyUnit, Hospital, Position, Severity, UnitStatus, UnitType  # noqa: E402
from reward import (  # noqa: E402
    calculate_call_outcome,
    calculate_episode_reward,
    calculate_outcome,
    cardiac_arrest_survival,
    fire_outcome,
    get_effectiveness,
    hospital_modifier,
    stroke_survival,
    trauma_survival,
)


# ----------------------------------------------------------------------------
# Survival curves
# ----------------------------------------------------------------------------


def test_cardiac_arrest_curve():
    assert cardiac_arrest_survival(1.0) == 0.95
    assert cardiac_arrest_survival(4.0) > 0.65 and cardiac_arrest_survival(4.0) < 0.75
    assert cardiac_arrest_survival(12.0) < 0.35
    assert cardiac_arrest_survival(30.0) < 0.10


def test_trauma_golden_hour():
    assert trauma_survival(5.0) == 0.95
    assert trauma_survival(15.0) > 0.85
    assert trauma_survival(45.0) > 0.40 and trauma_survival(45.0) < 0.55
    assert trauma_survival(90.0) < 0.15


def test_stroke_curve():
    assert stroke_survival(2.0) == 0.92
    assert stroke_survival(20.0) < 0.50
    assert stroke_survival(60.0) < 0.15


def test_fire_doubles_per_minute():
    assert fire_outcome(2.0) == 0.90
    assert fire_outcome(8.0) < 0.45
    assert fire_outcome(20.0) < 0.10


# ----------------------------------------------------------------------------
# Effectiveness lookup
# ----------------------------------------------------------------------------


def test_als_better_than_bls_for_cardiac():
    als = get_effectiveness(UnitType.ALS_AMBULANCE, EmergencyType.CARDIAC_ARREST)
    bls = get_effectiveness(UnitType.BLS_AMBULANCE, EmergencyType.CARDIAC_ARREST)
    assert als > bls
    assert als == 1.0
    assert bls == 0.5


def test_fire_engine_only_for_fire():
    assert get_effectiveness(UnitType.FIRE_ENGINE, EmergencyType.FIRE) == 1.0
    # ALS dispatched to a fire is mostly useless
    assert get_effectiveness(UnitType.ALS_AMBULANCE, EmergencyType.FIRE) < 0.3


# ----------------------------------------------------------------------------
# Hospital modifier
# ----------------------------------------------------------------------------


def _mk_call(t: EmergencyType) -> EmergencyCall:
    return EmergencyCall(
        call_id="C1",
        timestamp=0,
        caller_description="x",
        location=Position(x=1, y=1),
        true_type=t,
        true_severity=Severity.CRITICAL,
        reported_type=t,
        reported_severity=Severity.CRITICAL,
        optimal_unit_type=UnitType.ALS_AMBULANCE,
    )


def _mk_hospital(**flags) -> Hospital:
    return Hospital(
        hospital_id="H",
        name="X",
        position=Position(x=1, y=1),
        capacity=10,
        available_beds=5,
        has_trauma_center=flags.get("trauma", False),
        has_cardiac_unit=flags.get("cardiac", False),
        has_stroke_unit=flags.get("stroke", False),
        on_diversion=flags.get("diversion", False),
    )


def test_specialty_hospital_helps():
    call = _mk_call(EmergencyType.CARDIAC_ARREST)
    cardiac = _mk_hospital(cardiac=True)
    generic = _mk_hospital()
    assert hospital_modifier(call, cardiac) > 1.0
    assert hospital_modifier(call, generic) == 1.0


def test_diversion_hospital_hurts():
    call = _mk_call(EmergencyType.CARDIAC_ARREST)
    diverted = _mk_hospital(cardiac=True, diversion=True)
    assert hospital_modifier(call, diverted) < 1.0


# ----------------------------------------------------------------------------
# End-to-end episode reward
# ----------------------------------------------------------------------------


def test_perfect_episode_scores_high():
    """A perfect agent on a small scenario should score >= 0.80."""
    completed = [
        {
            "call_id": "C1",
            "true_type": "cardiac_arrest",
            "true_severity": 1,
            "response_time": 2.0,
            "outcome_score": 0.90,
            "unit_id": "ALS-1",
            "unit_type": "als_ambulance",
            "effectiveness": 1.0,
        },
        {
            "call_id": "C2",
            "true_type": "trauma",
            "true_severity": 2,
            "response_time": 5.0,
            "outcome_score": 0.95,
            "unit_id": "BLS-1",
            "unit_type": "bls_ambulance",
            "effectiveness": 0.7,
        },
    ]
    dispatches = [
        {
            "call_id": "C1",
            "unit_type": "als_ambulance",
            "true_type": "cardiac_arrest",
            "true_severity": 1,
            "arrival_time": 0,
            "dispatch_time": 1,
            "timeout_window": 20,
            "effectiveness": 1.0,
        },
        {
            "call_id": "C2",
            "unit_type": "bls_ambulance",
            "true_type": "trauma",
            "true_severity": 2,
            "arrival_time": 2,
            "dispatch_time": 3,
            "timeout_window": 20,
            "effectiveness": 0.7,
        },
    ]
    r = calculate_episode_reward(completed, [], 2, dispatches)
    assert r.total >= 0.80, f"perfect episode scored only {r.total}"
    assert r.penalty == 0.0


def test_silent_agent_scores_near_zero():
    """An agent that does NOTHING — every call times out — should score < 0.15."""
    timed_out = [
        {"call_id": "C1", "true_type": "cardiac_arrest", "true_severity": 1,
         "outcome_score": 0.0, "reason": "timed_out"},
        {"call_id": "C2", "true_type": "trauma", "true_severity": 2,
         "outcome_score": 0.0, "reason": "timed_out"},
    ]
    r = calculate_episode_reward([], timed_out, 2, [])
    # No dispatches => triage_accuracy must be 0.0 (the silent-agent loophole fix)
    assert r.triage_accuracy == 0.0
    assert r.total < 0.15, f"silent agent scored {r.total}, expected near zero"


def test_wrong_unit_penalised():
    """Sending a fire engine to a cardiac arrest must score worse than ALS."""
    bad = calculate_call_outcome(
        _mk_call(EmergencyType.CARDIAC_ARREST),
        EmergencyUnit(
            unit_id="FIRE-1",
            unit_type=UnitType.FIRE_ENGINE,
            position=Position(x=0, y=0),
            base_position=Position(x=0, y=0),
            speed_kmh=50,
        ),
        response_time=2.0,
    )
    good = calculate_call_outcome(
        _mk_call(EmergencyType.CARDIAC_ARREST),
        EmergencyUnit(
            unit_id="ALS-1",
            unit_type=UnitType.ALS_AMBULANCE,
            position=Position(x=0, y=0),
            base_position=Position(x=0, y=0),
            speed_kmh=60,
        ),
        response_time=2.0,
    )
    assert good > bad * 5  # ALS at least 5x better than fire engine here


if __name__ == "__main__":
    # Quick self-test runner
    test_cardiac_arrest_curve()
    test_trauma_golden_hour()
    test_stroke_curve()
    test_fire_doubles_per_minute()
    test_als_better_than_bls_for_cardiac()
    test_fire_engine_only_for_fire()
    test_specialty_hospital_helps()
    test_diversion_hospital_hurts()
    test_perfect_episode_scores_high()
    test_silent_agent_scores_near_zero()
    test_wrong_unit_penalised()
    print("All reward tests passed!")
