"""Reward function. Clinical survival curves + episode-level scoring.

The math is grounded in real EMS literature:
  - Cardiac arrest survival drops ~10% per minute without CPR
  - Trauma is the "golden hour" — sigmoid decay centered around 45 min
  - Stroke: time is brain, ~1.9M neurons lost per minute
  - Fire damage doubles roughly every minute

References:
  - Larsen et al. (1993). Predicting survival from out-of-hospital cardiac arrest
  - Saver (2006). Time is brain — quantified
  - Lerner & Moscati (2001). The golden hour: scientific fact or medical urban legend?
"""

from __future__ import annotations

import math
from typing import List, Optional

from models import (
    EmergencyCall,
    EmergencyType,
    EmergencyUnit,
    Hospital,
    Reward,
    UnitType,
)

# ----------------------------------------------------------------------------
# Survival / outcome curves: response_time (minutes) -> outcome score [0, 1]
# ----------------------------------------------------------------------------


def cardiac_arrest_survival(t: float) -> float:
    """~10% loss per minute. 1 min ≈ 0.95, 4 min ≈ 0.70, 12 min ≈ 0.31."""
    if t <= 1.0:
        return 0.95
    return max(0.0, 0.95 * math.exp(-0.10 * (t - 1.0)))


def trauma_survival(t: float) -> float:
    """Sigmoid centered around 45 min (golden hour)."""
    if t <= 5.0:
        return 0.95
    return max(0.05, 0.95 / (1.0 + math.exp(0.08 * (t - 45.0))))


def stroke_survival(t: float) -> float:
    """Steep neural decay; treatment window ~4.5h but outcomes degrade fast."""
    if t <= 2.0:
        return 0.92
    return max(0.05, 0.92 * math.exp(-0.04 * (t - 2.0)))


def fire_outcome(t: float) -> float:
    """Property saved. Fire roughly doubles every minute."""
    if t <= 2.0:
        return 0.90
    return max(0.02, 0.90 * math.exp(-0.15 * (t - 2.0)))


def breathing_outcome(t: float) -> float:
    """Less acute than cardiac arrest, but still time-sensitive."""
    if t <= 3.0:
        return 0.90
    return max(0.10, 0.90 * math.exp(-0.05 * (t - 3.0)))


def minor_injury_outcome(t: float) -> float:
    """Stable patient. Outcome barely degrades."""
    return max(0.50, 0.98 - 0.005 * t)


def mental_health_outcome(t: float) -> float:
    """De-escalation success rate proxy."""
    if t <= 10.0:
        return 0.85
    return max(0.20, 0.85 * math.exp(-0.03 * (t - 10.0)))


SURVIVAL_FUNCTIONS = {
    EmergencyType.CARDIAC_ARREST: cardiac_arrest_survival,
    EmergencyType.TRAUMA: trauma_survival,
    EmergencyType.STROKE: stroke_survival,
    EmergencyType.FIRE: fire_outcome,
    EmergencyType.BREATHING: breathing_outcome,
    EmergencyType.MINOR_INJURY: minor_injury_outcome,
    EmergencyType.MENTAL_HEALTH: mental_health_outcome,
}


def calculate_outcome(emergency_type: EmergencyType, response_time_minutes: float) -> float:
    """Return outcome score 0.0–1.0 for an emergency type and response time."""
    return SURVIVAL_FUNCTIONS[emergency_type](response_time_minutes)


# ----------------------------------------------------------------------------
# Unit-type effectiveness multipliers
# ----------------------------------------------------------------------------

UNIT_EFFECTIVENESS = {
    (UnitType.ALS_AMBULANCE, EmergencyType.CARDIAC_ARREST): 1.0,
    (UnitType.BLS_AMBULANCE, EmergencyType.CARDIAC_ARREST): 0.50,
    (UnitType.ALS_AMBULANCE, EmergencyType.TRAUMA): 1.0,
    (UnitType.BLS_AMBULANCE, EmergencyType.TRAUMA): 0.70,
    (UnitType.ALS_AMBULANCE, EmergencyType.STROKE): 1.0,
    (UnitType.BLS_AMBULANCE, EmergencyType.STROKE): 0.60,
    (UnitType.FIRE_ENGINE, EmergencyType.FIRE): 1.0,
    (UnitType.ALS_AMBULANCE, EmergencyType.FIRE): 0.20,
    (UnitType.BLS_AMBULANCE, EmergencyType.FIRE): 0.10,
    (UnitType.POLICE, EmergencyType.MENTAL_HEALTH): 0.80,
    (UnitType.ALS_AMBULANCE, EmergencyType.MENTAL_HEALTH): 0.90,
    (UnitType.BLS_AMBULANCE, EmergencyType.MINOR_INJURY): 1.0,
    (UnitType.ALS_AMBULANCE, EmergencyType.MINOR_INJURY): 1.0,
    (UnitType.ALS_AMBULANCE, EmergencyType.BREATHING): 1.0,
    (UnitType.BLS_AMBULANCE, EmergencyType.BREATHING): 0.70,
    (UnitType.POLICE, EmergencyType.TRAUMA): 0.30,
}


def get_effectiveness(unit_type: UnitType, emergency_type: EmergencyType) -> float:
    """Effectiveness multiplier for (unit, emergency). Default 0.10 if unmatched."""
    return UNIT_EFFECTIVENESS.get((unit_type, emergency_type), 0.10)


# ----------------------------------------------------------------------------
# Hospital quality bonus (FIX #2 + #3 from review: redirect + capacity matter)
# ----------------------------------------------------------------------------

HOSPITAL_SPECIALTY_BY_TYPE = {
    EmergencyType.CARDIAC_ARREST: "has_cardiac_unit",
    EmergencyType.STROKE: "has_stroke_unit",
    EmergencyType.TRAUMA: "has_trauma_center",
}


def hospital_modifier(call: EmergencyCall, hospital: Optional[Hospital]) -> float:
    """Multiplier in [0.85, 1.05] for hospital match quality.

    - Specialty match (e.g. cardiac unit for cardiac arrest): +0.05
    - On diversion or zero beds: -0.15 (patient delayed in transfer)
    - Default (no hospital chosen): 1.0
    """
    if hospital is None:
        return 1.0
    if hospital.on_diversion or hospital.available_beds <= 0:
        return 0.85
    needed_attr = HOSPITAL_SPECIALTY_BY_TYPE.get(call.true_type)
    if needed_attr and getattr(hospital, needed_attr, False):
        return 1.05
    return 1.0


# ----------------------------------------------------------------------------
# Per-call outcome (used by simulation when a unit reaches a call)
# ----------------------------------------------------------------------------


def calculate_call_outcome(
    call: EmergencyCall,
    unit: EmergencyUnit,
    response_time: float,
    hospital: Optional[Hospital] = None,
) -> float:
    """Final outcome [0,1] = survival_curve × unit_effectiveness × hospital_modifier."""
    base = calculate_outcome(call.true_type, response_time)
    eff = get_effectiveness(unit.unit_type, call.true_type)
    hosp = hospital_modifier(call, hospital)
    return max(0.0, min(1.0, base * eff * hosp))


# ----------------------------------------------------------------------------
# Episode reward
# ----------------------------------------------------------------------------

SEVERITY_WEIGHTS = {1: 3.0, 2: 2.0, 3: 1.5, 4: 1.0, 5: 0.5}


def calculate_episode_reward(
    completed_calls: List[dict],
    timed_out_calls: List[dict],
    total_calls: int,
    dispatches: List[dict],
) -> Reward:
    """Compose final reward in [0.0, 1.0] from four weighted components.

    Weights: survival 0.60 | efficiency 0.15 | triage 0.15 | penalty 0.10
    """

    # 1. SURVIVAL SCORE — severity-weighted average outcome
    total_weighted_outcome = 0.0
    total_weight = 0.0
    for c in completed_calls:
        w = SEVERITY_WEIGHTS[c["true_severity"]]
        total_weighted_outcome += c["outcome_score"] * w
        total_weight += w
    for c in timed_out_calls:
        w = SEVERITY_WEIGHTS[c["true_severity"]]
        total_weighted_outcome += 0.0  # explicit: timeouts contribute 0 outcome
        total_weight += w
    survival_score = (total_weighted_outcome / total_weight) if total_weight > 0 else 0.0

    # 2. EFFICIENCY SCORE — dispatched ratio, penalised for ALS waste on low-acuity
    calls_dispatched = len(completed_calls)
    dispatch_ratio = (calls_dispatched / total_calls) if total_calls > 0 else 0.0
    als_waste_count = sum(
        1
        for d in dispatches
        if d.get("unit_type") == "als_ambulance" and d.get("true_severity", 5) >= 4
    )
    als_waste_penalty = min(0.30, als_waste_count * 0.05)
    efficiency_score = max(0.0, dispatch_ratio - als_waste_penalty)

    # 3. TRIAGE ACCURACY — fraction of severity-1 calls dispatched within 25%
    #    of their timeout window. Cleaner and more interpretable than O(n²) pair
    #    comparisons. Returns 0.0 (not 1.0) if there were no critical calls AND
    #    no dispatches happened — silent agents don't get free credit.
    critical_dispatches = [d for d in dispatches if d.get("true_severity") == 1]
    if not critical_dispatches:
        # No critical calls in dispatches.
        # If there were also no critical calls overall, give full credit.
        critical_completed = [c for c in completed_calls if c.get("true_severity") == 1]
        critical_timed_out = [c for c in timed_out_calls if c.get("true_severity") == 1]
        if not critical_completed and not critical_timed_out:
            triage_accuracy = 1.0
        else:
            triage_accuracy = 0.0  # critical calls existed but none were dispatched
    else:
        on_time = sum(
            1
            for d in critical_dispatches
            if (d["dispatch_time"] - d.get("arrival_time", d["dispatch_time"]))
            <= max(1, d.get("timeout_window", 20) * 0.25)
        )
        triage_accuracy = on_time / len(critical_dispatches)

    # 4. PENALTY — for clearly bad actions
    penalty = 0.0
    critical_timeouts = sum(1 for c in timed_out_calls if c["true_severity"] <= 2)
    penalty += critical_timeouts * 0.08
    wrong_unit_count = sum(1 for d in dispatches if d.get("effectiveness", 1.0) < 0.20)
    penalty += wrong_unit_count * 0.05
    penalty = min(penalty, 0.50)

    total = (
        0.60 * survival_score
        + 0.15 * efficiency_score
        + 0.15 * triage_accuracy
        - 0.10 * penalty
    )
    total = max(0.0, min(1.0, total))

    return Reward(
        total=total,
        survival_score=survival_score,
        efficiency_score=efficiency_score,
        triage_accuracy=triage_accuracy,
        penalty=penalty,
        details=(
            f"survival={survival_score:.3f}  "
            f"efficiency={efficiency_score:.3f}  "
            f"triage={triage_accuracy:.3f}  "
            f"penalty=-{penalty:.3f}  "
            f"=> total={total:.3f}"
        ),
    )
