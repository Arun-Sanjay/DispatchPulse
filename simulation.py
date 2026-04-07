"""DispatchSimulation engine. Pure Python, deterministic, seedable."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from models import (
    EmergencyCall,
    EmergencyType,
    EmergencyUnit,
    Hospital,
    Position,
    Severity,
    UnitStatus,
    UnitType,
    WorldConfig,
)
from reward import calculate_call_outcome, get_effectiveness
from utils import (
    calculate_distance,
    calculate_eta,
    generate_caller_text,
    get_capable_units,
    get_optimal_unit,
)


def _parse_severity(value) -> Severity:
    if isinstance(value, Severity):
        return value
    return Severity(int(value))


def generate_call_schedule(scenario: dict, seed: int) -> List[EmergencyCall]:
    """Build a deterministic list of EmergencyCall objects from a scenario dict."""
    rng = np.random.RandomState(seed)
    calls: List[EmergencyCall] = []
    grid_size = scenario.get("grid_size", 10.0)
    inaccuracy = float(scenario.get("caller_inaccuracy", 0.0))

    for idx, call_cfg in enumerate(scenario["calls"], start=1):
        true_type = EmergencyType(call_cfg["type"])
        true_severity = _parse_severity(call_cfg["severity"])

        if inaccuracy > 0 and rng.random() < inaccuracy:
            other_types = [t for t in EmergencyType if t != true_type]
            reported_type = EmergencyType(str(rng.choice([t.value for t in other_types])))
            shifted = max(1, min(5, true_severity.value + int(rng.randint(-1, 2))))
            reported_severity = Severity(shifted)
        else:
            reported_type = true_type
            reported_severity = true_severity

        location = Position(
            x=round(float(rng.uniform(0.5, grid_size - 0.5)), 1),
            y=round(float(rng.uniform(0.5, grid_size - 0.5)), 1),
        )
        caller_text = generate_caller_text(true_type, reported_type, rng)

        calls.append(
            EmergencyCall(
                call_id=f"CALL-{idx:03d}",
                timestamp=int(call_cfg["arrival_minute"]),
                caller_description=caller_text,
                location=location,
                true_type=true_type,
                true_severity=true_severity,
                reported_type=reported_type,
                reported_severity=reported_severity,
                requires_unit_types=get_capable_units(true_type),
                optimal_unit_type=get_optimal_unit(true_type),
            )
        )

    calls.sort(key=lambda c: c.timestamp)
    return calls


# Scene-time table: how long a unit stays on scene treating a call
SCENE_TIME_MINUTES = {
    EmergencyType.CARDIAC_ARREST: 20,
    EmergencyType.TRAUMA: 25,
    EmergencyType.STROKE: 15,
    EmergencyType.FIRE: 30,
    EmergencyType.BREATHING: 15,
    EmergencyType.MINOR_INJURY: 10,
    EmergencyType.MENTAL_HEALTH: 20,
}


class DispatchSimulation:
    """Discrete-time simulation of an emergency dispatch episode."""

    def __init__(self, scenario: dict, seed: int = 42) -> None:
        self.scenario_name: str = scenario.get("name", "unnamed")
        self.scenario: dict = scenario
        self.seed: int = seed
        self.rng = np.random.RandomState(seed)

        world_cfg = scenario.get("world_config", {})
        self.config = WorldConfig(**world_cfg)

        self.current_time: int = 0
        self.episode_done: bool = False

        self.all_calls: List[EmergencyCall] = generate_call_schedule(scenario, seed)
        self.active_calls: List[EmergencyCall] = []
        self.completed_calls: List[dict] = []
        self.timed_out_calls: List[dict] = []
        self.dispatches: List[dict] = []

        self.units: Dict[str, EmergencyUnit] = {}
        for unit_cfg in scenario["units"]:
            unit = EmergencyUnit(**unit_cfg)
            self.units[unit.unit_id] = unit

        self.hospitals: Dict[str, Hospital] = {}
        for hosp_cfg in scenario["hospitals"]:
            hosp = Hospital(**hosp_cfg)
            self.hospitals[hosp.hospital_id] = hosp

        self.call_index: int = 0
        # Release any calls scheduled for time 0
        self._release_due_calls()

    # ------------------------------------------------------------------
    # Time advancement
    # ------------------------------------------------------------------

    def _release_due_calls(self) -> None:
        """Move calls whose arrival time has passed into the active queue."""
        while (
            self.call_index < len(self.all_calls)
            and self.all_calls[self.call_index].timestamp <= self.current_time
        ):
            call = self.all_calls[self.call_index]
            call.active = True
            self.active_calls.append(call)
            self.call_index += 1

    def advance_time(self, minutes: int = 1) -> None:
        """Step the simulation forward by ``minutes`` discrete minutes."""
        if self.episode_done:
            return
        minutes = max(1, int(minutes))
        for _ in range(minutes):
            self.current_time += 1
            self._tick_once()
            if self.episode_done:
                break

    def _tick_once(self) -> None:
        """Advance simulation by exactly one minute, updating units & calls."""
        # 1. Move units according to their status
        for unit in self.units.values():
            if unit.status == UnitStatus.EN_ROUTE:
                self._move_unit_toward_call(unit)
            elif unit.status == UnitStatus.ON_SCENE:
                if unit.busy_until is not None and self.current_time >= unit.busy_until:
                    unit.status = UnitStatus.RETURNING
                    unit.assigned_call_id = None
                    unit.assigned_hospital_id = None
            elif unit.status == UnitStatus.RETURNING:
                self._move_unit_toward_base(unit)

        # 2. Time-out any active call that has waited too long
        for call in list(self.active_calls):
            if call.dispatched_unit_id is None:
                wait = self.current_time - call.timestamp
                if wait >= self.config.call_timeout_minutes:
                    call.active = False
                    self.active_calls.remove(call)
                    self.timed_out_calls.append(
                        {
                            "call_id": call.call_id,
                            "true_type": call.true_type.value,
                            "true_severity": call.true_severity.value,
                            "outcome_score": 0.0,
                            "reason": "timed_out",
                        }
                    )

        # 3. Release new calls
        self._release_due_calls()

        # 4. Episode end conditions
        if self.current_time >= self.config.time_limit_minutes:
            self._finalize_episode("time_limit")
            return

        no_more_incoming = self.call_index >= len(self.all_calls)
        no_pending = all(c.dispatched_unit_id is not None for c in self.active_calls)
        all_units_idle = all(u.status == UnitStatus.AVAILABLE for u in self.units.values())
        if no_more_incoming and no_pending and all_units_idle and not self.active_calls:
            self._finalize_episode("all_resolved")

    def _finalize_episode(self, reason: str) -> None:
        """Mark episode done.

        Any remaining call (whether un-dispatched OR dispatched but the unit
        never actually arrived on scene) is recorded as a timeout — the agent
        failed to deliver care in time, so the patient outcome is 0.0.
        """
        self.episode_done = True
        for call in list(self.active_calls):
            self.timed_out_calls.append(
                {
                    "call_id": call.call_id,
                    "true_type": call.true_type.value,
                    "true_severity": call.true_severity.value,
                    "outcome_score": 0.0,
                    "reason": reason
                    if call.dispatched_unit_id is None
                    else f"{reason}_in_transit",
                }
            )
        self.active_calls.clear()

    # ------------------------------------------------------------------
    # Unit movement
    # ------------------------------------------------------------------

    def _move_unit_toward_call(self, unit: EmergencyUnit) -> None:
        call = self._get_call_by_id(unit.assigned_call_id) if unit.assigned_call_id else None
        if call is None:
            unit.status = UnitStatus.AVAILABLE
            unit.assigned_call_id = None
            return

        distance_per_step = (unit.speed_kmh / 60.0) * self.config.step_duration_minutes
        dist = calculate_distance(unit.position, call.location)

        if dist <= distance_per_step:
            unit.position = Position(x=call.location.x, y=call.location.y)
            unit.status = UnitStatus.ON_SCENE
            response_time = float(self.current_time - call.timestamp)
            call.response_time = response_time

            hospital = (
                self.hospitals.get(unit.assigned_hospital_id)
                if unit.assigned_hospital_id
                else None
            )

            outcome = calculate_call_outcome(call, unit, response_time, hospital)
            call.outcome_score = outcome
            call.active = False
            if call in self.active_calls:
                self.active_calls.remove(call)

            if hospital is not None and not hospital.on_diversion and hospital.available_beds > 0:
                hospital.available_beds = max(0, hospital.available_beds - 1)
                call.delivered_hospital_id = hospital.hospital_id

            self.completed_calls.append(
                {
                    "call_id": call.call_id,
                    "true_type": call.true_type.value,
                    "true_severity": call.true_severity.value,
                    "response_time": response_time,
                    "outcome_score": outcome,
                    "unit_id": unit.unit_id,
                    "unit_type": unit.unit_type.value,
                    "effectiveness": get_effectiveness(unit.unit_type, call.true_type),
                    "hospital_id": call.delivered_hospital_id,
                }
            )

            scene_time = SCENE_TIME_MINUTES.get(call.true_type, 15)
            unit.busy_until = self.current_time + scene_time
        else:
            ratio = distance_per_step / dist
            unit.position = Position(
                x=round(unit.position.x + (call.location.x - unit.position.x) * ratio, 3),
                y=round(unit.position.y + (call.location.y - unit.position.y) * ratio, 3),
            )

    def _move_unit_toward_base(self, unit: EmergencyUnit) -> None:
        distance_per_step = (unit.speed_kmh / 60.0) * self.config.step_duration_minutes
        dist = calculate_distance(unit.position, unit.base_position)
        if dist <= distance_per_step:
            unit.position = Position(x=unit.base_position.x, y=unit.base_position.y)
            unit.status = UnitStatus.AVAILABLE
            unit.busy_until = None
        else:
            ratio = distance_per_step / dist
            unit.position = Position(
                x=round(unit.position.x + (unit.base_position.x - unit.position.x) * ratio, 3),
                y=round(unit.position.y + (unit.base_position.y - unit.position.y) * ratio, 3),
            )

    # ------------------------------------------------------------------
    # Action handlers (called from the MCP environment)
    # ------------------------------------------------------------------

    def dispatch(
        self, call_id: str, unit_id: str, hospital_id: Optional[str] = None
    ) -> Tuple[float, str]:
        """Dispatch a unit to a call (optionally pre-assigning a destination hospital)."""
        call = self._get_active_undispatched_call(call_id)
        if call is None:
            return -0.05, f"Call {call_id} not found in pending queue."

        unit = self.units.get(unit_id)
        if unit is None:
            return -0.05, f"Unit {unit_id} not found."
        if unit.status != UnitStatus.AVAILABLE:
            return -0.05, f"Unit {unit_id} is {unit.status.value}, cannot dispatch."

        # Treat empty string / whitespace as "no hospital chosen"
        if isinstance(hospital_id, str):
            hospital_id = hospital_id.strip() or None

        chosen_hospital = None
        if hospital_id is not None:
            chosen_hospital = self.hospitals.get(hospital_id)
            if chosen_hospital is None:
                return -0.02, f"Hospital '{hospital_id}' not found."

        unit.status = UnitStatus.EN_ROUTE
        unit.assigned_call_id = call.call_id
        unit.assigned_hospital_id = hospital_id
        call.dispatched_unit_id = unit.unit_id

        eta = calculate_eta(unit, call.location)
        effectiveness = get_effectiveness(unit.unit_type, call.true_type)

        self.dispatches.append(
            {
                "call_id": call.call_id,
                "unit_id": unit.unit_id,
                "unit_type": unit.unit_type.value,
                "true_type": call.true_type.value,
                "true_severity": call.true_severity.value,
                "arrival_time": call.timestamp,
                "dispatch_time": self.current_time,
                "timeout_window": self.config.call_timeout_minutes,
                "eta": eta,
                "effectiveness": effectiveness,
                "hospital_id": hospital_id,
            }
        )

        msg = (
            f"Dispatched {unit.unit_id} to {call.call_id}. "
            f"ETA {eta:.1f} min. Unit effectiveness for {call.true_type.value}: "
            f"{effectiveness:.0%}."
        )
        if hospital_id is not None and chosen_hospital is not None:
            msg += f" Destination hospital: {chosen_hospital.name}."
        return 0.02 * effectiveness, msg

    def classify(self, call_id: str, severity: int) -> Tuple[float, str]:
        call = self._get_active_undispatched_call(call_id)
        if call is None:
            return -0.02, f"Call {call_id} not in pending queue."
        try:
            new_sev = Severity(int(severity))
        except ValueError:
            return -0.02, f"Invalid severity {severity}; must be 1-5."
        old = call.reported_severity
        call.reported_severity = new_sev
        return 0.01, f"Reclassified {call_id} severity from {old} to {new_sev.value}."

    def callback(self, call_id: str, question: str) -> Tuple[float, str]:
        call = self._get_active_undispatched_call(call_id)
        if call is None:
            return -0.02, f"Call {call_id} not in pending queue."
        # 70% chance the caller clarifies; 30% they're too distressed
        if self.rng.random() < 0.70:
            call.reported_type = call.true_type
            call.reported_severity = call.true_severity
            return (
                0.02,
                f"Caller for {call.call_id} confirms: this is a {call.true_type.value}, "
                f"severity {call.true_severity.value}.",
            )
        return 0.0, f"Caller for {call.call_id} is too distressed to give clear info."

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def _get_call_by_id(self, call_id: str) -> Optional[EmergencyCall]:
        for c in self.all_calls:
            if c.call_id == call_id:
                return c
        return None

    def _get_active_undispatched_call(self, call_id: str) -> Optional[EmergencyCall]:
        for c in self.active_calls:
            if c.call_id == call_id and c.dispatched_unit_id is None:
                return c
        return None

    def get_pending_calls(self) -> List[EmergencyCall]:
        return [c for c in self.active_calls if c.dispatched_unit_id is None]

    def get_available_units(self) -> List[EmergencyUnit]:
        return [u for u in self.units.values() if u.status == UnitStatus.AVAILABLE]

    def total_calls(self) -> int:
        return len(self.all_calls)
