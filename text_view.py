"""Render a DispatchSimulation as a human-readable text view for the LLM agent."""

from __future__ import annotations

from typing import List

from models import EmergencyCall, EmergencyUnit, Hospital, UnitStatus
from simulation import DispatchSimulation
from utils import calculate_distance, calculate_eta

# Maximum number of calls / units / outcomes to show in the text view.
# Truncation prevents context blow-up on the hard task (30 calls).
MAX_PENDING_CALLS = 8
MAX_BUSY_UNITS = 8
MAX_RECENT_OUTCOMES = 3


def _format_call(call: EmergencyCall, sim: DispatchSimulation) -> List[str]:
    wait = sim.current_time - call.timestamp
    rt = call.reported_type.value if call.reported_type else "unknown"
    rs = call.reported_severity.value if call.reported_severity else "?"
    return [
        f'  {call.call_id} [t={call.timestamp}min] "{call.caller_description}"',
        (
            f"    location=({call.location.x}, {call.location.y})  "
            f"reported={rt}/sev{rs}  waiting={wait}min"
        ),
    ]


def _format_unit(unit: EmergencyUnit, sim: DispatchSimulation, pending: list) -> str:
    base = (
        f"  {unit.unit_id:7s} | {unit.unit_type.value:14s} | "
        f"pos=({unit.position.x:.1f}, {unit.position.y:.1f})"
    )
    if pending:
        closest = min(pending, key=lambda c: calculate_distance(unit.position, c.location))
        eta = calculate_eta(unit, closest.location)
        base += f"  closest_call_eta={eta:.1f}min ({closest.call_id})"
    return base


def _format_busy_unit(unit: EmergencyUnit) -> str:
    detail = unit.status.value
    if unit.assigned_call_id:
        detail += f" -> {unit.assigned_call_id}"
    if unit.busy_until is not None and unit.status == UnitStatus.ON_SCENE:
        detail += f" (free at t={unit.busy_until}min)"
    return f"  {unit.unit_id:7s} | {unit.unit_type.value:14s} | {detail}"


def _format_hospital(hosp: Hospital) -> str:
    specs = []
    if hosp.has_trauma_center:
        specs.append("trauma")
    if hosp.has_cardiac_unit:
        specs.append("cardiac")
    if hosp.has_stroke_unit:
        specs.append("stroke")
    status = "DIVERSION" if hosp.on_diversion else "open"
    return (
        f"  {hosp.hospital_id} {hosp.name} ({hosp.position.x},{hosp.position.y})  "
        f"beds={hosp.available_beds}/{hosp.capacity}  "
        f"specialties=[{','.join(specs) or 'none'}]  status={status}"
    )


def render_dispatch_center(sim: DispatchSimulation, task_name: str) -> str:
    """Pretty-print the current state for the LLM agent."""
    lines: List[str] = []
    lines.append("=== DISPATCHPULSE DISPATCH CENTER ===")
    lines.append(
        f"task={task_name}  time={sim.current_time}min/"
        f"{sim.config.time_limit_minutes}min  "
        f"scenario={sim.scenario_name}"
    )
    lines.append("")

    # Pending calls (sorted by reported severity, then arrival time)
    pending = sim.get_pending_calls()
    pending_sorted = sorted(
        pending,
        key=lambda c: (
            c.reported_severity.value if c.reported_severity else 5,
            c.timestamp,
        ),
    )
    lines.append(f"PENDING CALLS ({len(pending_sorted)} total):")
    if not pending_sorted:
        lines.append("  (none)")
    for call in pending_sorted[:MAX_PENDING_CALLS]:
        lines.extend(_format_call(call, sim))
    if len(pending_sorted) > MAX_PENDING_CALLS:
        hidden = len(pending_sorted) - MAX_PENDING_CALLS
        lines.append(f"  ... and {hidden} more lower-priority calls")
    lines.append("")

    # Available units
    available = sim.get_available_units()
    lines.append(f"AVAILABLE UNITS ({len(available)} total):")
    if not available:
        lines.append("  (none — all units busy)")
    for unit in available:
        lines.append(_format_unit(unit, sim, pending_sorted[:MAX_PENDING_CALLS]))
    lines.append("")

    # Busy units
    busy = [u for u in sim.units.values() if u.status != UnitStatus.AVAILABLE]
    if busy:
        lines.append(f"BUSY UNITS ({len(busy)} total):")
        for unit in busy[:MAX_BUSY_UNITS]:
            lines.append(_format_busy_unit(unit))
        if len(busy) > MAX_BUSY_UNITS:
            lines.append(f"  ... and {len(busy) - MAX_BUSY_UNITS} more busy units")
        lines.append("")

    # Hospitals
    lines.append("HOSPITALS:")
    for hosp in sim.hospitals.values():
        lines.append(_format_hospital(hosp))
    lines.append("")

    # Recent outcomes
    if sim.completed_calls:
        recent = sim.completed_calls[-MAX_RECENT_OUTCOMES:]
        lines.append("RECENT OUTCOMES:")
        for r in recent:
            mark = "OK " if r["outcome_score"] >= 0.5 else "BAD"
            lines.append(
                f"  [{mark}] {r['call_id']} {r['true_type']} sev{r['true_severity']} "
                f"response={r['response_time']:.1f}min outcome={r['outcome_score']:.2f}"
            )
        lines.append("")

    # Stats footer
    lines.append(
        f"STATS: total={sim.total_calls()}  completed={len(sim.completed_calls)}  "
        f"timed_out={len(sim.timed_out_calls)}  pending={len(pending_sorted)}"
    )
    if sim.episode_done:
        lines.append("EPISODE: DONE")
    return "\n".join(lines)
