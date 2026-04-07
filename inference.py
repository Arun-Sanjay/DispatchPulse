"""DispatchPulse baseline inference script.

ROUND 1 REQUIREMENT: this file must be named ``inference.py`` and live at the
project root. Reproducible — uses fixed seeds.

Usage
-----
    # Run all 3 tasks with the rule-based heuristic agent (no API key needed)
    python inference.py --agent heuristic

    # Run all 3 tasks with an OpenAI LLM agent (needs OPENAI_API_KEY)
    python inference.py --agent llm --model gpt-4o-mini

    # Single task
    python inference.py --agent heuristic --task easy

The script writes ``baseline_results.json`` with per-task scores.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Callable, Dict, List, Optional, Tuple

# Make project root importable when running this script directly.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from grader import grade_simulation  # noqa: E402
from models import Severity, UnitStatus  # noqa: E402
from reward import get_effectiveness  # noqa: E402
from scenario_loader import list_tasks, load_scenario  # noqa: E402
from simulation import DispatchSimulation  # noqa: E402
from text_view import render_dispatch_center  # noqa: E402
from utils import calculate_distance  # noqa: E402

DEFAULT_MAX_STEPS = 250
DEFAULT_SEED = 42


# ============================================================================
# Heuristic agent — no LLM, no API key
# ============================================================================


def _pick_dispatch(sim: DispatchSimulation) -> Optional[Tuple[str, dict]]:
    """Pick a single dispatch action, or return None if no dispatch fits.

    Considers all pending calls and all available units, picking the
    (call, unit) pair that maximises priority * effectiveness / distance.
    """
    pending = sim.get_pending_calls()
    if not pending:
        return None
    available_units = sim.get_available_units()
    if not available_units:
        return None

    severity_weight = {1: 6.0, 2: 4.0, 3: 2.0, 4: 1.0, 5: 0.5}

    best_pair: Optional[Tuple] = None
    best_score: float = float("-inf")

    for call in pending:
        reported_sev = call.reported_severity.value if call.reported_severity else 5
        sev_w = severity_weight.get(reported_sev, 1.0)

        for unit in available_units:
            eff = get_effectiveness(unit.unit_type, call.true_type)
            if eff < 0.30:
                continue  # totally inappropriate (e.g. ALS to fire at 0.20)
            dist = calculate_distance(unit.position, call.location)
            # Reserve ALS for higher-acuity calls — keep them in reserve
            # for severity 1-2 emergencies that may arrive next.
            als_penalty = 0.0
            if (
                unit.unit_type.value == "als_ambulance"
                and call.true_severity.value >= 4
            ):
                als_penalty = 0.5
            score = sev_w * eff - als_penalty - 0.05 * dist
            if score > best_score:
                best_score = score
                best_pair = (call, unit)

    if best_pair is None:
        return None
    call, unit = best_pair

    # Pick the best matching hospital (only for medical emergencies)
    chosen_hospital: Optional[str] = None
    if call.true_type.value in {"cardiac_arrest", "stroke", "trauma"}:
        for hosp in sim.hospitals.values():
            if hosp.on_diversion or hosp.available_beds <= 0:
                continue
            if call.true_type.value == "cardiac_arrest" and hosp.has_cardiac_unit:
                chosen_hospital = hosp.hospital_id
                break
            if call.true_type.value == "stroke" and hosp.has_stroke_unit:
                chosen_hospital = hosp.hospital_id
                break
            if call.true_type.value == "trauma" and hosp.has_trauma_center:
                chosen_hospital = hosp.hospital_id
                break

    kwargs = {"call_id": call.call_id, "unit_id": unit.unit_id}
    if chosen_hospital:
        kwargs["hospital_id"] = chosen_hospital
    return "dispatch", kwargs


def heuristic_step(sim: DispatchSimulation) -> Tuple[str, dict]:
    """Pick the next action: a dispatch if any fits, otherwise wait."""
    pick = _pick_dispatch(sim)
    if pick is not None:
        return pick
    return "wait", {"minutes": 1}


def run_heuristic(sim: DispatchSimulation, max_steps: int = DEFAULT_MAX_STEPS) -> None:
    """Run the heuristic agent against the simulation until episode completes."""
    steps = 0
    while not sim.episode_done and steps < max_steps:
        action, kwargs = heuristic_step(sim)
        if action == "dispatch":
            sim.dispatch(**kwargs)
            sim.advance_time(1)
        elif action == "wait":
            sim.advance_time(int(kwargs.get("minutes", 1)))
        steps += 1


# ============================================================================
# LLM agent — calls OpenAI-compatible API
# ============================================================================

SYSTEM_PROMPT = """You are an experienced emergency dispatch coordinator running a 911 communications center. You receive incoming emergency calls and must dispatch the right unit at the right time to maximize patient survival outcomes.

DISPATCHER STANDARD OPERATING PROCEDURE:
1. CRITICAL CALLS FIRST. Severity 1 (cardiac arrest, severe trauma, stroke) is life-threatening — every minute matters. Survival drops ~10% per minute for cardiac arrest.
2. SEND THE RIGHT UNIT. ALS ambulance for cardiac/stroke/severe trauma. BLS ambulance for stable patients and minor injuries. Fire engine for fires only. Police for mental health crises. Sending a fire engine to a heart attack will not help.
3. CONSERVE ALS UNITS. Do not send your only ALS ambulance to a sprained ankle — a cardiac arrest may come in 3 minutes later.
4. CHOOSE THE RIGHT HOSPITAL. For cardiac arrest, pick a hospital with a cardiac unit. For stroke, pick stroke unit. For trauma, pick trauma center. Avoid hospitals on diversion or with zero beds.
5. CALLBACK WHEN UNCLEAR. If a caller's description is ambiguous or you suspect misreporting, use the callback tool.
6. WAIT WHEN APPROPRIATE. If all calls are dispatched and no decisions remain, use wait to skip ahead.

Available tools:
  - view_dispatch_center() - inspect current state (free, no time cost)
  - dispatch(call_id, unit_id, hospital_id="") - send a unit to a call
  - classify(call_id, severity) - reclassify a call's severity (1-5)
  - callback(call_id, question) - phone the caller back for clarification
  - wait(minutes=1) - skip ahead 1-5 minutes when there's nothing to do

You will be evaluated on patient survival outcomes using real clinical curves.
Respond with ONE tool call per turn, formatted as:
  TOOL: <name>
  ARGS: <key>=<value>; <key>=<value>; ...

Example:
  TOOL: dispatch
  ARGS: call_id=CALL-001; unit_id=ALS-1; hospital_id=H1
"""


def parse_llm_action(text: str) -> Tuple[str, dict]:
    """Parse the LLM's response into (tool_name, kwargs).

    Falls back to ``wait`` on parse failure rather than crashing.
    """
    text = text.strip()
    tool_name = "wait"
    kwargs: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("TOOL:"):
            tool_name = line.split(":", 1)[1].strip().lower()
        elif line.upper().startswith("ARGS:"):
            arg_str = line.split(":", 1)[1].strip()
            for pair in arg_str.split(";"):
                pair = pair.strip()
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    kwargs[k.strip()] = v.strip()
    return tool_name, kwargs


def run_llm(
    sim: DispatchSimulation,
    model: str = "gpt-4o-mini",
    max_steps: int = DEFAULT_MAX_STEPS,
    api_base: Optional[str] = None,
) -> None:
    """Run an OpenAI-compatible LLM agent against the simulation."""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError(
            "openai package not installed. Install with: pip install openai"
        ) from e

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN") or "missing-key"
    client_kwargs: Dict[str, str] = {"api_key": api_key}
    if api_base:
        client_kwargs["base_url"] = api_base
    client = OpenAI(**client_kwargs)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    steps = 0

    while not sim.episode_done and steps < max_steps:
        view = render_dispatch_center(sim, sim.scenario_name)
        messages.append({"role": "user", "content": view})

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=200,
                temperature=0.0,
            )
            action_text = resp.choices[0].message.content or ""
        except Exception as e:
            print(f"[warn] LLM call failed: {e}; falling back to wait", file=sys.stderr)
            action_text = "TOOL: wait\nARGS: minutes=1"

        messages.append({"role": "assistant", "content": action_text})

        tool, kwargs = parse_llm_action(action_text)

        try:
            if tool == "dispatch" and "call_id" in kwargs and "unit_id" in kwargs:
                sim.dispatch(
                    call_id=kwargs["call_id"],
                    unit_id=kwargs["unit_id"],
                    hospital_id=kwargs.get("hospital_id", "").strip() or None,
                )
                sim.advance_time(1)
            elif tool == "classify" and "call_id" in kwargs and "severity" in kwargs:
                sim.classify(kwargs["call_id"], int(kwargs["severity"]))
                sim.advance_time(1)
            elif tool == "callback" and "call_id" in kwargs:
                sim.callback(kwargs["call_id"], kwargs.get("question", ""))
                sim.advance_time(1)
            elif tool == "view_dispatch_center":
                pass  # free action
            else:
                minutes = int(kwargs.get("minutes", 1))
                sim.advance_time(max(1, min(minutes, 5)))
        except Exception as e:
            print(f"[warn] Tool call failed: {e}", file=sys.stderr)
            sim.advance_time(1)

        # Trim conversation to keep it short
        if len(messages) > 21:
            messages = messages[:1] + messages[-20:]
        steps += 1


# ============================================================================
# Driver
# ============================================================================


def run_task(
    task_name: str,
    agent: str,
    seed: int = DEFAULT_SEED,
    model: str = "gpt-4o-mini",
    api_base: Optional[str] = None,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> Dict[str, float]:
    """Run a single task end-to-end and return the score breakdown."""
    scenario = load_scenario(task_name)
    sim = DispatchSimulation(scenario, seed=seed)

    if agent == "heuristic":
        run_heuristic(sim, max_steps=max_steps)
    elif agent == "llm":
        run_llm(sim, model=model, max_steps=max_steps, api_base=api_base)
    else:
        raise ValueError(f"Unknown agent: {agent}")

    # Make sure the episode finalizes (in case agent stopped early)
    if not sim.episode_done:
        sim.advance_time(sim.config.time_limit_minutes)

    reward = grade_simulation(sim)
    return {
        "task": task_name,
        "agent": agent,
        "score": reward.total,
        "survival_score": reward.survival_score,
        "efficiency_score": reward.efficiency_score,
        "triage_accuracy": reward.triage_accuracy,
        "penalty": reward.penalty,
        "completed_calls": len(sim.completed_calls),
        "timed_out_calls": len(sim.timed_out_calls),
        "total_calls": sim.total_calls(),
        "details": reward.details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="DispatchPulse baseline runner")
    parser.add_argument(
        "--agent",
        choices=("heuristic", "llm"),
        default="heuristic",
        help="Which baseline agent to run.",
    )
    parser.add_argument(
        "--task",
        choices=list_tasks() + ["all"],
        default="all",
        help="Which task to run (default: all).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model id (only for --agent llm).")
    parser.add_argument("--api-base", default=None, help="OpenAI-compatible base URL (optional).")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--output", default="baseline_results.json")
    args = parser.parse_args()

    tasks_to_run: List[str] = list_tasks() if args.task == "all" else [args.task]

    print("=" * 64)
    print(f"DISPATCHPULSE BASELINE  agent={args.agent}  seed={args.seed}")
    if args.agent == "llm":
        print(f"  model={args.model}")
    print("=" * 64)

    results: Dict[str, dict] = {}
    for t in tasks_to_run:
        print(f"\n[task: {t}] running...")
        result = run_task(
            t,
            agent=args.agent,
            seed=args.seed,
            model=args.model,
            api_base=args.api_base,
            max_steps=args.max_steps,
        )
        results[t] = result
        print(f"  score: {result['score']:.4f}")
        print(f"  detail: {result['details']}")
        print(
            f"  calls: completed={result['completed_calls']} "
            f"timed_out={result['timed_out_calls']} total={result['total_calls']}"
        )

    avg = sum(r["score"] for r in results.values()) / max(1, len(results))
    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    for t, r in results.items():
        print(f"  {t:7s}  score={r['score']:.4f}")
    print(f"  AVERAGE  {avg:.4f}")

    payload = {
        "agent": args.agent,
        "model": args.model if args.agent == "llm" else None,
        "seed": args.seed,
        "average_score": avg,
        "results": results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
