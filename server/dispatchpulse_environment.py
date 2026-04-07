"""DispatchPulse MCP environment.

Inherits from openenv MCPEnvironment and exposes the dispatcher interface
as MCP tools (FastMCP). Each tool advances the simulation by 1 minute,
except `view_dispatch_center` (free inspection) and `wait` (custom n minutes).
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional
from uuid import uuid4

# Make package modules importable when running as `server.app:app`.
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

# Import the OpenEnv base classes.
try:
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State
except ImportError as e:  # pragma: no cover - we still want the file to import for tests
    MCPEnvironment = object  # type: ignore
    Action = object  # type: ignore
    Observation = dict  # type: ignore
    State = dict  # type: ignore
    _OPENENV_IMPORT_ERROR = e
else:
    _OPENENV_IMPORT_ERROR = None

try:
    from fastmcp import FastMCP
except ImportError as e:  # pragma: no cover
    FastMCP = None  # type: ignore
    _FASTMCP_IMPORT_ERROR = e
else:
    _FASTMCP_IMPORT_ERROR = None

from grader import grade_simulation
from scenario_loader import load_scenario, list_tasks
from simulation import DispatchSimulation
from text_view import render_dispatch_center


DEFAULT_TASK = "easy"
DEFAULT_SEED = 42


class DispatchPulseEnvironment(MCPEnvironment):  # type: ignore[misc]
    """Emergency-dispatch OpenEnv environment exposed as MCP tools."""

    def __init__(self) -> None:
        if _OPENENV_IMPORT_ERROR is not None:
            raise RuntimeError(
                "openenv-core is required to run the DispatchPulse server. "
                f"Original import error: {_OPENENV_IMPORT_ERROR}"
            )
        if _FASTMCP_IMPORT_ERROR is not None:
            raise RuntimeError(
                "fastmcp is required to run the DispatchPulse server. "
                f"Original import error: {_FASTMCP_IMPORT_ERROR}"
            )

        # Internal mutable state set by reset()
        self.sim: Optional[DispatchSimulation] = None
        self.task_name: str = DEFAULT_TASK
        self.seed: int = DEFAULT_SEED
        self.cumulative_step_reward: float = 0.0
        self.episode_count: int = 0

        mcp = FastMCP("dispatchpulse")

        # Capture self for tool closures
        env = self

        @mcp.tool
        def view_dispatch_center() -> str:
            """Return the current dispatch center view as text.

            This is a FREE inspection action — it does NOT advance the
            simulation clock. Use it whenever you need to re-check pending
            calls, available units, or hospital status before deciding what
            to do next.

            Returns:
                A formatted text snapshot of the dispatch center.
            """
            if env.sim is None:
                return "ERROR: environment not initialised. Call reset first."
            return render_dispatch_center(env.sim, env.task_name)

        @mcp.tool
        def dispatch(call_id: str, unit_id: str, hospital_id: str = "") -> str:
            """Dispatch an emergency unit to a pending call.

            This advances the simulation clock by 1 minute (the dispatcher's
            decision time).

            Args:
                call_id: ID of the call to send a unit to (e.g. "CALL-007").
                unit_id: ID of the unit to dispatch (e.g. "ALS-1").
                hospital_id: Optional destination hospital (e.g. "H1"). Leave
                    empty to defer the hospital choice. Choosing the hospital
                    that has the right specialty (cardiac/stroke/trauma) for
                    the call meaningfully improves patient outcome.

            Returns:
                Confirmation message followed by the new dispatch center view.
            """
            if env.sim is None:
                return "ERROR: environment not initialised. Call reset first."
            if env.sim.episode_done:
                return "ERROR: episode is already complete. Call reset to start a new one."
            chosen_hospital = hospital_id.strip() or None
            step_reward, msg = env.sim.dispatch(call_id, unit_id, chosen_hospital)
            env.cumulative_step_reward += step_reward
            env.sim.advance_time(1)
            return msg + "\n\n" + render_dispatch_center(env.sim, env.task_name)

        @mcp.tool
        def classify(call_id: str, severity: int) -> str:
            """Reclassify the severity of a pending call.

            Use this when you suspect the caller's reported severity is wrong
            (for example, after gathering more details). Severity is on a
            1-5 scale where 1 is life-threatening and 5 is a false alarm.
            Advances the simulation clock by 1 minute.

            Args:
                call_id: ID of the call to reclassify.
                severity: New severity level (1=critical, 2=urgent, 3=moderate,
                    4=low, 5=false alarm).

            Returns:
                Confirmation message followed by the new dispatch center view.
            """
            if env.sim is None:
                return "ERROR: environment not initialised. Call reset first."
            if env.sim.episode_done:
                return "ERROR: episode is already complete."
            step_reward, msg = env.sim.classify(call_id, severity)
            env.cumulative_step_reward += step_reward
            env.sim.advance_time(1)
            return msg + "\n\n" + render_dispatch_center(env.sim, env.task_name)

        @mcp.tool
        def callback(call_id: str, question: str) -> str:
            """Phone the caller back to clarify their emergency.

            Useful when the caller's description is ambiguous and you want
            ground-truth on the emergency type before committing your most
            valuable units. There's a 70% chance the caller will clarify;
            otherwise they'll be too distressed. Advances the clock by 1
            minute (you spent that minute on the phone).

            Args:
                call_id: ID of the call to phone back.
                question: The clarifying question to ask (free text).

            Returns:
                The caller's response followed by the dispatch center view.
            """
            if env.sim is None:
                return "ERROR: environment not initialised. Call reset first."
            if env.sim.episode_done:
                return "ERROR: episode is already complete."
            step_reward, msg = env.sim.callback(call_id, question)
            env.cumulative_step_reward += step_reward
            env.sim.advance_time(1)
            return msg + "\n\n" + render_dispatch_center(env.sim, env.task_name)

        @mcp.tool
        def wait(minutes: int = 1) -> str:
            """Skip ahead in the simulation by the given number of minutes.

            Use this when there are no decisions to make right now (e.g. all
            units are en route and you're waiting for one to free up). The
            cap is 5 minutes per call to keep the agent in the loop on
            incoming calls. Calling wait while critical calls are unhandled
            costs you score.

            Args:
                minutes: Number of simulation minutes to skip (1-5).

            Returns:
                The new dispatch center view after time has advanced.
            """
            if env.sim is None:
                return "ERROR: environment not initialised. Call reset first."
            if env.sim.episode_done:
                return "ERROR: episode is already complete."
            n = max(1, min(int(minutes), env.sim.config.max_wait_step_minutes))
            pending_before = len(env.sim.get_pending_calls())
            env.sim.advance_time(n)
            # Slight per-minute penalty for waiting while pending calls exist
            env.cumulative_step_reward -= 0.005 * n * pending_before
            return f"Advanced {n} minute(s).\n\n" + render_dispatch_center(
                env.sim, env.task_name
            )

        # Register MCP server with the base class
        super().__init__(mcp)
        self._state = State(episode_id=str(uuid4()), step_count=0)

        # Auto-bootstrap with the default task so single-shot HTTP /step calls
        # (which create a fresh env per request) start in a usable state.
        # WebSocket / MCP sessions can still call reset() explicitly with a
        # different task_name to override.
        self._auto_reset()

    def _auto_reset(self) -> None:
        try:
            scenario = load_scenario(DEFAULT_TASK)
            self.sim = DispatchSimulation(scenario, seed=DEFAULT_SEED)
            self.task_name = DEFAULT_TASK
            self.seed = DEFAULT_SEED
            self.cumulative_step_reward = 0.0
        except Exception:  # pragma: no cover - never crash __init__
            self.sim = None

    # ------------------------------------------------------------------
    # OpenEnv lifecycle methods
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """Reset the environment to the start of a fresh episode.

        Args:
            seed: random seed for reproducibility (default 42).
            episode_id: Optional caller-supplied episode ID.
            task_name: One of {"easy", "medium", "hard"} (default "easy").

        Returns:
            Observation with the initial dispatch center view in metadata["text"].
        """
        chosen_task = (task_name or DEFAULT_TASK).strip().lower()
        if chosen_task not in list_tasks():
            chosen_task = DEFAULT_TASK
        chosen_seed = int(seed) if seed is not None else DEFAULT_SEED

        scenario = load_scenario(chosen_task)
        self.sim = DispatchSimulation(scenario, seed=chosen_seed)
        self.task_name = chosen_task
        self.seed = chosen_seed
        self.cumulative_step_reward = 0.0
        self.episode_count += 1

        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "status": "ready",
                "task": chosen_task,
                "seed": chosen_seed,
                "tasks_available": list_tasks(),
                "text": render_dispatch_center(self.sim, self.task_name),
                "tools": [
                    "view_dispatch_center",
                    "dispatch",
                    "classify",
                    "callback",
                    "wait",
                ],
            },
        )

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "error": (
                    f"Unknown action type {type(action).__name__}. "
                    "Use ListToolsAction or CallToolAction (MCP)."
                )
            },
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Execute one step. Tools advance the sim; we then enrich the obs."""
        self._state.step_count += 1
        obs = super().step(action, timeout_s=timeout_s, **kwargs)
        return self._enrich_observation(obs)

    async def step_async(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        self._state.step_count += 1
        obs = await super().step_async(action, timeout_s=timeout_s, **kwargs)
        return self._enrich_observation(obs)

    @property
    def state(self) -> State:
        return self._state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _enrich_observation(self, obs: Observation) -> Observation:
        """Inject reward / done / sim metadata onto a base Observation."""
        if self.sim is None:
            return obs

        obs.done = bool(self.sim.episode_done)
        if self.sim.episode_done:
            final = grade_simulation(self.sim)
            obs.reward = float(final.total)
            md = obs.metadata or {}
            md.update(
                {
                    "final_reward": final.model_dump(),
                    "task": self.task_name,
                    "completed_calls": len(self.sim.completed_calls),
                    "timed_out_calls": len(self.sim.timed_out_calls),
                    "total_calls": self.sim.total_calls(),
                }
            )
            obs.metadata = md
        else:
            obs.reward = float(self.cumulative_step_reward)
            md = obs.metadata or {}
            md.update(
                {
                    "current_time": self.sim.current_time,
                    "calls_pending": len(self.sim.get_pending_calls()),
                    "units_available": len(self.sim.get_available_units()),
                    "task": self.task_name,
                }
            )
            obs.metadata = md
        return obs
