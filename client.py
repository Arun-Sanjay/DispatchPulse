"""DispatchPulse client.

Subclasses ``openenv.core.env_client.EnvClient`` and implements the three
required hooks: ``_step_payload``, ``_parse_result``, ``_parse_state``.

The client speaks WebSocket to the env server (a FastAPI app created via
``create_fastapi_app``). Use ``DispatchPulseEnv.from_docker_image(image)``
to spin up a local container, or ``DispatchPulseEnv(base_url=...)`` to
connect to an already-running server (e.g. a Hugging Face Space).
"""

from __future__ import annotations

from typing import Any, Dict

from openenv.core.env_client import EnvClient, StepResult

from models import DispatchPulseAction, DispatchPulseObservation, DispatchPulseState


class DispatchPulseEnv(
    EnvClient[DispatchPulseAction, DispatchPulseObservation, DispatchPulseState]
):
    """Async client for the DispatchPulse OpenEnv environment.

    Example (Docker image)::

        env = await DispatchPulseEnv.from_docker_image("dispatchpulse:latest")
        result = await env.reset(task_name="easy", seed=42)
        while not result.done:
            action = DispatchPulseAction(action_type="wait", minutes=1)
            result = await env.step(action)
        await env.close()

    Example (remote URL)::

        async with DispatchPulseEnv(base_url="https://...hf.space") as env:
            result = await env.reset(task_name="easy", seed=42)
    """

    def _step_payload(self, action: DispatchPulseAction) -> Dict[str, Any]:
        return action.model_dump(mode="json")

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[DispatchPulseObservation]:
        obs_data = payload.get("observation", payload) or {}
        # Drop unknown keys defensively (model_config is extra=forbid)
        allowed = set(DispatchPulseObservation.model_fields.keys())
        obs_clean = {k: v for k, v in obs_data.items() if k in allowed}
        observation = DispatchPulseObservation(**obs_clean)
        return StepResult(
            observation=observation,
            reward=payload.get("reward", observation.reward),
            done=payload.get("done", observation.done),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> DispatchPulseState:
        allowed = set(DispatchPulseState.model_fields.keys())
        state_clean = {k: v for k, v in payload.items() if k in allowed}
        return DispatchPulseState(**state_clean)
