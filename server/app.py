"""FastAPI application for DispatchPulse, served via openenv-core's create_app."""

from __future__ import annotations

# Support both in-repo and standalone imports.
try:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

    from .dispatchpulse_environment import DispatchPulseEnvironment
except ImportError:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from server.dispatchpulse_environment import DispatchPulseEnvironment

# create_app expects the environment class (not instance) so each WebSocket
# session gets its own environment object — this enables concurrent grading
# without cross-session state leakage.
app = create_app(
    DispatchPulseEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name="dispatchpulse",
)


def main() -> None:
    """Entry point for ``uv run --project . server`` or direct execution."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
