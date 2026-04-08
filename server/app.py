"""FastAPI application for DispatchPulse.

Uses ``create_fastapi_app(env_factory, ActionCls, ObservationCls)`` from
openenv-core's HTTP server, which exposes the standard ``/reset``, ``/step``,
``/state``, ``/health``, ``/metadata``, ``/schema``, and ``/ws`` routes.
"""

from __future__ import annotations

# Support both in-repo and standalone imports.
try:
    from openenv.core.env_server.http_server import create_fastapi_app

    from .environment import DispatchPulseEnvironment
except ImportError:  # pragma: no cover
    from openenv.core.env_server.http_server import create_fastapi_app
    from server.environment import DispatchPulseEnvironment

# Import the typed Action / Observation classes from the project root models.py
import os
import sys

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from models import DispatchPulseAction, DispatchPulseObservation  # noqa: E402

# Pass the class (factory) so each session gets its own env instance.
app = create_fastapi_app(
    DispatchPulseEnvironment,
    DispatchPulseAction,
    DispatchPulseObservation,
    max_concurrent_envs=8,
)


def main() -> None:
    """Entry point for ``uv run server`` or direct execution."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
