"""FastAPI application for DispatchPulse.

Uses ``create_app(env_factory, ActionCls, ObservationCls)`` from
openenv-core's HTTP server. This wrapper:

- When ``ENABLE_WEB_INTERFACE=true`` (set in the Dockerfile): serves the
  OpenEnv Gradio web UI at ``/`` so judges visiting the Space in a browser
  see a friendly project page.
- Always registers the standard ``/reset``, ``/step``, ``/state``,
  ``/health``, ``/metadata``, ``/schema``, and ``/ws`` routes — these are
  what the hackathon grader actually hits.
"""

from __future__ import annotations

import os
import sys

# Support both in-repo and standalone imports.
try:
    from openenv.core.env_server.http_server import create_app

    from .environment import DispatchPulseEnvironment
except ImportError:  # pragma: no cover
    from openenv.core.env_server.http_server import create_app
    from server.environment import DispatchPulseEnvironment

# Import the typed Action / Observation classes from the project root models.py
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from models import DispatchPulseAction, DispatchPulseObservation  # noqa: E402

# Pass the class (factory) so each session gets its own env instance.
# ``env_name`` controls the web UI title and README lookup.
app = create_app(
    DispatchPulseEnvironment,
    DispatchPulseAction,
    DispatchPulseObservation,
    env_name="dispatchpulse",
    max_concurrent_envs=8,
)


def main() -> None:
    """Entry point for ``uv run server`` or direct execution."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
