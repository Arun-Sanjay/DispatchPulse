"""DispatchPulse — emergency-dispatch OpenEnv environment.

A real-world OpenEnv environment where an AI agent acts as a 911 emergency
dispatch coordinator. The agent triages incoming calls, dispatches limited
units (ALS / BLS ambulances, fire engines, police), and selects destination
hospitals. Patient outcomes are scored against real clinical survival
curves (cardiac arrest, trauma golden hour, stroke, fire, breathing,
mental health, minor injury).

Tasks: easy / medium / hard
Tools: view_dispatch_center, dispatch, classify, callback, wait
"""

from client import DispatchPulseEnv

try:
    from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction
except ImportError:  # pragma: no cover
    CallToolAction = None  # type: ignore
    ListToolsAction = None  # type: ignore

__all__ = ["DispatchPulseEnv", "CallToolAction", "ListToolsAction"]
__version__ = "1.0.0"
