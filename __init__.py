"""DispatchPulse — emergency-dispatch OpenEnv environment.

A real-world OpenEnv environment where an AI agent acts as a 911 emergency
dispatch coordinator. The agent triages incoming calls, dispatches limited
units (ALS / BLS ambulances, fire engines, police), and selects destination
hospitals. Patient outcomes are scored against real clinical survival
curves (cardiac arrest, trauma golden hour, stroke, fire, breathing,
mental health, minor injury).

Tasks: easy / medium / hard
"""

from client import DispatchPulseEnv
from models import (
    DispatchPulseAction,
    DispatchPulseObservation,
    DispatchPulseState,
)

__all__ = [
    "DispatchPulseEnv",
    "DispatchPulseAction",
    "DispatchPulseObservation",
    "DispatchPulseState",
]
__version__ = "1.0.0"
