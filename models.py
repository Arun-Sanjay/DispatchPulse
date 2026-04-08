"""DispatchPulse data models.

Two layers:

1. **OpenEnv interface models** — ``DispatchPulseAction``, ``DispatchPulseObservation``,
   ``DispatchPulseState``. These inherit directly from openenv-core base classes
   and form the wire format the server/client/grader exchange.

2. **Internal simulation models** — ``Position``, ``EmergencyType``, ``Severity``,
   ``UnitType``, ``UnitStatus``, ``EmergencyCall``, ``EmergencyUnit``, ``Hospital``,
   ``WorldConfig``, ``Reward``. These are plain Pydantic models the simulation
   engine uses internally; they never cross the OpenEnv boundary directly.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# OpenEnv base classes
# ---------------------------------------------------------------------------
from openenv.core.env_server.types import Action, Observation, State


# ===========================================================================
# OpenEnv-facing wire types
# ===========================================================================


class DispatchPulseAction(Action):
    """A single dispatcher action.

    The agent supplies ``action_type`` plus optional fields. The simplest
    possible interface for an LLM is the ``text`` field — the server will
    parse it as a command line like ``"dispatch CALL-001 ALS-1 H1"``.

    Supported action types:
        - ``dispatch``  : send a unit to a call (call_id, unit_id, hospital_id?)
        - ``classify``  : reclassify a call's severity (call_id, severity)
        - ``callback``  : phone the caller back (call_id, message)
        - ``wait``      : skip ahead in simulation time (minutes)
        - ``view``      : free inspection (no time cost)
    """

    action_type: str = Field(
        ..., description="One of: dispatch, classify, callback, wait, view"
    )
    text: str = Field(
        default="",
        description="Free-text representation of the action (e.g. 'dispatch CALL-001 ALS-1 H1')",
    )
    call_id: Optional[str] = Field(default=None)
    unit_id: Optional[str] = Field(default=None)
    hospital_id: Optional[str] = Field(default=None)
    severity: Optional[int] = Field(default=None, ge=1, le=5)
    message: Optional[str] = Field(default=None)
    minutes: Optional[int] = Field(default=None, ge=1, le=5)


class DispatchPulseObservation(Observation):
    """What the dispatcher sees each turn.

    The ``text`` field is the human-readable dispatch center view that the
    LLM agent reads. The structured fields underneath are useful for
    programmatic agents and grading.
    """

    text: str = Field(default="", description="Formatted dispatch center view for the agent")
    current_time: int = Field(default=0, description="Simulation minute")
    time_limit: int = Field(default=30, description="Episode time limit (minutes)")
    calls_pending: int = Field(default=0, description="Number of calls waiting for dispatch")
    units_available: int = Field(default=0, description="Number of free units")
    calls_completed: int = Field(default=0)
    calls_timed_out: int = Field(default=0)
    total_calls: int = Field(default=0)
    last_action_error: Optional[str] = Field(
        default=None, description="Error message from the last action, or None"
    )
    info_message: Optional[str] = Field(
        default=None, description="Free-text message describing what just happened"
    )


class DispatchPulseState(State):
    """Internal state snapshot exposed via ``GET /state``."""

    current_time: int = Field(default=0)
    episode_done: bool = Field(default=False)
    total_calls: int = Field(default=0)
    calls_dispatched: int = Field(default=0)
    calls_completed: int = Field(default=0)
    calls_timed_out: int = Field(default=0)
    calls_pending: int = Field(default=0)
    units_available: int = Field(default=0)
    running_reward: float = Field(default=0.0)
    task_name: str = Field(default="easy")


# ===========================================================================
# Internal simulation models (plain Pydantic, never cross OpenEnv boundary)
# ===========================================================================


class Position(BaseModel):
    """A 2D coordinate on the city grid (km)."""

    x: float = Field(..., ge=0.0)
    y: float = Field(..., ge=0.0)


class EmergencyType(str, Enum):
    CARDIAC_ARREST = "cardiac_arrest"
    TRAUMA = "trauma"
    STROKE = "stroke"
    FIRE = "fire"
    MINOR_INJURY = "minor_injury"
    BREATHING = "breathing_difficulty"
    MENTAL_HEALTH = "mental_health_crisis"


class Severity(int, Enum):
    CRITICAL = 1
    URGENT = 2
    MODERATE = 3
    LOW = 4
    FALSE_ALARM = 5


class UnitType(str, Enum):
    ALS_AMBULANCE = "als_ambulance"
    BLS_AMBULANCE = "bls_ambulance"
    FIRE_ENGINE = "fire_engine"
    POLICE = "police"


class UnitStatus(str, Enum):
    AVAILABLE = "available"
    EN_ROUTE = "en_route"
    ON_SCENE = "on_scene"
    RETURNING = "returning"


class EmergencyCall(BaseModel):
    call_id: str
    timestamp: int
    caller_description: str
    location: Position
    true_type: EmergencyType
    true_severity: Severity
    reported_type: Optional[EmergencyType] = None
    reported_severity: Optional[Severity] = None
    requires_unit_types: List[UnitType] = Field(default_factory=list)
    optimal_unit_type: UnitType
    active: bool = True
    dispatched_unit_id: Optional[str] = None
    response_time: Optional[float] = None
    outcome_score: Optional[float] = None
    delivered_hospital_id: Optional[str] = None


class EmergencyUnit(BaseModel):
    unit_id: str
    unit_type: UnitType
    position: Position
    base_position: Position
    status: UnitStatus = UnitStatus.AVAILABLE
    speed_kmh: float = Field(..., gt=0)
    assigned_call_id: Optional[str] = None
    assigned_hospital_id: Optional[str] = None
    busy_until: Optional[int] = None
    capabilities: List[EmergencyType] = Field(default_factory=list)


class Hospital(BaseModel):
    hospital_id: str
    name: str
    position: Position
    capacity: int = Field(..., ge=0)
    available_beds: int = Field(..., ge=0)
    has_trauma_center: bool = False
    has_cardiac_unit: bool = False
    has_stroke_unit: bool = False
    on_diversion: bool = False


class WorldConfig(BaseModel):
    grid_size_km: float = 10.0
    time_limit_minutes: int = 30
    step_duration_minutes: int = 1
    call_timeout_minutes: int = 20
    max_wait_step_minutes: int = 5


class Reward(BaseModel):
    """Final episode reward, all components in [0.0, 1.0]."""

    total: float = Field(..., ge=0.0, le=1.0)
    survival_score: float
    efficiency_score: float
    triage_accuracy: float
    penalty: float
    details: str = ""
