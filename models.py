"""DispatchPulse data models. All Pydantic v2."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class Position(BaseModel):
    """A 2D coordinate on the city grid (km)."""

    x: float = Field(..., ge=0.0, description="X coordinate in km")
    y: float = Field(..., ge=0.0, description="Y coordinate in km")


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
    timestamp: int = Field(..., description="Minute of simulation when call arrived")
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


class EnvironmentSnapshot(BaseModel):
    """Lightweight environment state snapshot for state() / debugging."""

    current_time: int
    episode_done: bool
    total_calls: int
    calls_dispatched: int
    calls_completed: int
    calls_timed_out: int
    calls_pending: int
    units_available: int
    step_count: int
    running_reward: float
    task_name: Optional[str] = None
