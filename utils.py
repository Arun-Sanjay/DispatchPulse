"""Helper functions for distance, ETA, lookup tables, caller text generation."""

from __future__ import annotations

import math
from typing import List

import numpy as np

from models import (
    EmergencyCall,
    EmergencyType,
    EmergencyUnit,
    Position,
    Severity,
    UnitType,
)

# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------


def calculate_distance(a: Position, b: Position) -> float:
    """Euclidean distance in km."""
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def calculate_eta(unit: EmergencyUnit, destination: Position) -> float:
    """ETA in minutes assuming constant speed."""
    distance_km = calculate_distance(unit.position, destination)
    return (distance_km / unit.speed_kmh) * 60.0


# ----------------------------------------------------------------------------
# Capability lookups
# ----------------------------------------------------------------------------

CAPABLE_UNITS = {
    EmergencyType.CARDIAC_ARREST: [UnitType.ALS_AMBULANCE, UnitType.BLS_AMBULANCE],
    EmergencyType.TRAUMA: [UnitType.ALS_AMBULANCE, UnitType.BLS_AMBULANCE],
    EmergencyType.STROKE: [UnitType.ALS_AMBULANCE, UnitType.BLS_AMBULANCE],
    EmergencyType.FIRE: [UnitType.FIRE_ENGINE],
    EmergencyType.BREATHING: [UnitType.ALS_AMBULANCE, UnitType.BLS_AMBULANCE],
    EmergencyType.MINOR_INJURY: [UnitType.BLS_AMBULANCE, UnitType.ALS_AMBULANCE],
    EmergencyType.MENTAL_HEALTH: [UnitType.POLICE, UnitType.ALS_AMBULANCE],
}

OPTIMAL_UNIT = {
    EmergencyType.CARDIAC_ARREST: UnitType.ALS_AMBULANCE,
    EmergencyType.TRAUMA: UnitType.ALS_AMBULANCE,
    EmergencyType.STROKE: UnitType.ALS_AMBULANCE,
    EmergencyType.FIRE: UnitType.FIRE_ENGINE,
    EmergencyType.BREATHING: UnitType.ALS_AMBULANCE,
    EmergencyType.MINOR_INJURY: UnitType.BLS_AMBULANCE,
    EmergencyType.MENTAL_HEALTH: UnitType.POLICE,
}


def get_capable_units(emergency_type: EmergencyType) -> List[UnitType]:
    return CAPABLE_UNITS[emergency_type]


def get_optimal_unit(emergency_type: EmergencyType) -> UnitType:
    return OPTIMAL_UNIT[emergency_type]


# ----------------------------------------------------------------------------
# Caller description templates
# ----------------------------------------------------------------------------

CALLER_TEMPLATES = {
    EmergencyType.CARDIAC_ARREST: [
        "Man collapsed on the sidewalk, he's not breathing! Please hurry!",
        "My husband just grabbed his chest and fell down, he's turning blue!",
        "Someone collapsed at the bus stop, no pulse, people are trying CPR!",
        "Elderly woman found unconscious in her apartment, not responding!",
    ],
    EmergencyType.TRAUMA: [
        "Car accident on the main road, driver is bleeding from the head!",
        "Construction worker fell from scaffolding, conscious but can't move his legs!",
        "Two-wheeler hit a pedestrian, person on the road with a broken leg!",
        "Fight broke out, one person stabbed, lots of blood!",
    ],
    EmergencyType.STROKE: [
        "My mother's face is drooping on one side and she can't speak properly!",
        "Colleague suddenly can't move his right arm, speech is slurred!",
        "Old man at the temple suddenly confused, can't walk straight!",
    ],
    EmergencyType.FIRE: [
        "Kitchen fire in our apartment, smoke everywhere, we're on the 3rd floor!",
        "Warehouse on fire near the industrial area, flames visible from outside!",
        "Electrical fire in the office building, fire alarm going off!",
        "Grass fire spreading toward houses near the park!",
    ],
    EmergencyType.MINOR_INJURY: [
        "I slipped on the stairs and twisted my ankle, it's swollen.",
        "Small cut on my hand from a knife, bleeding a bit but not too bad.",
        "Child fell off bicycle, scraped knee and elbow, crying but alert.",
        "Bumped my head on a low beam, small bump, feeling a bit dizzy.",
    ],
    EmergencyType.BREATHING: [
        "My asthma is really bad, can't catch my breath, inhaler isn't working!",
        "Allergic reaction, my throat is swelling up, hard to breathe!",
        "Elderly patient with COPD, oxygen levels dropping, very short of breath!",
    ],
    EmergencyType.MENTAL_HEALTH: [
        "My neighbor is threatening to hurt himself, he's very distressed!",
        "Person on the building ledge, seems very agitated, won't come down!",
        "Family member having a severe panic attack, can't calm down, hyperventilating!",
    ],
}

MISREPORT_TEMPLATES = {
    (EmergencyType.MENTAL_HEALTH, EmergencyType.CARDIAC_ARREST): [
        "Someone is clutching their chest and can't breathe! I think it's a heart attack!",
    ],
    (EmergencyType.MINOR_INJURY, EmergencyType.TRAUMA): [
        "Bad accident! Person is hurt, there's some blood!",
    ],
    (EmergencyType.BREATHING, EmergencyType.STROKE): [
        "My father can't talk properly and is struggling, please come fast!",
    ],
    (EmergencyType.CARDIAC_ARREST, EmergencyType.MENTAL_HEALTH): [
        "He's holding his chest and crying, very upset, hyperventilating!",
    ],
    (EmergencyType.TRAUMA, EmergencyType.MINOR_INJURY): [
        "Someone fell off a ladder, looks like a small bump but they're complaining a lot.",
    ],
}


def generate_caller_text(
    true_type: EmergencyType,
    reported_type: EmergencyType,
    rng: np.random.RandomState,
) -> str:
    """Generate a caller description string from templates."""
    if true_type != reported_type:
        key = (true_type, reported_type)
        if key in MISREPORT_TEMPLATES:
            return rng.choice(MISREPORT_TEMPLATES[key])
    return rng.choice(CALLER_TEMPLATES[reported_type])


# ----------------------------------------------------------------------------
# Severity helpers
# ----------------------------------------------------------------------------


def severity_from_int(s: int) -> Severity:
    return Severity(int(s))
