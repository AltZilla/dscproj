"""
Constraint Management Module
==============================
Enforces system limitations including device operational bounds,
energy caps, comfort thresholds, and scheduling conflicts.
"""

import numpy as np
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Constraint Definitions
# ---------------------------------------------------------------------------
APPLIANCE_NAMES = [
    "hvac", "washer", "dryer", "dishwasher",
    "ev_charger", "lighting", "water_heater"
]

# Minimum ON-duration (consecutive slots) — if turned on, must stay on this long
MIN_RUN_SLOTS = {
    "hvac": 4,        # 1 hour
    "washer": 3,      # 45 min
    "dryer": 3,       # 45 min
    "dishwasher": 3,  # 45 min
    "ev_charger": 8,  # 2 hours
    "lighting": 1,    # 15 min
    "water_heater": 2,  # 30 min
}

# Sequence constraints: (A must finish before B can start)
SEQUENCE_CONSTRAINTS = [
    ("washer", "dryer"),
]


# ---------------------------------------------------------------------------
# Individual Constraint Checkers
# ---------------------------------------------------------------------------
def check_energy_cap(schedule: np.ndarray, rated_powers: np.ndarray,
                     max_hourly_kw: float = 15.0,
                     max_daily_kwh: float = 80.0) -> float:
    """
    Check if total energy exceeds hourly power or daily energy caps.
    Returns penalty (0 = no violation).
    """
    power_per_slot = schedule * rated_powers[:, np.newaxis]
    total_per_slot = power_per_slot.sum(axis=0)

    penalty = 0.0

    # Hourly power cap (check each slot)
    violations = np.maximum(total_per_slot - max_hourly_kw, 0)
    penalty += violations.sum()

    # Daily energy cap (total kWh for all slots)
    total_kwh = total_per_slot.sum() * (15 / 60)  # Convert to kWh
    if total_kwh > max_daily_kwh:
        penalty += (total_kwh - max_daily_kwh)

    return penalty


def check_min_run_duration(schedule: np.ndarray) -> float:
    """
    Check that appliances run for at least their minimum duration.
    Returns penalty for violations.
    """
    penalty = 0.0
    for i, name in enumerate(APPLIANCE_NAMES):
        if i >= schedule.shape[0]:
            break
        min_run = MIN_RUN_SLOTS.get(name, 1)
        row = schedule[i]

        # Find runs of consecutive ON slots
        runs = []
        current_run = 0
        for val in row:
            if val > 0.5:
                current_run += 1
            else:
                if current_run > 0:
                    runs.append(current_run)
                current_run = 0
        if current_run > 0:
            runs.append(current_run)

        # Penalize runs shorter than minimum
        for run_len in runs:
            if run_len < min_run:
                penalty += (min_run - run_len)

    return penalty


def check_sequence_constraints(schedule: np.ndarray) -> float:
    """
    Check that sequence constraints are satisfied (A finishes before B starts).
    Returns penalty for violations.
    """
    penalty = 0.0
    for a_name, b_name in SEQUENCE_CONSTRAINTS:
        if a_name in APPLIANCE_NAMES and b_name in APPLIANCE_NAMES:
            a_idx = APPLIANCE_NAMES.index(a_name)
            b_idx = APPLIANCE_NAMES.index(b_name)

            if a_idx < schedule.shape[0] and b_idx < schedule.shape[0]:
                a_row = schedule[a_idx]
                b_row = schedule[b_idx]

                # Find last ON slot for A and first ON slot for B
                a_slots = np.where(a_row > 0.5)[0]
                b_slots = np.where(b_row > 0.5)[0]

                if len(a_slots) > 0 and len(b_slots) > 0:
                    a_last = a_slots[-1]
                    b_first = b_slots[0]
                    if b_first <= a_last:
                        penalty += (a_last - b_first + 1)

    return penalty


def check_comfort_constraints(schedule: np.ndarray,
                              rated_powers: np.ndarray,
                              occupancy: np.ndarray = None) -> float:
    """
    Check minimum usage requirements for comfort-critical appliances.
    If occupancy is provided (96 slots), HVAC and lighting must be ON
    during occupied hours.
    """
    penalty = 0.0

    if occupancy is not None:
        # Occupancy-aware: HVAC and lighting must be ON when people are home
        occupied_mask = occupancy > 0  # Boolean (96,)

        # HVAC (index 0) must be on during occupied slots
        if 0 < schedule.shape[0]:
            hvac_violations = occupied_mask & (schedule[0] < 0.5)
            penalty += hvac_violations.sum() * 0.5

        # Lighting (index 5) must be on during occupied evening slots (18:00+)
        lighting_idx = APPLIANCE_NAMES.index("lighting") if "lighting" in APPLIANCE_NAMES else -1
        if lighting_idx >= 0 and lighting_idx < schedule.shape[0]:
            evening_occupied = occupied_mask.copy()
            evening_occupied[:72] = False  # Only enforce after 6 PM
            light_violations = evening_occupied & (schedule[lighting_idx] < 0.5)
            penalty += light_violations.sum() * 0.3
    else:
        # Fallback: static comfort rules
        if 0 < schedule.shape[0]:
            hvac_ratio = schedule[0].mean()
            if hvac_ratio < 0.2:
                penalty += (0.2 - hvac_ratio) * 10

        lighting_idx = APPLIANCE_NAMES.index("lighting") if "lighting" in APPLIANCE_NAMES else -1
        if lighting_idx >= 0 and lighting_idx < schedule.shape[0]:
            evening_slots = schedule[lighting_idx, 72:96]
            if evening_slots.mean() < 0.5:
                penalty += (0.5 - evening_slots.mean()) * 5

    return penalty


# ---------------------------------------------------------------------------
# Combined Constraint Function (used by DE)
# ---------------------------------------------------------------------------
def evaluate_constraints(schedule: np.ndarray, rated_powers: np.ndarray,
                         config: dict = None,
                         occupancy: np.ndarray = None) -> float:
    """
    Combined constraint evaluation. Returns total penalty score.
    Penalty = 0 means all constraints satisfied.
    """
    if config is not None:
        cfg = config.get("constraints", {})
        max_daily = cfg.get("max_daily_energy_kwh", 80.0)
        max_hourly = cfg.get("max_hourly_power_kw", 15.0)
    else:
        max_daily = 80.0
        max_hourly = 15.0

    penalty = 0.0
    penalty += check_energy_cap(schedule, rated_powers, max_hourly, max_daily)
    penalty += check_min_run_duration(schedule)
    penalty += check_sequence_constraints(schedule)
    penalty += check_comfort_constraints(schedule, rated_powers, occupancy)

    return penalty


def get_constraint_function(config: dict, occupancy: np.ndarray = None):
    """Return a constraint function bound to the given config and occupancy."""
    def fn(schedule, rated_powers):
        return evaluate_constraints(schedule, rated_powers, config, occupancy)
    return fn


def report_violations(schedule: np.ndarray, rated_powers: np.ndarray,
                      config: dict = None) -> Dict[str, float]:
    """Return a breakdown of constraint violations."""
    cfg = config.get("constraints", {}) if config else {}
    max_daily = cfg.get("max_daily_energy_kwh", 80.0)
    max_hourly = cfg.get("max_hourly_power_kw", 15.0)

    return {
        "energy_cap": check_energy_cap(schedule, rated_powers, max_hourly, max_daily),
        "min_run_duration": check_min_run_duration(schedule),
        "sequence": check_sequence_constraints(schedule),
        "comfort": check_comfort_constraints(schedule, rated_powers),
        "total": evaluate_constraints(schedule, rated_powers, config),
    }
