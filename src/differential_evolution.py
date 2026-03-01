"""
Differential Evolution (DE) Optimization Module
=================================================
Executes global optimization of appliance scheduling and energy allocation
using the DE/rand/1/bin evolutionary strategy.
"""

import numpy as np
from typing import Callable, Tuple, Optional, Dict
from pathlib import Path


# ---------------------------------------------------------------------------
# Fitness Function
# ---------------------------------------------------------------------------
def compute_fitness(schedule: np.ndarray, prices: np.ndarray,
                    rated_powers: np.ndarray, preferred_schedule: np.ndarray,
                    cost_weight: float, comfort_weight: float,
                    peak_weight: float, constraints_fn: Callable) -> float:
    """
    Evaluate a candidate schedule.

    Args:
        schedule: Binary array (num_appliances × num_slots) — 1=ON, 0=OFF
        prices: Price per slot (num_slots,)
        rated_powers: Rated power per appliance (num_appliances,)
        preferred_schedule: User's preferred schedule (same shape as schedule)
        cost_weight, comfort_weight, peak_weight: Objective weights
        constraints_fn: Returns penalty score for constraint violations

    Returns:
        Fitness value (lower is better)
    """
    num_appliances, num_slots = schedule.shape

    # Energy cost (primary objective)
    power_per_slot = schedule * rated_powers[:, np.newaxis]  # (appliances, slots)
    total_per_slot = power_per_slot.sum(axis=0)  # (slots,)
    energy_cost = np.sum(total_per_slot * prices * (15 / 60))  # kWh cost for 15-min

    # Comfort penalty — weighted by appliance power (shifting heavy appliances penalized more)
    power_weighted_diff = np.abs(schedule - preferred_schedule) * rated_powers[:, np.newaxis]
    comfort_penalty = np.sum(power_weighted_diff) * 0.5  # Scale to INR-comparable range

    # Peak load penalty (penalize concentration)
    peak_load = np.max(total_per_slot)
    avg_load = np.mean(total_per_slot) + 1e-8
    peak_ratio = peak_load / avg_load
    peak_penalty = max(0, peak_ratio - 2.0) * 50  # Moderate penalty

    # Usage conservation — total ON-time shouldn't change too much
    orig_usage = preferred_schedule.sum()
    new_usage = schedule.sum()
    usage_penalty = abs(new_usage - orig_usage) * 0.1

    # Constraint penalty (reduced from 100× to avoid dominating cost)
    constraint_penalty = constraints_fn(schedule, rated_powers)

    fitness = (cost_weight * energy_cost
               + comfort_weight * comfort_penalty
               + peak_weight * peak_penalty
               + usage_penalty
               + 10 * constraint_penalty)

    return fitness


# ---------------------------------------------------------------------------
# DE Algorithm
# ---------------------------------------------------------------------------
class DifferentialEvolution:
    """DE/rand/1/bin optimizer for appliance scheduling."""

    def __init__(self, num_appliances: int, num_slots: int,
                 population_size: int = 100,
                 mutation_factor: float = 0.8,
                 crossover_rate: float = 0.9,
                 max_generations: int = 500):
        self.num_appliances = num_appliances
        self.num_slots = num_slots
        self.pop_size = population_size
        self.F = mutation_factor
        self.CR = crossover_rate
        self.max_gen = max_generations

        # Flatten schedule to 1D for DE operations
        self.dim = num_appliances * num_slots

    def _init_population(self, seed_individual: np.ndarray = None) -> np.ndarray:
        """Initialize population. Seed with preferred schedule + random variations."""
        pop = np.random.random((self.pop_size, self.dim))

        if seed_individual is not None:
            # First 20% of population are variations of the seed
            seed_flat = seed_individual.flatten()
            n_seeded = self.pop_size // 5
            for i in range(n_seeded):
                # Flip random 10-30% of bits from the seed
                flip_rate = np.random.uniform(0.1, 0.3)
                flip_mask = np.random.random(self.dim) < flip_rate
                pop[i] = np.where(flip_mask, 1 - seed_flat, seed_flat)

        return (pop > 0.5).astype(float)

    def _to_schedule(self, individual: np.ndarray) -> np.ndarray:
        """Reshape flat vector to (num_appliances, num_slots) schedule."""
        return individual.reshape(self.num_appliances, self.num_slots)

    def optimize(self, fitness_fn: Callable, seed: np.ndarray = None,
                 verbose: bool = True) -> Tuple[np.ndarray, dict]:
        """
        Run DE optimization.

        Args:
            fitness_fn: Takes schedule (num_appliances × num_slots) → float (minimize)
            seed: Optional initial schedule to seed the population with

        Returns:
            (best_schedule, history)
        """
        pop = self._init_population(seed)

        # Evaluate initial population
        fitness = np.array([fitness_fn(self._to_schedule(ind)) for ind in pop])

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = pop[best_idx].copy()

        history = {
            "best_fitness": [best_fitness],
            "avg_fitness": [fitness.mean()],
            "worst_fitness": [fitness.max()],
        }

        for gen in range(self.max_gen):
            for i in range(self.pop_size):
                # Select 3 random distinct individuals (not i)
                candidates = list(range(self.pop_size))
                candidates.remove(i)
                r1, r2, r3 = np.random.choice(candidates, 3, replace=False)

                # Mutation: v = x_r1 + F * (x_r2 - x_r3)
                mutant = pop[r1] + self.F * (pop[r2] - pop[r3])

                # Binarize mutant (sigmoid threshold)
                mutant = 1.0 / (1.0 + np.exp(-5 * (mutant - 0.5)))
                mutant = (mutant > 0.5).astype(float)

                # Crossover: binomial
                cross_mask = np.random.random(self.dim) < self.CR
                # Ensure at least one dimension from mutant
                j_rand = np.random.randint(self.dim)
                cross_mask[j_rand] = True
                trial = np.where(cross_mask, mutant, pop[i])

                # Selection
                trial_fitness = fitness_fn(self._to_schedule(trial))
                if trial_fitness <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial.copy()

            history["best_fitness"].append(best_fitness)
            history["avg_fitness"].append(fitness.mean())
            history["worst_fitness"].append(fitness.max())

            if verbose and ((gen + 1) % 50 == 0 or gen == 0):
                print(f"  Gen {gen+1}/{self.max_gen}  "
                      f"Best: {best_fitness:.4f}  "
                      f"Avg: {fitness.mean():.4f}")

        best_schedule = self._to_schedule(best_individual)
        return best_schedule, history


# ---------------------------------------------------------------------------
# High-level runner
# ---------------------------------------------------------------------------
def run_differential_evolution(config: dict, stackelberg_results: dict,
                               constraints_fn: Callable) -> dict:
    """
    Run DE optimization using TOU prices.

    Optimizes scheduling for a single representative day (96 time slots).
    The preferred schedule concentrates usage during peak hours;
    the optimizer shifts loads to off-peak for cost savings.
    """
    cfg = config["differential_evolution"]
    twin_cfg = config["digital_twin"]
    pricing_cfg = twin_cfg["pricing"]

    # Appliance list and rated powers
    appliance_names = list(twin_cfg["appliances"].keys())
    num_appliances = len(appliance_names)
    rated_powers = np.array([
        twin_cfg["appliances"][name]["rated_power_kw"]
        for name in appliance_names
    ])

    num_slots = 96  # One day at 15-min resolution

    # Build PURE TOU price vector (no blending — preserve full price spread)
    prices = np.zeros(num_slots)
    for slot in range(num_slots):
        hour = (slot * 15) // 60
        for period_name in ("off_peak", "shoulder", "peak"):
            period = pricing_cfg[period_name]
            if hour in period["hours"]:
                prices[slot] = period["rate"]
                break

    # Preferred schedule: PEAK-HEAVY usage (represents naive/unoptimized user)
    # This gives the optimizer expensive loads to shift to cheaper hours
    preferred = np.zeros((num_appliances, num_slots))

    # Hourly probability of being ON — concentrated during expensive peak hours
    BASE_SCHEDULES = {
        "hvac":         [0.2]*6 + [0.4]*4 + [0.3]*4 + [0.4]*4 + [0.9]*4 + [0.8]*2,
        "washer":       [0.0]*6 + [0.0]*4 + [0.0]*4 + [0.0]*4 + [0.7]*4 + [0.3]*2,
        "dryer":        [0.0]*6 + [0.0]*4 + [0.0]*4 + [0.0]*4 + [0.6]*4 + [0.4]*2,
        "dishwasher":   [0.0]*6 + [0.0]*4 + [0.0]*4 + [0.0]*4 + [0.8]*4 + [0.5]*2,
        "ev_charger":   [0.0]*6 + [0.0]*4 + [0.0]*4 + [0.0]*4 + [0.9]*4 + [0.7]*2,
        "lighting":     [0.0]*6 + [0.2]*4 + [0.1]*4 + [0.2]*4 + [0.9]*4 + [0.8]*2,
        "water_heater": [0.1]*6 + [0.3]*4 + [0.1]*4 + [0.2]*4 + [0.8]*4 + [0.6]*2,
    }

    for i, name in enumerate(appliance_names):
        hourly_probs = BASE_SCHEDULES.get(name, [0.3]*24)
        # Expand to 15-min slots (4 slots per hour)
        slot_probs = np.repeat(hourly_probs, 4)
        preferred[i] = (slot_probs > 0.5).astype(float)

    # Build fitness function
    def fitness_fn(schedule):
        return compute_fitness(
            schedule, prices, rated_powers, preferred,
            cfg["cost_weight"], cfg["comfort_weight"], cfg["peak_weight"],
            constraints_fn
        )

    print(f"DE Optimization: {num_appliances} appliances x {num_slots} slots")
    print(f"  Pop: {cfg['population_size']}, F: {cfg['mutation_factor']}, "
          f"CR: {cfg['crossover_rate']}, Gens: {cfg['max_generations']}")

    # Print price spread
    print(f"  Prices: off-peak=₹{pricing_cfg['off_peak']['rate']}/kWh, "
          f"shoulder=₹{pricing_cfg['shoulder']['rate']}/kWh, "
          f"peak=₹{pricing_cfg['peak']['rate']}/kWh")

    de = DifferentialEvolution(
        num_appliances, num_slots,
        cfg["population_size"], cfg["mutation_factor"],
        cfg["crossover_rate"], cfg["max_generations"]
    )

    best_schedule, history = de.optimize(fitness_fn, seed=preferred, verbose=True)

    # Compute before/after costs
    unoptimized_cost = np.sum(preferred * rated_powers[:, np.newaxis] * prices * (15/60))
    optimized_cost = np.sum(best_schedule * rated_powers[:, np.newaxis] * prices * (15/60))
    savings_pct = (1 - optimized_cost / (unoptimized_cost + 1e-8)) * 100

    # Peak load comparison
    unopt_peak = np.max((preferred * rated_powers[:, np.newaxis]).sum(axis=0))
    opt_peak = np.max((best_schedule * rated_powers[:, np.newaxis]).sum(axis=0))
    peak_reduction_pct = (1 - opt_peak / (unopt_peak + 1e-8)) * 100

    # Per-appliance usage summary
    print(f"\nDE Results:")
    print(f"  Unoptimized cost: ₹{unoptimized_cost:.2f}")
    print(f"  Optimized cost:   ₹{optimized_cost:.2f}")
    print(f"  Cost savings:     {savings_pct:.1f}%")
    print(f"  Peak load:        {unopt_peak:.1f} -> {opt_peak:.1f} kW ({peak_reduction_pct:.1f}% reduction)")

    print(f"\n  Per-appliance ON-slots (original -> optimized):")
    for i, name in enumerate(appliance_names):
        orig_on = int(preferred[i].sum())
        opt_on = int(best_schedule[i].sum())
        print(f"    {name:15s}: {orig_on:3d} -> {opt_on:3d} slots")

    return {
        "best_schedule": best_schedule,
        "appliance_names": appliance_names,
        "rated_powers": rated_powers,
        "prices": prices,
        "preferred_schedule": preferred,
        "history": history,
        "unoptimized_cost": unoptimized_cost,
        "optimized_cost": optimized_cost,
        "savings_pct": savings_pct,
        "peak_reduction_pct": peak_reduction_pct,
    }


# ---------------------------------------------------------------------------
# Quick Optimize (for dashboard interactive use)
# ---------------------------------------------------------------------------
APPLIANCE_NAMES = [
    "hvac", "washer", "dryer", "dishwasher",
    "ev_charger", "lighting", "water_heater"
]

RATED_POWERS = {
    "hvac": 3.5, "washer": 0.5, "dryer": 2.0,
    "dishwasher": 1.8, "ev_charger": 7.0,
    "lighting": 0.3, "water_heater": 4.5,
}


def quick_optimize(user_schedule_hourly: np.ndarray,
                   prices_hourly: np.ndarray,
                   occupancy_hourly: np.ndarray,
                   generations: int = 200,
                   population: int = 80,
                   grid_capacity_kw: float = 15.0,
                   stackelberg_prices: np.ndarray = None,
                   forecast_loads_hourly: np.ndarray = None) -> dict:
    """
    Model-driven optimizer — DE explores the space, fitness function guides decisions.

    No hardcoded slot placement. The only hard constraint is ON-count conservation
    (total hours per appliance match user input). All domain knowledge is encoded
    as fitness penalties that the DE learns to minimize:
      - Energy cost (primary objective)
      - Occupancy-comfort: comfort appliances penalized when OFF during occupied hours
      - Temporal proximity: penalize shifting ON-hours far from user's original schedule
      - Consecutiveness: washer/dryer benefit from adjacent ON-hours
      - Peak load: penalize concentration of load
      - Grid capacity: penalize exceeding grid_capacity_kw at any hour

    Args:
        grid_capacity_kw: Maximum grid capacity per hour (kW). Exceeding this
            incurs a steep penalty.
        stackelberg_prices: Optional (24,) array of Stackelberg equilibrium
            prices. When provided, blended with TOU prices so game-theoretic
            signals influence scheduling.
        forecast_loads_hourly: Optional (24,) GRU-forecasted total load profile.
            When provided, optimization is softly guided toward this profile.
    """
    from src.constraints import get_constraint_function

    num_appliances = len(APPLIANCE_NAMES)
    num_slots = 24
    rated_powers = np.array([RATED_POWERS[n] for n in APPLIANCE_NAMES])

    # Blend TOU prices with Stackelberg equilibrium prices when available
    if stackelberg_prices is not None:
        # 50/50 blend: keeps TOU structure but shifts costs via game theory
        prices = 0.5 * prices_hourly + 0.5 * stackelberg_prices
    else:
        prices = prices_hourly.copy()

    occupancy = occupancy_hourly.copy()
    forecast_loads = None
    if forecast_loads_hourly is not None:
        forecast_loads = np.asarray(forecast_loads_hourly, dtype=float).flatten()
        if forecast_loads.shape[0] != num_slots:
            raise ValueError("forecast_loads_hourly must have length 24")
        forecast_loads = np.maximum(forecast_loads, 0.0)

    # Target ON-hours per appliance (hard constraint — always conserved)
    target_on = np.array([int(user_schedule_hourly[i].sum()) for i in range(num_appliances)])

    # Pre-compute: which hours each appliance was originally ON
    user_on_hours = []
    for i in range(num_appliances):
        user_on_hours.append(set(np.where(user_schedule_hourly[i] > 0.5)[0]))

    # Pre-compute: occupancy-derived data
    occupied = (occupancy > 0).astype(float)
    max_occ = max(occupancy.max(), 1)

    # Appliance categories (derived from appliance properties, not hardcoded slots)
    # Comfort appliances: high-power, occupancy-dependent usage
    COMFORT_APPS = {"hvac", "lighting", "water_heater"}
    # Sequential appliances: benefit from consecutive ON-hours
    SEQUENTIAL_APPS = {"washer", "dryer"}

    # Build constraint function (use grid capacity for max_hourly_power_kw)
    occupancy_96 = np.repeat(occupancy_hourly, 4)
    config = {"constraints": {"max_daily_energy_kwh": 120.0, "max_hourly_power_kw": grid_capacity_kw}}
    constraints_fn = get_constraint_function(config, occupancy_96)

    def _enforce_on_count(schedule):
        """Only hard constraint: preserve exact ON-hours per appliance.
        If over → turn off most expensive. If under → turn on cheapest."""
        fixed = schedule.copy()
        for i in range(num_appliances):
            target = target_on[i]
            row = fixed[i]

            # For lighting: force OFF during unoccupied hours first
            if APPLIANCE_NAMES[i] == "lighting":
                for h in range(24):
                    if occupancy[h] == 0:
                        row[h] = 0.0

            on_idx = np.where(row > 0.5)[0]
            off_idx = np.where(row <= 0.5)[0]
            current = len(on_idx)

            if current > target and len(on_idx) > 0:
                costs = prices[on_idx]
                turn_off = on_idx[np.argsort(-costs)][:current - target]
                fixed[i, turn_off] = 0.0
            elif current < target and len(off_idx) > 0:
                # For lighting: only add ON-hours during occupied times
                if APPLIANCE_NAMES[i] == "lighting":
                    occupied_off = off_idx[occupancy[off_idx] > 0]
                    if len(occupied_off) > 0:
                        costs = prices[occupied_off]
                        turn_on = occupied_off[np.argsort(costs)][:target - current]
                        fixed[i, turn_on] = 1.0
                else:
                    costs = prices[off_idx]
                    turn_on = off_idx[np.argsort(costs)][:target - current]
                    fixed[i, turn_on] = 1.0
        return fixed

    def fitness_fn(schedule):
        sched = _enforce_on_count(schedule)

        power_per_slot = sched * rated_powers[:, np.newaxis]
        total_per_slot = power_per_slot.sum(axis=0)

        # === 1. Energy cost (primary) ===
        energy_cost = float(np.sum(total_per_slot * prices))

        # === 2. Occupancy-comfort penalty ===
        # Comfort appliances should be ON when people are home
        # Weighted by occupancy count (3 people home = stronger signal)
        comfort_penalty = 0.0
        for i, app in enumerate(APPLIANCE_NAMES):
            if app not in COMFORT_APPS:
                continue
            power = rated_powers[i]
            for h in range(24):
                occ = occupancy[h]
                if occ > 0 and sched[i, h] < 0.5:
                    # Comfort appliance OFF when people home — bad
                    comfort_penalty += power * (occ / max_occ) * 3
                elif occ == 0 and sched[i, h] > 0.5:
                    # ON when nobody home — wasteful (especially lighting)
                    if app == "lighting":
                        comfort_penalty += power * 8  # lights in empty house is pointless
                    else:
                        comfort_penalty += power * 0.5  # minor waste for HVAC/WH

        # Lighting must stay ON when user scheduled it AND people are home
        # Lighting must turn OFF when nobody is home
        lighting_idx = APPLIANCE_NAMES.index("lighting")
        for h in range(24):
            if h in user_on_hours[lighting_idx] and occupancy[h] > 0 and sched[lighting_idx, h] < 0.5:
                comfort_penalty += 500
            if occupancy[h] == 0 and sched[lighting_idx, h] > 0.5:
                comfort_penalty += 500
        # === 3. Temporal proximity penalty ===
        # Penalize moving ON-hours far from user's original schedule
        # This keeps water heater near morning/evening blocks, etc.
        shift_penalty = 0.0
        for i in range(num_appliances):
            if not user_on_hours[i]:
                continue
            new_on = np.where(sched[i] > 0.5)[0]
            for h in new_on:
                # Minimum distance to any of user's original ON-hours
                min_dist = min(abs(h - uh) for uh in user_on_hours[i])
                # Quadratic penalty: distance 1 = 1, distance 3 = 9, distance 6 = 36
                # Increased multiplier so it stays closer to the user's intended block
                # EV charger is highly flexible — user just needs it charged, not at a specific time
                flex = 0.03 if APPLIANCE_NAMES[i] == "ev_charger" else 1.0
                shift_penalty += (min_dist ** 2) * rated_powers[i] * flex

        # === 4. Consecutiveness reward (negative penalty) for sequential appliances ===
        consec_penalty = 0.0
        for i, app in enumerate(APPLIANCE_NAMES):
            if app not in SEQUENTIAL_APPS:
                continue
            on_hours = sorted(np.where(sched[i] > 0.5)[0])
            if len(on_hours) <= 1:
                continue
            # Count gaps between ON-hours
            for k in range(1, len(on_hours)):
                gap = on_hours[k] - on_hours[k-1]
                if gap > 1:
                    consec_penalty += gap * rated_powers[i] * 2

        # === 4b. Thermostat pulsing penalty ===
        # Only penalise consecutive runs LONGER than the user's own longest
        # block.  If the user scheduled a solid 6-hour HVAC block, allow up
        # to 6 consecutive hours without penalty.
        thermostat_penalty = 0.0
        THERMOSTAT_APPS = {"hvac", "water_heater"}
        for i, app in enumerate(APPLIANCE_NAMES):
            if app not in THERMOSTAT_APPS:
                continue

            # Compute user's max consecutive run for this appliance
            user_on = sorted(user_on_hours[i])
            user_max_run = 1
            if len(user_on) > 1:
                run = 1
                for k in range(1, len(user_on)):
                    if user_on[k] - user_on[k-1] == 1:
                        run += 1
                        user_max_run = max(user_max_run, run)
                    else:
                        run = 1

            on_hours = sorted(np.where(sched[i] > 0.5)[0])
            if len(on_hours) <= 1:
                continue

            consecutive_count = 1
            for k in range(1, len(on_hours)):
                if on_hours[k] - on_hours[k-1] == 1:
                    consecutive_count += 1
                    if consecutive_count > user_max_run:
                        thermostat_penalty += ((consecutive_count - user_max_run) ** 2) * rated_powers[i] * 5.0
                else:
                    consecutive_count = 1

            # Penalize large gaps within user blocks (fragmentation).
            # 1hr gap = pulsing (OK, saves money during peak).
            # 3+hr gap = house drifts too far from setpoint (BAD).
            # Find user's consecutive blocks for this appliance
            user_blocks = []
            if user_on:
                bs, be = user_on[0], user_on[0]
                for uh in user_on[1:]:
                    if uh - be == 1:
                        be = uh
                    else:
                        user_blocks.append((bs, be))
                        bs, be = uh, uh
                user_blocks.append((bs, be))

            # For each user block, check optimizer's gaps in that range
            for bs, be in user_blocks:
                # Allow ±2hr for preheating/precooling
                block_on = [h for h in on_hours if bs - 2 <= h <= be + 2]
                if len(block_on) > 1:
                    for k in range(1, len(block_on)):
                        gap = block_on[k] - block_on[k - 1]
                        if gap >= 3:  # 3+hr gap within a block
                            thermostat_penalty += (gap - 1) ** 2 * rated_powers[i] * 3.0

        # === 5. Peak load penalty ===
        peak_load = np.max(total_per_slot)
        avg_load = np.mean(total_per_slot) + 1e-8
        peak_penalty = max(0, peak_load / avg_load - 1.5) * 30

        # === 6. Grid capacity penalty ===
        # Steep quadratic penalty for exceeding grid capacity at any hour
        capacity_penalty = 0.0
        for h in range(24):
            if total_per_slot[h] > grid_capacity_kw:
                excess = total_per_slot[h] - grid_capacity_kw
                capacity_penalty += (excess ** 2) * 50  # Very steep

        # === 7. Constraint check ===
        sched_96 = np.repeat(sched, 4, axis=1)
        constraint_penalty = constraints_fn(sched_96, rated_powers)

        # === 8. Forecast alignment penalty (optional) ===
        forecast_penalty = 0.0
        if forecast_loads is not None:
            # Keep optimized demand reasonably close to GRU-predicted demand.
            forecast_penalty = np.mean(np.abs(total_per_slot - forecast_loads)) * 2.0

        return (energy_cost
                + comfort_penalty
                + shift_penalty
                + consec_penalty
                + thermostat_penalty
                + peak_penalty
                + capacity_penalty
                + 5 * constraint_penalty
                + forecast_penalty)

    # Run DE — the model explores and the fitness guides
    de = DifferentialEvolution(
        num_appliances, num_slots,
        population_size=population,
        mutation_factor=0.9,
        crossover_rate=0.95,
        max_generations=generations
    )

    best_raw, history = de.optimize(fitness_fn, seed=user_schedule_hourly, verbose=False)
    optimized_hourly = _enforce_on_count(best_raw)

    # Compute costs
    unoptimized_cost = np.sum(user_schedule_hourly * rated_powers[:, np.newaxis] * prices_hourly)
    optimized_cost = np.sum(optimized_hourly * rated_powers[:, np.newaxis] * prices_hourly)
    savings_pct = (1 - optimized_cost / (unoptimized_cost + 1e-8)) * 100

    unopt_peak = np.max((user_schedule_hourly * rated_powers[:, np.newaxis]).sum(axis=0))
    opt_peak = np.max((optimized_hourly * rated_powers[:, np.newaxis]).sum(axis=0))
    peak_reduction_pct = (1 - opt_peak / (unopt_peak + 1e-8)) * 100

    return {
        "optimized_hourly": optimized_hourly,
        "rated_powers": rated_powers,
        "appliance_names": APPLIANCE_NAMES,
        "unoptimized_cost": float(unoptimized_cost),
        "optimized_cost": float(optimized_cost),
        "savings_pct": float(savings_pct),
        "unopt_peak": float(unopt_peak),
        "opt_peak": float(opt_peak),
        "peak_reduction_pct": float(peak_reduction_pct),
        "history": history,
    }

