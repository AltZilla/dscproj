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
                   population: int = 80) -> dict:
    """
    Fast DE optimization for dashboard interactive use.

    Uses an aggressive cost-focused fitness that freely shifts
    flexible appliances to cheap hours while respecting occupancy
    and constraint requirements.
    """
    from src.constraints import get_constraint_function

    num_appliances = len(APPLIANCE_NAMES)
    rated_powers = np.array([RATED_POWERS[n] for n in APPLIANCE_NAMES])

    # Expand hourly → 15-min slots (repeat each hour 4×)
    user_schedule = np.repeat(user_schedule_hourly, 4, axis=1)  # (7, 96)
    prices = np.repeat(prices_hourly, 4)  # (96,)
    occupancy = np.repeat(occupancy_hourly, 4)  # (96,)

    num_slots = 96

    # Build constraint function with occupancy
    config = {
        "constraints": {
            "max_daily_energy_kwh": 120.0,
            "max_hourly_power_kw": 20.0,
        }
    }
    constraints_fn = get_constraint_function(config, occupancy)

    # Normalize price range for fitness scaling
    price_range = float(prices.max() - prices.min()) + 1e-8

    # Custom fitness — cost-focused with strict usage conservation
    def fitness_fn(schedule):
        num_app, num_sl = schedule.shape

        # 1) Energy cost (PRIMARY)
        power_per_slot = schedule * rated_powers[:, np.newaxis]
        total_per_slot = power_per_slot.sum(axis=0)
        energy_cost = np.sum(total_per_slot * prices * (15 / 60))

        # 2) Usage conservation — HARD: each appliance must keep same or less ON-time
        #    Heavy penalty for adding slots, light penalty for reducing
        usage_penalty = 0.0
        for i in range(num_app):
            orig_on = user_schedule[i].sum()
            new_on = schedule[i].sum()
            excess = new_on - orig_on
            if excess > 0:
                usage_penalty += excess * 50  # Heavily penalize extra on-time
            else:
                usage_penalty += abs(excess) * 0.5  # Lightly penalize reduction

        # 3) Peak load penalty
        peak_load = np.max(total_per_slot)
        avg_load = np.mean(total_per_slot) + 1e-8
        peak_ratio = peak_load / avg_load
        peak_penalty = max(0, peak_ratio - 1.5) * 20

        # 4) Constraint violations
        constraint_penalty = constraints_fn(schedule, rated_powers)

        return energy_cost + usage_penalty + 0.1 * peak_penalty + 5 * constraint_penalty

    # Run DE with higher mutation for exploration
    de = DifferentialEvolution(
        num_appliances, num_slots,
        population_size=population,
        mutation_factor=0.9,
        crossover_rate=0.95,
        max_generations=generations
    )

    best_schedule, history = de.optimize(fitness_fn, seed=user_schedule, verbose=False)

    # Collapse to hourly using majority vote (>=2 of 4 sub-slots ON → hour ON)
    optimized_hourly = np.zeros((num_appliances, 24))
    for h in range(24):
        optimized_hourly[:, h] = (best_schedule[:, h*4:(h+1)*4].mean(axis=1) >= 0.5).astype(float)

    # Compute costs from HOURLY schedules so metrics match what the heatmap shows
    unoptimized_cost = np.sum(user_schedule_hourly * rated_powers[:, np.newaxis] * prices_hourly * 1.0)
    optimized_cost = np.sum(optimized_hourly * rated_powers[:, np.newaxis] * prices_hourly * 1.0)
    savings_pct = (1 - optimized_cost / (unoptimized_cost + 1e-8)) * 100

    unopt_peak = np.max((user_schedule_hourly * rated_powers[:, np.newaxis]).sum(axis=0))
    opt_peak = np.max((optimized_hourly * rated_powers[:, np.newaxis]).sum(axis=0))
    peak_reduction_pct = (1 - opt_peak / (unopt_peak + 1e-8)) * 100

    return {
        "optimized_hourly": optimized_hourly,
        "optimized_96": best_schedule,
        "user_96": user_schedule,
        "prices_96": prices,
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
