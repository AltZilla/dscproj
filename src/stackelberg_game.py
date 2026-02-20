"""
Hierarchical Stackelberg Game Module
======================================
Models incentive-driven leader–follower interactions for economically
rational decision-making between grid operators and smart home users.
"""

import numpy as np
from typing import Tuple, Dict, List


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------
def user_utility(consumption: np.ndarray, comfort_weight: float) -> np.ndarray:
    """
    User utility function (diminishing returns).
    U(c) = comfort_weight * sqrt(c)  (concave utility)
    """
    return comfort_weight * np.sqrt(np.maximum(consumption, 0))


def user_cost(consumption: np.ndarray, price: float) -> np.ndarray:
    """Energy cost for the user."""
    return price * consumption


def leader_revenue(price: float, total_demand: float) -> float:
    """Grid operator revenue."""
    return price * total_demand


def leader_cost(total_demand: float, supply_capacity: float) -> float:
    """
    Grid operator cost: quadratic cost for exceeding capacity.
    Incentivizes demand below capacity.
    """
    if total_demand > supply_capacity:
        excess = total_demand - supply_capacity
        return 0.5 * excess ** 2
    return 0.0


# ---------------------------------------------------------------------------
# Follower (Smart Home User) Best Response
# ---------------------------------------------------------------------------
def follower_best_response(price: float, comfort_weight: float,
                           max_consumption: float) -> float:
    """
    Solve: max_c  U(c) - p*c
    For U(c) = w * sqrt(c):
        dU/dc = w / (2*sqrt(c)) = p
        c* = (w / (2p))^2
    """
    if price <= 0:
        return max_consumption

    optimal_c = (comfort_weight / (2 * price)) ** 2
    return min(optimal_c, max_consumption)


# ---------------------------------------------------------------------------
# Leader (Grid Operator) Optimization
# ---------------------------------------------------------------------------
def compute_total_demand(price: float, num_users: int,
                         comfort_weights: np.ndarray,
                         max_consumptions: np.ndarray) -> Tuple[float, np.ndarray]:
    """Compute total demand given price and all users' best responses."""
    demands = np.array([
        follower_best_response(price, cw, mc)
        for cw, mc in zip(comfort_weights, max_consumptions)
    ])
    return demands.sum(), demands


def leader_objective(price: float, num_users: int,
                     comfort_weights: np.ndarray,
                     max_consumptions: np.ndarray,
                     supply_capacity: float) -> float:
    """
    Leader maximizes: Revenue - Cost
    = p * D(p) - penalty(D(p))
    """
    total_demand, _ = compute_total_demand(
        price, num_users, comfort_weights, max_consumptions
    )
    revenue = leader_revenue(price, total_demand)
    cost = leader_cost(total_demand, supply_capacity)
    return revenue - cost


# ---------------------------------------------------------------------------
# Stackelberg Equilibrium Solver
# ---------------------------------------------------------------------------
def find_stackelberg_equilibrium(config: dict, num_users: int,
                                 forecasted_loads: np.ndarray) -> dict:
    """
    Find the Stackelberg Equilibrium using grid search + refinement.

    Args:
        config: Stackelberg config section
        num_users: Number of smart home users
        forecasted_loads: Predicted load per user (from GRU)

    Returns:
        Dictionary with equilibrium price, demands, and convergence history
    """
    cfg = config["stackelberg"]
    comfort_weight = cfg["follower_comfort_weight"]
    supply_capacity = cfg["supply_capacity_kw"]
    price_min = cfg["price_min"]
    price_max = cfg["price_max"]
    num_iterations = cfg["num_iterations"]

    # Per-user parameters (slight variation)
    np.random.seed(42)
    comfort_weights = np.full(num_users, comfort_weight) * np.random.uniform(0.8, 1.2, num_users)
    max_consumptions = np.maximum(forecasted_loads, 0.1)

    history = {
        "prices": [],
        "total_demands": [],
        "revenues": [],
        "leader_profits": [],
    }

    # Grid search for optimal price
    best_price = price_min
    best_profit = -np.inf

    for iteration in range(num_iterations):
        # Progressively narrow search range
        if iteration < num_iterations // 2:
            # Coarse search
            prices = np.linspace(price_min, price_max, 50)
        else:
            # Fine search around current best
            delta = (price_max - price_min) / (iteration + 1)
            prices = np.linspace(
                max(price_min, best_price - delta),
                min(price_max, best_price + delta),
                50
            )

        for p in prices:
            profit = leader_objective(p, num_users, comfort_weights,
                                      max_consumptions, supply_capacity)
            if profit > best_profit:
                best_profit = profit
                best_price = p

        total_demand, individual_demands = compute_total_demand(
            best_price, num_users, comfort_weights, max_consumptions
        )

        history["prices"].append(best_price)
        history["total_demands"].append(total_demand)
        history["revenues"].append(leader_revenue(best_price, total_demand))
        history["leader_profits"].append(best_profit)

    # Final equilibrium
    eq_demand, eq_individual = compute_total_demand(
        best_price, num_users, comfort_weights, max_consumptions
    )

    # User surplus
    user_surpluses = user_utility(eq_individual, comfort_weights) - user_cost(eq_individual, best_price)

    print(f"\nStackelberg Equilibrium:")
    print(f"  Equilibrium Price:   ₹{best_price:.2f}/kWh")
    print(f"  Total Demand:        {eq_demand:.2f} kW")
    print(f"  Leader Profit:       ₹{best_profit:.2f}")
    print(f"  Avg User Surplus:    ₹{user_surpluses.mean():.2f}")
    print(f"  Supply Utilization:  {eq_demand/supply_capacity*100:.1f}%")

    return {
        "equilibrium_price": best_price,
        "total_demand": eq_demand,
        "individual_demands": eq_individual,
        "user_surpluses": user_surpluses,
        "leader_profit": best_profit,
        "comfort_weights": comfort_weights,
        "history": history,
    }


def run_stackelberg(config: dict, gru_results: dict,
                    num_homes: int) -> dict:
    """Run the Stackelberg game using forecasted loads."""
    print("Running Stackelberg Game...")

    # Use average predicted load per home as representative demand
    preds = gru_results["test_predictions"]
    # Distribute predictions across homes (rough approximation)
    loads_per_home = np.array_split(preds, num_homes)
    avg_loads = np.array([chunk.mean() if len(chunk) > 0 else 0.1
                          for chunk in loads_per_home])

    result = find_stackelberg_equilibrium(config, num_homes, avg_loads)
    return result


# ---------------------------------------------------------------------------
# Hourly Stackelberg Game (runs equilibrium per hour)
# ---------------------------------------------------------------------------
def run_hourly_stackelberg(
    tou_prices_hourly: np.ndarray,
    user_schedule_hourly: np.ndarray,
    rated_powers: np.ndarray,
    forecasted_loads_hourly: np.ndarray = None,
    num_users: int = 50,
    supply_capacity_kw: float = 15.0,
    comfort_weight: float = 2.5,
    price_min: float = None,
    price_max: float = None,
    num_iterations: int = 50,
) -> dict:
    """
    Run a Stackelberg equilibrium for each of 24 hours independently.

    Uses the user's current schedule as the demand forecast per hour.
    The leader (grid operator) sets a price for each hour; followers
    respond with optimal consumption.

    Args:
        tou_prices_hourly: TOU prices per hour (24,)
        user_schedule_hourly: User schedule (num_appliances × 24), binary
        rated_powers: Rated power per appliance (num_appliances,)
        forecasted_loads_hourly: Optional GRU forecasted total load (24,).
            When provided, this demand profile is used instead of deriving
            demand from the manual schedule.
        num_users: Number of follower users in the game
        supply_capacity_kw: Grid capacity per hour (kW)
        comfort_weight: Follower comfort weight
        price_min: Min price to search (defaults to min TOU)
        price_max: Max price to search (defaults to max TOU * 1.5)
        num_iterations: Iterations for equilibrium search per hour

    Returns:
        Dictionary with hourly_prices (24,) and per-hour game details
    """
    if price_min is None:
        price_min = max(0.5, float(tou_prices_hourly.min()) * 0.5)
    if price_max is None:
        price_max = float(tou_prices_hourly.max()) * 1.5

    # Compute hourly demand profile (kW per hour)
    if forecasted_loads_hourly is not None:
        power_per_hour = np.asarray(forecasted_loads_hourly, dtype=float).flatten()
        if power_per_hour.shape[0] != 24:
            raise ValueError("forecasted_loads_hourly must have length 24")
        power_per_hour = np.maximum(power_per_hour, 0.1)
    else:
        power_per_hour = (user_schedule_hourly * rated_powers[:, np.newaxis]).sum(axis=0)

    hourly_prices = np.zeros(24)
    hourly_demands = np.zeros(24)
    hourly_profits = np.zeros(24)

    np.random.seed(42)
    comfort_weights = np.full(num_users, comfort_weight) * np.random.uniform(
        0.8, 1.2, num_users
    )

    for hour in range(24):
        # Forecasted load per user at this hour
        total_demand_kw = power_per_hour[hour]
        per_user_load = max(total_demand_kw / max(num_users, 1), 0.1)
        max_consumptions = np.full(num_users, per_user_load) * np.random.uniform(
            0.7, 1.3, num_users
        )

        # Grid search for optimal leader price at this hour
        best_price = price_min
        best_profit = -np.inf

        for iteration in range(num_iterations):
            if iteration < num_iterations // 2:
                prices = np.linspace(price_min, price_max, 30)
            else:
                delta = (price_max - price_min) / (iteration + 1)
                prices = np.linspace(
                    max(price_min, best_price - delta),
                    min(price_max, best_price + delta),
                    30,
                )

            for p in prices:
                profit = leader_objective(
                    p, num_users, comfort_weights,
                    max_consumptions, supply_capacity_kw
                )
                if profit > best_profit:
                    best_profit = profit
                    best_price = p

        hourly_prices[hour] = best_price
        eq_demand, _ = compute_total_demand(
            best_price, num_users, comfort_weights, max_consumptions
        )
        hourly_demands[hour] = eq_demand
        hourly_profits[hour] = best_profit

    print(f"\nHourly Stackelberg Equilibrium:")
    print(f"  Price range:  ₹{hourly_prices.min():.2f} – ₹{hourly_prices.max():.2f}/kWh")
    print(f"  Avg price:    ₹{hourly_prices.mean():.2f}/kWh")
    print(f"  Grid capacity: {supply_capacity_kw:.1f} kW")

    return {
        "hourly_prices": hourly_prices,
        "hourly_demands": hourly_demands,
        "hourly_profits": hourly_profits,
        "supply_capacity_kw": supply_capacity_kw,
    }
