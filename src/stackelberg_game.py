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
        forecasted_loads: Max consumption capacity per user

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
    """Run the Stackelberg game using forecasted loads and appliance ratings.

    Max consumption per home is derived from actual appliance rated powers
    (physical capacity) rather than GRU predictions, which are too small to
    create meaningful Stackelberg dynamics.  GRU predictions are used as
    heterogeneity weights so homes with higher forecasted demand get
    proportionally higher max-consumption caps.
    """
    print("Running Stackelberg Game...")

    # Derive realistic max consumption from appliance ratings
    twin_cfg = config.get("digital_twin", {})
    appliances = twin_cfg.get("appliances", {})
    total_rated_kw = sum(a["rated_power_kw"] for a in appliances.values())

    # Apply a realistic capacity factor (25-30% of total rated power).
    # A typical home doesn't run all appliances at full power simultaneously.
    capacity_factor = 0.25
    base_max_consumption = total_rated_kw * capacity_factor  # ~4.9 kW

    # Use GRU predictions to create per-home heterogeneity
    preds = gru_results["test_predictions"]
    loads_per_home = np.array_split(preds, num_homes)
    avg_loads = np.array([chunk.mean() if len(chunk) > 0 else 1.0
                          for chunk in loads_per_home])
    # Normalize to [0.7, 1.3] range for variation
    load_min, load_max = avg_loads.min(), avg_loads.max()
    if load_max - load_min > 1e-6:
        weights = 0.7 + 0.6 * (avg_loads - load_min) / (load_max - load_min)
    else:
        weights = np.ones(num_homes)

    max_consumptions = base_max_consumption * weights
    print(f"  Max consumption per home: {max_consumptions.mean():.2f} kW "
          f"(range: {max_consumptions.min():.2f}-{max_consumptions.max():.2f})")

    result = find_stackelberg_equilibrium(config, num_homes, max_consumptions)
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
    comfort_weight: float = 15.0,
    price_min: float = None,
    price_max: float = None,
    num_iterations: int = 50,
) -> dict:
    """
    Run a Stackelberg equilibrium for each of 24 hours independently.

    Pricing follows correct scarcity economics:
      - No demand  → SG = TOU (no game needed)
      - Low demand → SG ≤ TOU (leader lowers price to encourage consumption)
      - High demand near capacity → SG > TOU (scarcity signal to shift load)

    Args:
        tou_prices_hourly: TOU prices per hour (24,)
        user_schedule_hourly: User schedule (num_appliances × 24), binary
        rated_powers: Rated power per appliance (num_appliances,)
        forecasted_loads_hourly: Optional GRU forecasted total load (24,).
        num_users: Number of follower users in the game
        supply_capacity_kw: Grid capacity per hour (kW)
        comfort_weight: Follower comfort weight
        price_min: Min price to search (defaults to min TOU * 0.5)
        price_max: Global max price (defaults to max TOU * 1.5)
        num_iterations: Iterations for equilibrium search per hour

    Returns:
        Dictionary with hourly_prices (24,) and per-hour game details
    """
    global_price_min = price_min if price_min is not None else max(0.5, float(tou_prices_hourly.min()) * 0.5)
    global_price_max = price_max if price_max is not None else float(tou_prices_hourly.max()) * 1.5

    # Compute hourly demand profile (kW per hour)
    if forecasted_loads_hourly is not None:
        power_per_hour = np.asarray(forecasted_loads_hourly, dtype=float).flatten()
        if power_per_hour.shape[0] != 24:
            raise ValueError("forecasted_loads_hourly must have length 24")
        power_per_hour = np.maximum(power_per_hour, 0.0)
    else:
        power_per_hour = (user_schedule_hourly * rated_powers[:, np.newaxis]).sum(axis=0)

    hourly_prices = np.zeros(24)
    hourly_demands = np.zeros(24)
    hourly_profits = np.zeros(24)

    np.random.seed(42)
    comfort_weights = np.full(num_users, comfort_weight) * np.random.uniform(
        0.8, 1.2, num_users
    )

    # Fixed per-user max consumption (physical appliance capability).
    # This does NOT change by hour — only the number of active users does.
    per_user_cap = supply_capacity_kw / num_users * 2.5
    np.random.seed(43)  # separate seed for max_consumptions
    base_max_consumptions = np.full(num_users, per_user_cap) * np.random.uniform(
        0.7, 1.3, num_users
    )

    for hour in range(24):
        tou_price = float(tou_prices_hourly[hour])

        # Skip game for hours with negligible demand — use TOU price directly
        total_demand_kw = power_per_hour[hour]
        if total_demand_kw < 0.5:
            hourly_prices[hour] = tou_price
            hourly_demands[hour] = 0.0
            hourly_profits[hour] = 0.0
            continue

        # Price range scales with congestion (demand / capacity ratio).
        # Low demand → search up to TOU (leader may lower price).
        # High demand near/above capacity → search above TOU (scarcity).
        congestion = total_demand_kw / max(supply_capacity_kw, 0.1)
        # At congestion=0: max = TOU.  At congestion=1: max = TOU*1.5
        hour_price_max = min(tou_price * (1.0 + 0.5 * congestion), global_price_max)
        hour_price_min = global_price_min

        # Scale number of ACTIVE USERS by demand level.
        # More demand at this hour = more users competing for the same
        # supply capacity → higher equilibrium price.  This is the correct
        # economic model: scarcity comes from more participants, not from
        # changing individual capabilities.
        demand_ratio = total_demand_kw / max(power_per_hour.max(), 0.1)
        effective_users = max(int(num_users * demand_ratio), 2)
        hour_comfort = comfort_weights[:effective_users]
        hour_max_c = base_max_consumptions[:effective_users]

        # Grid search for optimal leader price at this hour
        best_price = hour_price_min
        best_profit = -np.inf

        for iteration in range(num_iterations):
            if iteration < num_iterations // 2:
                prices = np.linspace(hour_price_min, hour_price_max, 30)
            else:
                delta = (hour_price_max - hour_price_min) / (iteration + 1)
                prices = np.linspace(
                    max(hour_price_min, best_price - delta),
                    min(hour_price_max, best_price + delta),
                    30,
                )

            for p in prices:
                profit = leader_objective(
                    p, effective_users, hour_comfort,
                    hour_max_c, supply_capacity_kw
                )
                if profit > best_profit:
                    best_profit = profit
                    best_price = p

        hourly_prices[hour] = best_price
        eq_demand, _ = compute_total_demand(
            best_price, effective_users, hour_comfort, hour_max_c
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

