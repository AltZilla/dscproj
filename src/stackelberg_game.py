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
