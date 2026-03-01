"""
Partial pipeline re-run: only Stackelberg + DE.
Keeps existing VAE and GRU results from the cache.
Then regenerates all 12 graphs as PDF + PNG.
"""
import sys, os, json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(r"c:\Users\lsuni\OneDrive\Desktop\Github Projects\dscproj")
sys.path.insert(0, str(PROJECT_ROOT))

CACHE_FILE = PROJECT_ROOT / "results" / "pipeline_cache.json"

# Load existing cache (keeps VAE + GRU intact)
print("Loading existing cache (VAE + GRU preserved)...")
with open(CACHE_FILE) as f:
    data = json.load(f)

print(f"  VAE epochs:    {len(data['vae_history']['train_loss'])}")
print(f"  GRU rounds:    {len(data['gru_history']['round_loss'])}")
print(f"  GRU MAE:       {data['gru_metrics']['mae']:.4f}")
print(f"  OLD SG price:  ₹{data['sg_eq_price']:.2f}")
print()

# Re-run only Stackelberg + DE with fixed code
from src.digital_twin import load_config
from src.stackelberg_game import run_stackelberg, run_hourly_stackelberg
from src.differential_evolution import run_differential_evolution
from src.constraints import get_constraint_function

config = load_config()
num_homes = config["digital_twin"]["num_homes"]

# Build a fake gru_results dict from cached data (enough for run_stackelberg)
gru_results = {
    "test_predictions": np.array(data["gru_predictions"]),
    "test_targets": np.array(data["gru_targets"]),
    "history": data["gru_history"],
    "metrics": data["gru_metrics"],
}

# ── Stage 5: Re-run Stackelberg ──
print("=" * 60)
print("RE-RUNNING STACKELBERG GAME (with fixed code)")
print("=" * 60)
sg_results = run_stackelberg(config, gru_results, num_homes)
data["sg_history"] = {
    k: (v.tolist() if isinstance(v, np.ndarray) else v)
    for k, v in sg_results["history"].items()
}
data["sg_eq_price"] = float(sg_results["equilibrium_price"])
data["sg_total_demand"] = float(sg_results["total_demand"])
print(f"  NEW SG price: ₹{data['sg_eq_price']:.2f}")

# Hourly Stackelberg
twin_cfg = config["digital_twin"]
pricing_cfg = twin_cfg["pricing"]
tou_hourly = np.zeros(24)
for period_name in ("off_peak", "shoulder", "peak"):
    period = pricing_cfg[period_name]
    for h in period["hours"]:
        tou_hourly[h] = period["rate"]

appliance_names = list(twin_cfg["appliances"].keys())
rated_powers = np.array([twin_cfg["appliances"][n]["rated_power_kw"] for n in appliance_names])
representative_schedule = np.ones((len(appliance_names), 24)) * 0.3

hourly_sg = run_hourly_stackelberg(
    tou_prices_hourly=tou_hourly,
    user_schedule_hourly=representative_schedule,
    rated_powers=rated_powers,
    num_users=num_homes,
    supply_capacity_kw=config["stackelberg"]["supply_capacity_kw"],
)
data["hourly_sg_prices"] = hourly_sg["hourly_prices"].tolist()
data["hourly_sg_demands"] = hourly_sg["hourly_demands"].tolist()
data["tou_hourly"] = tou_hourly.tolist()

# ── Stage 6: Re-run DE ──
print("\n" + "=" * 60)
print("RE-RUNNING DIFFERENTIAL EVOLUTION (with updated SG prices)")
print("=" * 60)
constraints_fn = get_constraint_function(config)
de_results = run_differential_evolution(config, sg_results, constraints_fn)
data["de_history"] = {
    k: (v.tolist() if isinstance(v, np.ndarray) else v)
    for k, v in de_results["history"].items()
}
data["de_savings_pct"] = float(de_results["savings_pct"])
data["de_peak_reduction_pct"] = float(de_results["peak_reduction_pct"])
data["de_unoptimized_cost"] = float(de_results["unoptimized_cost"])
data["de_optimized_cost"] = float(de_results["optimized_cost"])
data["de_best_schedule"] = de_results["best_schedule"].tolist()
data["de_preferred_schedule"] = de_results["preferred_schedule"].tolist()
data["de_prices"] = de_results["prices"].tolist()
data["de_rated_powers"] = de_results["rated_powers"].tolist()
data["de_appliance_names"] = de_results["appliance_names"]

print(f"\n  NEW Cost Savings: {data['de_savings_pct']:.1f}%")
print(f"  NEW Peak Reduction: {data['de_peak_reduction_pct']:.1f}%")
print(f"  Unoptimized: ₹{data['de_unoptimized_cost']:.2f}")
print(f"  Optimized:   ₹{data['de_optimized_cost']:.2f}")

# Save updated cache
with open(CACHE_FILE, "w") as f:
    json.dump(data, f, indent=2)
print(f"\nCache updated: {CACHE_FILE}")

print("\n" + "=" * 60)
print("DONE — Now run: python results/generate_graphs.py")
print("=" * 60)
