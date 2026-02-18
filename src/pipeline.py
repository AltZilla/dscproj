"""
Pipeline Orchestrator
======================
Runs the end-to-end pipeline: Digital Twin → Preprocess → VAE → 
Federated GRU → Stackelberg Game → Differential Evolution.
"""

import time
import torch
import numpy as np
from pathlib import Path
from typing import Optional

from src.digital_twin import load_config, run_digital_twin, DigitalTwinSimulator
from src.preprocessing import run_preprocessing
from src.vae import run_vae
from src.federated_gru import run_federated_gru
from src.stackelberg_game import run_stackelberg
from src.differential_evolution import run_differential_evolution
from src.constraints import get_constraint_function


def run_pipeline(config_path: Optional[str] = None,
                 skip_simulation: bool = False,
                 device: str = "cpu") -> dict:
    """
    Run the full pipeline.

    Args:
        config_path: Path to config.yaml (None = auto-detect)
        skip_simulation: If True, skip data generation and load existing CSVs
        device: PyTorch device ("cpu" or "cuda")

    Returns:
        Dictionary with results from all modules
    """
    start_time = time.time()
    config = load_config(config_path)
    results = {}

    # ---------------------------------------------------------------
    # Stage 1: Digital Twin Simulation
    # ---------------------------------------------------------------
    print("=" * 60)
    print("STAGE 1: Digital Twin Simulation")
    print("=" * 60)

    if not skip_simulation:
        df = run_digital_twin(config_path)
    else:
        from src.preprocessing import load_raw_data
        print("Loading existing simulation data...")
        df = load_raw_data()

    results["simulation_df"] = df

    # ---------------------------------------------------------------
    # Stage 2: Preprocessing
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 2: Data Preprocessing")
    print("=" * 60)

    preprocessed = run_preprocessing(config)
    results["preprocessed"] = preprocessed

    # ---------------------------------------------------------------
    # Stage 3: VAE (Representation Learning)
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 3: VAE - Latent Feature Extraction")
    print("=" * 60)

    vae_results = run_vae(config, preprocessed, device)
    results["vae"] = vae_results

    # ---------------------------------------------------------------
    # Stage 4: Federated GRU (Load Forecasting)
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 4: Federated GRU - Load Forecasting")
    print("=" * 60)

    gru_results = run_federated_gru(config, preprocessed, vae_results, device)
    results["gru"] = gru_results

    # ---------------------------------------------------------------
    # Stage 5: Stackelberg Game (Economic Decisions)
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 5: Stackelberg Game - Economic Equilibrium")
    print("=" * 60)

    num_homes = config["digital_twin"]["num_homes"]
    stackelberg_results = run_stackelberg(config, gru_results, num_homes)
    results["stackelberg"] = stackelberg_results

    # ---------------------------------------------------------------
    # Stage 6: Differential Evolution (Optimization)
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 6: Differential Evolution - Schedule Optimization")
    print("=" * 60)

    constraints_fn = get_constraint_function(config)
    de_results = run_differential_evolution(config, stackelberg_results, constraints_fn)
    results["de"] = de_results

    # Save DE schedules for dashboard visualization
    schedule_dir = Path(__file__).resolve().parent.parent / "outputs" / "results"
    schedule_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        schedule_dir / "de_schedules.npz",
        optimized=de_results["best_schedule"],
        unoptimized=de_results["preferred_schedule"],
        prices=de_results["prices"],
        rated_powers=de_results["rated_powers"],
        appliance_names=de_results["appliance_names"],
        savings_pct=de_results["savings_pct"],
        peak_reduction_pct=de_results["peak_reduction_pct"],
        unoptimized_cost=de_results["unoptimized_cost"],
        optimized_cost=de_results["optimized_cost"],
    )
    print(f"Schedules saved to {schedule_dir / 'de_schedules.npz'}")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Total time:     {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"VAE final loss: {vae_results['history']['train_loss'][-1]:.4f}")
    print(f"GRU MAE:        {gru_results['metrics']['mae']:.4f}")
    print(f"GRU RMSE:       {gru_results['metrics']['rmse']:.4f}")
    print(f"Game Eq Price:  ₹{stackelberg_results['equilibrium_price']:.2f}/kWh")
    print(f"Cost Savings:   {de_results['savings_pct']:.1f}%")
    print(f"Peak Reduction: {de_results['peak_reduction_pct']:.1f}%")

    # Save summary report
    report_dir = Path(__file__).resolve().parent / "outputs" / "results"
    report_dir.mkdir(parents=True, exist_ok=True)

    with open(report_dir / "pipeline_report.txt", "w", encoding="utf-8") as f:
        f.write("Privacy-Preserving Load Forecasting & Energy Optimization\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Simulation: {num_homes} homes, {config['digital_twin']['simulation_days']} days\n")
        f.write(f"Total records: {len(df):,}\n\n")
        f.write(f"VAE:\n")
        f.write(f"  Final train loss: {vae_results['history']['train_loss'][-1]:.4f}\n")
        f.write(f"  Final val loss:   {vae_results['history']['val_loss'][-1]:.4f}\n\n")
        f.write(f"Federated GRU:\n")
        f.write(f"  MAE:  {gru_results['metrics']['mae']:.4f}\n")
        f.write(f"  RMSE: {gru_results['metrics']['rmse']:.4f}\n")
        f.write(f"  MAPE: {gru_results['metrics']['mape']:.2f}%\n\n")
        f.write(f"Stackelberg Game:\n")
        f.write(f"  Equilibrium Price: ₹{stackelberg_results['equilibrium_price']:.2f}/kWh\n")
        f.write(f"  Total Demand:      {stackelberg_results['total_demand']:.2f} kW\n\n")
        f.write(f"Differential Evolution:\n")
        f.write(f"  Cost Savings:      {de_results['savings_pct']:.1f}%\n")
        f.write(f"  Peak Reduction:    {de_results['peak_reduction_pct']:.1f}%\n")
        f.write(f"  Unoptimized Cost:  ₹{de_results['unoptimized_cost']:.2f}\n")
        f.write(f"  Optimized Cost:    ₹{de_results['optimized_cost']:.2f}\n\n")
        f.write(f"Total Pipeline Time: {elapsed:.1f}s\n")

    print(f"\nReport saved to {report_dir / 'pipeline_report.txt'}")

    results["config"] = config
    return results
