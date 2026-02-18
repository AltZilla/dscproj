"""
Privacy-Preserving Load Forecasting & Energy Optimization
==========================================================
CLI entry point for running the pipeline or individual modules.
"""

import argparse
import sys
import os

# Ensure the project root is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description="Smart Home Energy Optimization Framework"
    )
    parser.add_argument(
        "--run-all", action="store_true",
        help="Run the full pipeline (simulate → preprocess → train → optimize)"
    )
    parser.add_argument(
        "--module", type=str, default=None,
        choices=["digital_twin", "preprocessing", "vae",
                 "federated_gru", "stackelberg", "de_optimization"],
        help="Run a specific module only"
    )
    parser.add_argument(
        "--skip-simulation", action="store_true",
        help="Skip data generation, use existing CSVs"
    )
    parser.add_argument(
        "--dashboard", action="store_true",
        help="Launch the Plotly Dash visual dashboard"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda"],
        help="PyTorch device"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.yaml"
    )

    args = parser.parse_args()

    if args.dashboard:
        print("Launching dashboard...")
        from app.dashboard import create_app
        app = create_app()
        app.run(debug=True, port=8050)
        return

    if args.run_all:
        from src.pipeline import run_pipeline
        run_pipeline(args.config, args.skip_simulation, args.device)
        return

    if args.module == "digital_twin":
        from src.digital_twin import run_digital_twin
        run_digital_twin(args.config)

    elif args.module == "preprocessing":
        from src.digital_twin import load_config
        from src.preprocessing import run_preprocessing
        config = load_config(args.config)
        run_preprocessing(config)

    elif args.module:
        print(f"Module '{args.module}' requires prior stages. Use --run-all instead.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
