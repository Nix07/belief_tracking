import argparse
import os
import subprocess
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src import global_utils


def read_models(models_file: str) -> list[str]:
    """Read model names from the models.txt file."""
    models = []
    with open(models_file, "r") as f:
        for line in f:
            model = line.strip()
            if model and not model.startswith("#"):  # Skip empty lines and comments
                models.append(model)
    return models


def run_evaluation(
    model: str,
    visibility: bool = False,
    remote: bool = False,
    n_samples: int = 10,
    batch_size: int = 1,
    n_runs: int = 10,
    seed: int = 10,
    save_results: str = None,
) -> bool:
    """
    Run evaluation for a single model.
    
    Returns:
        bool: True if evaluation succeeded, False otherwise
    """
    script_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "evaluate_causalToM.py"
    )
    
    cmd = [
        sys.executable,
        script_path,
        "--model", model,
        "--n-samples", str(n_samples),
        "--batch-size", str(batch_size),
        "--n-runs", str(n_runs),
        "--seed", str(seed),
    ]
    
    if visibility:
        cmd.append("--visibility")
    
    if remote:
        cmd.append("--remote")
    
    if save_results:
        cmd.extend(["--save-results", save_results])
    
    print(f"\n{'='*80}")
    print(f"Running evaluation for: {model}")
    print(f"Visibility: {visibility}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Evaluation failed for {model} (visibility={visibility})")
        print(f"Return code: {e.returncode}")
        return False
    except Exception as e:
        print(f"ERROR: Unexpected error while evaluating {model} (visibility={visibility})")
        print(f"Error: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all models from models.txt in both visibility conditions"
    )
    parser.add_argument(
        "--models-file",
        type=str,
        default=os.path.join(project_root, "src", "models.txt"),
        help="Path to models.txt file",
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Run model inference remotely",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples for evaluation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=10,
        help="Number of independent evaluation runs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=10,
        help="Random seed",
    )
    parser.add_argument(
        "--save-results",
        type=str,
        default=f"{global_utils.PROJECT_ROOT}/results/model_evaluations/",
        help="Path to save results JSON files",
    )
    parser.add_argument(
        "--skip-no-visibility",
        action="store_true",
        help="Skip no_visibility evaluation",
    )
    parser.add_argument(
        "--skip-visibility",
        action="store_true",
        help="Skip visibility evaluation",
    )

    args = parser.parse_args()

    # Read models
    print(f"Reading models from: {args.models_file}")
    models = read_models(args.models_file)
    print(f"Found {len(models)} models: {models}\n")

    # Track results
    results_summary = {
        "no_visibility": {"success": [], "failed": []},
        "visibility": {"success": [], "failed": []},
    }

    # Evaluate each model
    for idx, model in enumerate(models, 1):
        print(f"\n{'#'*80}")
        print(f"Processing model {idx}/{len(models)}: {model}")
        print(f"{'#'*80}")

        # Run no_visibility evaluation
        if not args.skip_no_visibility:
            success = run_evaluation(
                model=model,
                visibility=False,
                remote=args.remote,
                n_samples=args.n_samples,
                batch_size=args.batch_size,
                n_runs=args.n_runs,
                seed=args.seed,
                save_results=args.save_results,
            )
            if success:
                results_summary["no_visibility"]["success"].append(model)
            else:
                results_summary["no_visibility"]["failed"].append(model)
        else:
            print("Skipping no_visibility evaluation (--skip-no-visibility flag set)")

        # Run visibility evaluation
        if not args.skip_visibility:
            success = run_evaluation(
                model=model,
                visibility=True,
                remote=args.remote,
                n_samples=args.n_samples,
                batch_size=args.batch_size,
                n_runs=args.n_runs,
                seed=args.seed,
                save_results=args.save_results,
            )
            if success:
                results_summary["visibility"]["success"].append(model)
            else:
                results_summary["visibility"]["failed"].append(model)
        else:
            print("Skipping visibility evaluation (--skip-visibility flag set)")

    # Print summary
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}\n")

    if not args.skip_no_visibility:
        print("No Visibility Condition:")
        print(f"  Success: {len(results_summary['no_visibility']['success'])}/{len(models)}")
        if results_summary["no_visibility"]["success"]:
            print(f"  Successful models: {results_summary['no_visibility']['success']}")
        if results_summary["no_visibility"]["failed"]:
            print(f"  Failed models: {results_summary['no_visibility']['failed']}")
        print()

    if not args.skip_visibility:
        print("Visibility Condition:")
        print(f"  Success: {len(results_summary['visibility']['success'])}/{len(models)}")
        if results_summary["visibility"]["success"]:
            print(f"  Successful models: {results_summary['visibility']['success']}")
        if results_summary["visibility"]["failed"]:
            print(f"  Failed models: {results_summary['visibility']['failed']}")
        print()

    print(f"{'='*80}")
    print("All evaluations complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

