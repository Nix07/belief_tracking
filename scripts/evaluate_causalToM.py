import argparse
import json
import os
import random
import sys

import torch
from nnsight import CONFIG, LanguageModel
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src import global_utils
from src.dataset import Dataset, Sample


# Set up environment
CONFIG.APP.REMOTE_LOGGING = False
CONFIG.set_default_api_key(global_utils.load_env_var("NDIF_KEY"))
os.environ["HF_TOKEN"] = global_utils.load_env_var("HF_WRITE")


def load_entities():
    """Load characters, objects, and states from data files."""
    all_characters = json.load(
        open(
            os.path.join(global_utils.DATA_DIR, "synthetic_entities", "characters.json"),
            "r",
        )
    )
    all_objects = json.load(
        open(
            os.path.join(global_utils.DATA_DIR, "synthetic_entities", "bottles.json"),
            "r",
        )
    )
    all_states = json.load(
        open(
            os.path.join(global_utils.DATA_DIR, "synthetic_entities", "drinks.json"),
            "r",
        )
    )
    return all_characters, all_objects, all_states


def evaluation(
    model: LanguageModel, 
    all_characters: list, 
    all_objects: list, 
    all_states: list, 
    n_samples: int = 10, 
    batch_size: int = 1, 
    is_remote: bool = False,
    visibility: bool = False
    ) -> float:
    """
    Basic accuracy evaluation on a simple dataset.
    
    Args:
        model: LanguageModel object to evaluate
        all_characters: List of available characters
        all_objects: List of available objects
        all_states: List of available states
        n_samples: Number of samples to evaluate
        batch_size: Batch size for evaluation
        is_remote: Whether to run model inference remotely
        visibility: Whether to use visibility samples

    Returns:
        float: Accuracy score
    """
    samples = []
    sample_groups = []  # Track which instances belong to the same logical sample
    
    for i in range(n_samples):
        characters = random.sample(all_characters, 2)
        objects = random.sample(all_objects, 2)
        states = random.sample(all_states, 2)
        
        if visibility:
            # Create 2 instances for each sample: one with template_idx 0 and one with template_idx 1
            sample_0 = Sample(0, characters, objects, states)
            sample_1 = Sample(1, characters, objects, states)
            samples.append(sample_0)
            samples.append(sample_1)
            # Track that these two instances belong to the same logical sample
            sample_groups.append([len(samples) - 2, len(samples) - 1])
        else:
            template_idx = 2
            samples.append(Sample(template_idx, characters, objects, states))
            # Each instance is its own group when not using visibility
            sample_groups.append([len(samples) - 1])

    dataset = Dataset(samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Store predictions for each instance
    all_predictions = []
    all_targets = []
    
    for bi, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Basic evaluation"):
        prompt = batch["prompt"]
        target = batch["target"]
        current_batch_size = len(prompt)

        with torch.no_grad():
            with model.trace(prompt, remote=is_remote):
                pred = model.lm_head.output[:, -1].argmax(dim=-1).save()

            for i in range(current_batch_size):
                pred_str = model.tokenizer.decode([pred[i]]).lower().strip()
                target_str = target[i].lower().strip()
                all_predictions.append(pred_str == target_str)
                all_targets.append(target_str)

            del pred
            torch.cuda.empty_cache()

    # Evaluate correctness: for visibility, a sample is correct only if both instances are correct
    correct, total = 0, 0
    for group in sample_groups:
        # Check if all instances in this group are correct
        group_correct = all(all_predictions[idx] for idx in group)
        if group_correct:
            correct += 1
        total += 1

    acc = round(correct / total, 2)
    return acc


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on causalToM dataset")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3-70B-Instruct",
        help="Model to evaluate",
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Run model inference remotely",
    )
    parser.add_argument(
        "--visibility",
        action="store_true",
        help="Run model inference remotely",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10,
        help="Number of samples for evaluation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for evaluation",
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
        help="Path to save results JSON file",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=10,
        help="Number of independent evaluation runs",
    )

    args = parser.parse_args()

    # Set random seed for reproducibility (but each run will use different random samples)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load entities
    print("Loading entities...")
    all_characters, all_objects, all_states = load_entities()
    print(f"#characters: {len(all_characters)}")
    print(f"#objects: {len(all_objects)}")
    print(f"#states: {len(all_states)}")

    # Load model
    print(f"\nLoading model: {args.model}")
    if args.remote:
        model = LanguageModel(args.model)
    else:
        model = LanguageModel(
            args.model,
            device_map="auto",
            dtype=torch.float16 if "Meta-Llama-3-70B-Instruct" in args.model else torch.float32,
            dispatch=True,
        )
    print("Model loaded successfully\n")

    # Run multiple independent evaluations
    print("=" * 50)
    print(f"Running {args.n_runs} independent evaluations... (Remote: {args.remote})")
    print("=" * 50)
    
    accuracies = []
    for run_idx in range(args.n_runs):
        print(f"\n--- Run {run_idx + 1}/{args.n_runs} ---")
        # Each evaluation creates a new dataset with random samples
        acc = evaluation(
            model=model, 
            all_characters=all_characters, 
            all_objects=all_objects, 
            all_states=all_states, 
            n_samples=args.n_samples, 
            batch_size=args.batch_size, 
            is_remote=args.remote,
            visibility=args.visibility
        )
        accuracies.append(acc)
        print(f"Run {run_idx + 1} accuracy: {acc}")

    # Compute statistics
    mean_acc = round(sum(accuracies) / len(accuracies), 3)
    std_acc = round(
        (sum((x - mean_acc) ** 2 for x in accuracies) / len(accuracies)) ** 0.5, 3
    )
    min_acc = min(accuracies)
    max_acc = max(accuracies)

    print("\n" + "=" * 50)
    print("Evaluation Summary")
    print("=" * 50)
    print(f"Number of runs: {args.n_runs}")
    print(f"Mean accuracy: {mean_acc}")
    print(f"Std accuracy: {std_acc}")
    print(f"Min accuracy: {min_acc}")
    print(f"Max accuracy: {max_acc}")
    print(f"All accuracies: {accuracies}")

    results = {
        "n_runs": args.n_runs,
        "n_samples_per_run": args.n_samples,
        "accuracies": accuracies,
        "mean": mean_acc,
        "std": std_acc,
        "min": min_acc,
        "max": max_acc,
    }

    # Save results
    if args.save_results:
        # Include model name in the save path
        if not args.visibility:
            args.save_results = os.path.join(args.save_results, args.model.split("/")[-1] + ".json")
        else:
            args.save_results = os.path.join(args.save_results, args.model.split("/")[-1] + "_vis.json")
        os.makedirs(os.path.dirname(args.save_results) or ".", exist_ok=True)
        with open(args.save_results, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.save_results}")

    print("\n" + "=" * 50)
    print("Evaluation complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()

