import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Callable

import fire
from nnsight import CONFIG
from torch.utils.data import DataLoader
from utils import (
    find_correct_samples,
    get_character_tracing_exps,
    get_object_tracing_exps,
    get_state_tracing_exps,
    load_entity_data,
    load_model,
    run_tracing_experiment,
)

# Add project root to path before importing from src
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)
from src import global_utils

# Get credentials from environment variables
nnsight_api_key = global_utils.load_env_var("NDIF_KEY")
hf_token = global_utils.load_env_var("HF_WRITE")

# Set credentials
CONFIG.set_default_api_key(nnsight_api_key)
os.environ["HF_TOKEN"] = hf_token
logger = global_utils.logger


@dataclass
class Tracer:
    """Class to run Causal Mediation Analysis experiments for tracing the information flow."""

    entity_type: str
    model_name: str
    data_dir: str
    results_dir: str
    num_samples: int
    batch_size: int
    tracing_batch_size: int
    start_layer: int
    start_token: int
    layer_step: int
    is_remote: bool
    verbose: bool

    def _get_dataset_generator(self) -> Callable:
        """Selects the appropriate dataset generator method based on entity type.

        Returns:
            Callable: The appropriate dataset generator method.
        """

        if self.entity_type == "character":
            return get_character_tracing_exps
        elif self.entity_type == "object":
            return get_object_tracing_exps
        elif self.entity_type == "state":
            return get_state_tracing_exps

    def run(self) -> None:
        """Run the Causal Mediation Analysis experiment, as described in Section 3 of the paper."""

        results_path = os.path.join(self.results_dir, f"{self.entity_type}.json")
        logger.info(f"Running {self.entity_type} tracing experiment.")

        logger.info(f"Loading model {self.model_name}...")
        model = load_model(self.model_name, is_remote=self.is_remote)
        logger.info("Model loaded.")

        logger.info("Generating intervention experiment samples...")
        dataset_generator = self._get_dataset_generator()
        all_characters, all_objects, all_states = load_entity_data(self.data_dir)

        # Generate twice the required number of samples to ensure we have enough correct samples
        dataset = dataset_generator(
            all_characters,
            all_objects,
            all_states,
            self.num_samples * 2,
        )
        logger.info("Samples generated.")

        logger.info("Finding correct samples...")
        corrects = find_correct_samples(
            model,
            dataset,
            batch_size=self.batch_size,
            num_samples=self.num_samples,
            is_remote=self.is_remote,
            verbose=self.verbose,
        )
        assert len(corrects) == self.num_samples, (
            f"Expected {self.num_samples} correct samples, but found {len(corrects)}"
        )
        logger.info(f"Found {len(corrects)} correct samples.")

        filtered_dataset = [dataset[i] for i in corrects]
        filtered_dataloader = DataLoader(
            filtered_dataset,
            batch_size=self.tracing_batch_size,
            shuffle=False,
        )

        logger.info("Running tracing experiment...")
        tracing_results = run_tracing_experiment(
            model,
            filtered_dataloader,
            self.start_token,
            self.start_layer,
            self.layer_step,
            results_path,
            is_remote=self.is_remote,
            verbose=self.verbose,
        )
        logger.info("Tracing experiment completed.")

        logger.info("Saving final tracing results.")
        with open(results_path, "w") as f:
            json.dump(tracing_results, f, indent=4)
        logger.info(f"Results saved to {results_path}")


def main(
    entity_type: str,
    model_name: str,
    results_dir: str = f"{global_utils.PROJECT_ROOT}/results",
    data_dir: str = "data",
    num_samples: int = 50,
    batch_size: int = 10,
    tracing_batch_size: int = 25,
    start_token: int = 180,
    start_layer: int = 0,
    layer_step: int = 1,
    is_remote: bool = False,
    verbose: bool = False,
    random_seed: int = 10,
):
    """Run Causal Mediation Analysis experiments
    Args:
        entity_type: Type of entity to trace (character, object, or state)
        model_name: Name of the model to use
        results_dir: Directory to save results

    Returns:
        None
    """

    random.seed(random_seed)
    CONFIG.APP.REMOTE_LOGGING = verbose

    # Convert relative paths to absolute paths
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(global_utils.PROJECT_ROOT, data_dir)
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(global_utils.PROJECT_ROOT, results_dir)

    # Print the parameters
    logger.info(f"Entity type: {entity_type}")
    logger.info(f"Model name: {model_name}")
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Number of samples: {num_samples}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Tracing batch size: {tracing_batch_size}")
    logger.info(f"Start token: {start_token}")
    logger.info(f"Start layer: {start_layer}")
    logger.info(f"Layer step: {layer_step}")
    logger.info(f"Is remote: {is_remote}")
    logger.info(f"Verbose: {verbose}")
    logger.info(f"Random seed: {random_seed}")

    assert entity_type in ["character", "object", "state"], (
        f"Expected entity type to be one of 'character', 'object', or 'state', but got {entity_type}"
    )

    # Check if data_dir directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} does not exist")

    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    tracer = Tracer(
        entity_type=entity_type,
        model_name=model_name,
        data_dir=data_dir,
        results_dir=results_dir,
        num_samples=num_samples,
        batch_size=batch_size,
        tracing_batch_size=tracing_batch_size,
        start_token=start_token,
        start_layer=start_layer,
        layer_step=layer_step,
        is_remote=is_remote,
        verbose=verbose,
    )

    tracer.run()


if __name__ == "__main__":
    fire.Fire(main)
