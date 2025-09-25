"""
Utility functions for causal mediation analysis experiments
"""

import json
import os
import random
import sys
from collections import defaultdict

import torch
from nnsight import LanguageModel
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path before importing from src
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)

from src.dataset import Dataset, Sample


def load_model(
    model_name: str, is_remote: bool = False, cache_dir: str = None
) -> LanguageModel:
    """Load the language model and tokenizer as LanguageModel object.

    Args:
        model_name: Name of the model to use
        is_remote: Whether to run model inference remotely
        cache_dir: Directory to cache model files

    Returns:
        LanguageModel object
    """

    if is_remote:
        model = LanguageModel(model_name)
    else:
        model = LanguageModel(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            dispatch=True,
            cache_dir=cache_dir,
        )
    return model


def load_entity_data(data_dir: str) -> tuple[list, list, list]:
    """Loads a list of synthetic characters, objects, and states.
    Args:
        data_dir: Directory containing the data
    Returns:
        tuple: Tuple containing (all_characters, all_objects, all_states)
    """
    all_characters = json.load(
        open(
            os.path.join(data_dir, "synthetic_entities", "characters.json"),
            "r",
        )
    )
    all_objects = json.load(
        open(
            os.path.join(data_dir, "synthetic_entities", "bottles.json"),
            "r",
        )
    )
    all_states = json.load(
        open(
            os.path.join(data_dir, "synthetic_entities", "drinks.json"),
            "r",
        )
    )

    return list(all_characters), list(all_objects), list(all_states)


def _sample_entities(
    all_characters: list,
    all_objects: list,
    all_states: list,
    num_samples: int,
) -> tuple[list, list, list]:
    """
    Randomly samples num_samples pairs of characters, objects, and states.

    Args:
        all_characters (list): List of all available characters.
        all_objects (list): List of all available objects.
        all_states (list): List of all available states.
        num_samples (int): Number of samples to generate.

    Returns:
        tuple: Tuple containing (characters_list, objects_list, states_list) where
        each *_list contains num_samples randomly selected pairs.
    """

    characters_list = []
    objects_list = []
    states_list = []

    for _ in range(num_samples):
        sample_characters = random.sample(all_characters, 2)

        sample_containers = random.sample(all_objects, 2)

        sample_states = random.sample(all_states, 2)

        characters_list.append(sample_characters)
        objects_list.append(sample_containers)
        states_list.append(sample_states)

    return characters_list, objects_list, states_list


def _generate_causalToM_samples(
    clean_configs: list,
    counterfactual_configs: list,
    dataset_class: Dataset,
) -> list:
    """
    Create CausalToM samples from given configurations.

    Args:
        clean_configs (list): List of clean configuration objects.
        counterfactual_configs (list): List of counterfactual configuration objects.
        dataset_class (class): Dataset class to use for generating samples.

    Returns:
        list: List of sample dictionaries with clean and counterfactual prompts and answers
    """
    clean_dataset = dataset_class(clean_configs)
    counterfactual_dataset = dataset_class(counterfactual_configs)

    samples = []
    num_samples = len(clean_configs)

    for idx in range(num_samples):
        # Ask the question about the first character and the first container, hence
        # set_character=0 and set_container=0.
        clean_item = clean_dataset.__getitem__(idx, set_character=0, set_container=0)
        counterfactual_item = counterfactual_dataset.__getitem__(
            idx, set_character=0, set_container=0
        )

        clean_prompt = clean_item["prompt"]
        clean_target = clean_item["target"]
        counterfactual_prompt = counterfactual_item["prompt"]
        counterfactual_target = counterfactual_item["target"]

        # Replace the question in the clean prompt with the counterfactual question
        clean_prompt = clean_prompt.replace(
            clean_item["question"], counterfactual_item["question"]
        )
        # Set clean target to "unknown" since we modified the question
        clean_target = "unknown"

        sample = {
            "clean_prompt": clean_prompt,
            "clean_ans": clean_target,
            "counterfactual_prompt": counterfactual_prompt,
            "counterfactual_ans": counterfactual_target,
            # The target depends on the specific experiment type and is set by the caller
        }

        samples.append(sample)

    return samples


def get_character_tracing_exps(
    all_characters: list,
    all_objects: list,
    all_states: list,
    num_samples: int,
) -> list:
    """
    Generate counterfactual samples for character tracing experiments, as described in Section 3 of the paper.

    Args:
        all_characters (list): List of all available characters.
        all_objects (list): List of all available objects.
        all_states (list): List of all available states.
        num_samples (int): Number of samples to generate.

    Returns:
        list: List of dictionaries containing clean and counterfactual samples.
    """
    characters_list, objects_list, states_list = _sample_entities(
        all_characters, all_objects, all_states, num_samples
    )

    clean_configs, counterfactual_configs = [], []

    for idx in range(num_samples):
        counterfactual_config = Sample(
            template_idx=2,
            characters=characters_list[idx],
            objects=objects_list[idx],
            states=states_list[idx],
        )
        counterfactual_configs.append(counterfactual_config)

        # For clean config, replace the first character with a random character
        # different from the original one. Keep the second character the same.
        random_character = random.choice(all_characters)
        while random_character in characters_list[idx]:
            random_character = random.choice(all_characters)
        modified_characters = [random_character, characters_list[idx][1]]

        clean_config = Sample(
            template_idx=2,
            characters=modified_characters,
            objects=objects_list[idx],
            states=states_list[idx],
        )
        clean_configs.append(clean_config)

    samples = _generate_causalToM_samples(
        clean_configs, counterfactual_configs, Dataset
    )

    # Set the target for each sample (first state from clean config)
    for idx, sample in enumerate(samples):
        sample["target"] = " " + clean_configs[idx].states[0]

    return samples


def get_object_tracing_exps(
    all_characters: list,
    all_objects: list,
    all_states: list,
    num_samples: int,
) -> list:
    """
    Generate counterfactual samples for object tracing experiments, as described in Section 3 of the paper.

    Args:
        all_characters (list): List of all available characters.
        all_objects (list): List of all available objects.
        all_states (list): List of all available states.
        num_samples (int): Number of samples to generate.

    Returns:
        list: List of dictionaries containing clean and counterfactual samples.
    """
    characters_list, objects_list, states_list = _sample_entities(
        all_characters, all_objects, all_states, num_samples
    )

    clean_configs, counterfactual_configs = [], []

    for idx in range(num_samples):
        counterfactual_config = Sample(
            template_idx=2,
            characters=characters_list[idx],
            objects=objects_list[idx],
            states=states_list[idx],
        )
        counterfactual_configs.append(counterfactual_config)

        # For clean config, replace the first object with a random object
        # different from the original one. Keep the second object the same.
        random_object = random.choice(all_objects)
        while random_object in objects_list[idx]:
            random_object = random.choice(all_objects)
        modified_objects = [random_object, objects_list[idx][1]]

        clean_config = Sample(
            template_idx=2,
            characters=characters_list[idx],
            objects=modified_objects,
            states=states_list[idx],
        )
        clean_configs.append(clean_config)

    # Create samples from configurations
    samples = _generate_causalToM_samples(
        clean_configs, counterfactual_configs, Dataset
    )

    # Set the target for each sample (first state from clean config)
    for idx, sample in enumerate(samples):
        sample["target"] = " " + clean_configs[idx].states[0]

    return samples


def get_state_tracing_exps(
    all_characters: list,
    all_objects: list,
    all_states: list,
    num_samples: int,
) -> list:
    """
    Generate counterfactual samples for state tracing experiments, as described in Section 3 of the paper.

    Args:
        all_characters (list): List of all available characters.
        all_objects (list): List of all available objects.
        all_states (list): List of all available states.
        num_samples (int): Number of samples to generate.

    Returns:
        list: List of dictionaries containing clean and counterfactual samples.
        all_characters (list): List of all available characters
        all_objects (dict): Dictionary mapping container types to lists of containers
        all_states (dict): Dictionary mapping state types to lists of states
        num_samples (int): Number of samples to generate

    Returns:
        list: List of dictionaries containing clean and counterfactual samples.
    """
    characters_list, objects_list, states_list = _sample_entities(
        all_characters, all_objects, all_states, num_samples
    )

    clean_configs, counterfactual_configs = [], []

    for idx in range(num_samples):
        counterfactual_config = Sample(
            template_idx=2,
            characters=characters_list[idx],
            objects=objects_list[idx],
            states=states_list[idx],
        )
        counterfactual_configs.append(counterfactual_config)

        # For clean config, replace the first state with a random state
        # different from the original one. Keep the second state the same.
        random_state = random.choice(all_states)
        while random_state in states_list[idx]:
            random_state = random.choice(all_states)
        modified_states = [random_state, states_list[idx][1]]

        clean_config = Sample(
            template_idx=2,
            characters=characters_list[idx],
            objects=objects_list[idx],
            states=modified_states,
        )
        clean_configs.append(clean_config)

    samples = _generate_causalToM_samples(
        clean_configs, counterfactual_configs, Dataset, use_corrupt_question=False
    )

    # For state tracing, the target is the first state from counterfactual config (different from other tracers)
    for idx, sample in enumerate(samples):
        sample["target"] = " " + counterfactual_configs[idx].states[0]

    return samples


def find_correct_samples(
    model: LanguageModel,
    dataset: list,
    batch_size: int = 10,
    num_samples: int = 50,
    is_remote: bool = False,
    verbose: bool = False,
) -> list:
    """Find samples where both clean and counterfactual prompts produce correct predictions.

    Args:
        model: LanguageModel object
        dataset: List of samples
        batch_size: Batch size
        num_samples: Number of samples to find

    Returns:
        list: List of indices of correct samples
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    corrects = []
    for bi, batch in enumerate(tqdm(dataloader, desc="Finding correct samples")):
        counterfactual_prompt = batch["counterfactual_prompt"]
        clean_prompt = batch["clean_prompt"]
        counterfactual_target = batch["counterfactual_ans"]
        clean_target = batch["clean_ans"]
        batch_size = len(clean_target)

        with torch.no_grad():
            with model.trace(remote=is_remote) as tracer:
                with tracer.invoke(clean_prompt):
                    clean_pred = model.lm_head.output[:, -1].argmax(dim=-1).save()

                with tracer.invoke(counterfactual_prompt):
                    counterfactual_pred = (
                        model.lm_head.output[:, -1].argmax(dim=-1).save()
                    )

        for i in range(batch_size):
            clean_pred_token = model.tokenizer.decode([clean_pred[i]]).lower().strip()
            counterfactual_pred_token = (
                model.tokenizer.decode([counterfactual_pred[i]]).lower().strip()
            )
            clean_target_token = clean_target[i].lower().strip()
            counterfactual_target_token = counterfactual_target[i].lower().strip()
            if verbose:
                print(
                    f"Clean pred token: {clean_pred_token} | Clean target token: {clean_target_token}"
                )
                print(
                    f"Counterfactual pred token: {counterfactual_pred_token} | Counterfactual target token: {counterfactual_target_token}"
                )
            index = bi * batch_size + i
            if (
                clean_pred_token == clean_target_token
                and counterfactual_pred_token == counterfactual_target_token
            ):
                corrects.append(index)

        del clean_pred, counterfactual_pred
        torch.cuda.empty_cache()

        if len(corrects) >= num_samples:
            corrects = corrects[:num_samples]
            break

    return corrects


def load_or_init_tracing_results(
    results_path: str, start_token: int = 180, start_layer: int = 0, layer_step: int = 1
) -> tuple[dict, int, int]:
    """Load existing tracing results or initialize new ones.

    Args:
        results_path: Path to save results
        start_token: Starting token index
        start_layer: Starting layer index
        layer_step: Layer step

    Returns:
        tuple: (tracing_results, start_token, start_layer)
    """
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            tracing_results = json.load(f)
        start_token = list(tracing_results.keys())[-1]
        start_layer = list(tracing_results[start_token].keys())[-1]
        return (
            defaultdict(dict, tracing_results),
            int(start_token),
            int(start_layer) + layer_step,
        )
    else:
        return defaultdict(dict), start_token, start_layer


def run_tracing_experiment(
    model: LanguageModel,
    dataloader: DataLoader,
    start_token: int,
    start_layer: int,
    layer_step: int,
    results_path: str,
    is_remote: bool = False,
    verbose: bool = False,
) -> dict:
    """Run the tracing experiment, by patching the residual vector corresponding
    to each token and layer one by one from counterfactual to clean run.

    Args:
        model: LanguageModel object
        dataloader: DataLoader object
        start_token: Starting token index
        start_layer: Starting layer index
        layer_step: Layer step
        results_path: Path to save results
        is_remote: Whether to run model inference remotely
        verbose: Whether to print verbose output

    Returns:
        dict: Tracing results
    """
    tracing_results, start_token, start_layer = load_or_init_tracing_results(
        results_path, start_token, start_layer, layer_step
    )
    STORY_START_TOKEN = 128

    for t in tqdm(range(start_token, STORY_START_TOKEN, -1)):
        # Reset start layer for each new token
        if t < start_token:
            start_layer = 0

        for layer_idx in range(start_layer, model.config.num_hidden_layers, layer_step):
            correct, total = 0, 0

            for batch in dataloader:
                counterfactual_prompt = batch["counterfactual_prompt"]
                clean_prompt = batch["clean_prompt"]
                target = batch["target"]
                batch_size = len(target)

                with torch.no_grad():
                    with model.trace(remote=is_remote) as tracer:
                        barrier = tracer.barrier(2)

                        with tracer.invoke(counterfactual_prompt):
                            counterfactual_layer_out = (
                                model.model.layers[layer_idx].output[:, t].clone()
                            )
                            barrier()

                        with tracer.invoke(clean_prompt):
                            barrier()
                            model.model.layers[layer_idx].output[:, t] = (
                                counterfactual_layer_out
                            )
                            pred = model.lm_head.output[:, -1].argmax(dim=-1).save()

                for i in range(batch_size):
                    pred_token = model.tokenizer.decode([pred[i]]).lower().strip()
                    target_token = target[i].lower().strip()
                    if verbose:
                        print(
                            f"Pred token: {pred_token} | Target token: {target_token}"
                        )
                    if pred_token == target_token:
                        correct += 1
                    total += 1

                del pred
                torch.cuda.empty_cache()

            acc = round(correct / total, 2)
            tracing_results[t][layer_idx] = acc
            print(f"Token: {t} | Layer: {layer_idx} | Accuracy: {acc}")

            # Save results periodically
            with open(results_path, "w") as f:
                json.dump(tracing_results, f, indent=4)

    return tracing_results
