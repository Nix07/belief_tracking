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

# Add the project root directory to Python path to allow importing from src
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)

from src.dataset import Dataset, Sample

random.seed(10)


class CMAExperimentDataset(Dataset):
    """Base class for causal mediation analysis datasets"""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _create_base_configurations(
    all_characters,
    all_containers,
    all_states,
    num_samples,
):
    """
    Create base configurations for tracing experiments.

    This helper function generates the common base configurations needed for all types
    of tracing experiments. It selects characters, containers, and states based on the
    template requirements.

    Args:
        story_templates (dict): Dictionary containing story templates
        all_characters (list): List of all available characters
        all_containers (dict): Dictionary mapping container types to lists of containers
        all_states (dict): Dictionary mapping state types to lists of states
        num_samples (int): Number of samples to generate
        template_idx (int, optional): Index of the template to use. Defaults to 2.

    Returns:
        tuple: Tuple containing (template, characters_list, containers_list, states_list)
               where each *_list contains num_samples randomly selected pairs
    """

    characters_list = []
    containers_list = []
    states_list = []

    # Generate configurations for each sample
    for _ in range(num_samples):
        sample_characters = random.sample(all_characters, 2)

        sample_containers = random.sample(all_containers, 2)

        sample_states = random.sample(all_states, 2)

        characters_list.append(sample_characters)
        containers_list.append(sample_containers)
        states_list.append(sample_states)

    return characters_list, containers_list, states_list


def _create_samples_from_configs(
    clean_configs, corrupt_configs, dataset_class, use_corrupt_question=True
):
    """
    Create samples from clean and corrupt configurations.

    This helper function generates samples for tracing experiments by combining
    clean and corrupt configurations according to the experiment's requirements.

    Args:
        clean_configs (list): List of clean configuration objects
        corrupt_configs (list): List of corrupt configuration objects
        dataset_class (class): Dataset class to use for generating samples
        use_corrupt_question (bool, optional): If True, use the corrupt question in the
                                              clean prompt. Defaults to True.

    Returns:
        list: List of sample dictionaries with clean and corrupt prompts and answers
    """
    clean_dataset = dataset_class(clean_configs)
    corrupt_dataset = dataset_class(corrupt_configs)

    samples = []
    num_samples = len(clean_configs)

    for idx in range(num_samples):
        # Get items from datasets with specific settings
        clean_item = clean_dataset.__getitem__(idx, set_character=0, set_container=0)
        corrupt_item = corrupt_dataset.__getitem__(
            idx, set_character=0, set_container=0
        )

        clean_prompt = clean_item["prompt"]
        clean_target = clean_item["target"]
        corrupt_prompt = corrupt_item["prompt"]
        corrupt_target = corrupt_item["target"]

        # If using corrupt question, replace the question in the clean prompt
        if use_corrupt_question:
            clean_prompt = clean_prompt.replace(
                clean_item["question"], corrupt_item["question"]
            )
            # Set clean target to "unknown" since we modified the question
            clean_target = "unknown"

        # Create sample with clean and corrupt prompts and targets
        sample = {
            "clean_prompt": clean_prompt,
            "clean_ans": clean_target,
            "corrupt_prompt": corrupt_prompt,
            "corrupt_ans": corrupt_target,
            # The target depends on the specific experiment type and is set by the caller
        }

        samples.append(sample)

    return samples


def get_character_tracing_exps(
    story_templates, all_characters, all_containers, all_states, num_samples
):
    """
    Generate samples for character tracing experiments.

    This function creates samples where the character entity is modified between
    clean and corrupt configurations. The experiment tests whether the model can
    trace information about characters through its layers.

    Args:
        story_templates (dict): Dictionary containing story templates
        all_characters (list): List of all available characters
        all_containers (dict): Dictionary mapping container types to lists of containers
        all_states (dict): Dictionary mapping state types to lists of states
        num_samples (int): Number of samples to generate

    Returns:
        list: List of sample dictionaries for character tracing experiments
    """
    characters_list, containers_list, states_list = _create_base_configurations(
        all_characters, all_containers, all_states, num_samples
    )

    clean_configs, corrupt_configs = [], []

    for idx in range(num_samples):
        # For corrupt config, use the original characters
        corrupt_config = Sample(
            template_idx=2,
            characters=characters_list[idx],
            containers=containers_list[idx],
            states=states_list[idx],
        )
        corrupt_configs.append(corrupt_config)

        # For clean config, replace the first character with a random different one
        random_character = random.choice(all_characters)
        while random_character in characters_list[idx]:
            random_character = random.choice(all_characters)

        modified_characters = [random_character, characters_list[idx][1]]

        clean_config = Sample(
            template_idx=2,
            characters=modified_characters,
            containers=containers_list[idx],
            states=states_list[idx],
        )
        clean_configs.append(clean_config)

    samples = _create_samples_from_configs(
        clean_configs, corrupt_configs, Dataset, use_corrupt_question=True
    )

    # Set the target for each sample (first state from clean config)
    for idx, sample in enumerate(samples):
        sample["target"] = " " + clean_configs[idx].states[0]

    return samples


def get_object_tracing_exps(
    story_templates, all_characters, all_containers, all_states, num_samples
):
    """
    Generate samples for object (container) tracing experiments.

    This function creates samples where the container entity is modified between
    clean and corrupt configurations. The experiment tests whether the model can
    trace information about containers through its layers.

    Args:
        story_templates (dict): Dictionary containing story templates
        all_characters (list): List of all available characters
        all_containers (dict): Dictionary mapping container types to lists of containers
        all_states (dict): Dictionary mapping state types to lists of states
        num_samples (int): Number of samples to generate

    Returns:
        list: List of sample dictionaries for object tracing experiments
    """
    characters_list, containers_list, states_list = _create_base_configurations(
        all_characters, all_containers, all_states, num_samples
    )

    clean_configs, corrupt_configs = [], []

    for idx in range(num_samples):
        # For corrupt config, use the original containers
        corrupt_config = Sample(
            template_idx=2,
            characters=characters_list[idx],
            containers=containers_list[idx],
            states=states_list[idx],
        )
        corrupt_configs.append(corrupt_config)

        # For clean config, replace the first container with a random different one
        random_container = random.choice(all_containers)
        while random_container in containers_list[idx]:
            random_container = random.choice(all_containers)

        modified_containers = containers_list[idx].copy()
        modified_containers[0] = random_container

        clean_config = Sample(
            template_idx=2,
            characters=characters_list[idx],
            containers=modified_containers,
            states=states_list[idx],
        )
        clean_configs.append(clean_config)

    # Create samples from configurations
    samples = _create_samples_from_configs(
        clean_configs, corrupt_configs, Dataset, use_corrupt_question=True
    )

    # Set the target for each sample (first state from clean config)
    for idx, sample in enumerate(samples):
        sample["target"] = " " + clean_configs[idx].states[0]

    return samples


def get_state_tracing_exps(
    story_templates, all_characters, all_containers, all_states, num_samples
):
    """
    Generate samples for state tracing experiments.

    This function creates samples where the state entity is modified between
    clean and corrupt configurations. The experiment tests whether the model can
    trace information about states through its layers.

    Args:
        story_templates (dict): Dictionary containing story templates
        all_characters (list): List of all available characters
        all_containers (dict): Dictionary mapping container types to lists of containers
        all_states (dict): Dictionary mapping state types to lists of states
        num_samples (int): Number of samples to generate

    Returns:
        list: List of sample dictionaries for state tracing experiments
    """
    characters_list, containers_list, states_list = _create_base_configurations(
        all_characters, all_containers, all_states, num_samples
    )

    clean_configs, corrupt_configs = [], []

    for idx in range(num_samples):
        # For corrupt config, use the original states
        corrupt_config = Sample(
            template_idx=2,
            characters=characters_list[idx],
            containers=containers_list[idx],
            states=states_list[idx],
        )
        corrupt_configs.append(corrupt_config)

        # For clean config, replace the first state with a random different one
        random_state = random.choice(all_states)
        while random_state in states_list[idx]:
            random_state = random.choice(all_states)

        modified_states = states_list[idx].copy()
        modified_states[0] = random_state

        clean_config = Sample(
            template_idx=2,
            characters=characters_list[idx],
            containers=containers_list[idx],
            states=modified_states,
        )
        clean_configs.append(clean_config)

    samples = _create_samples_from_configs(
        clean_configs, corrupt_configs, Dataset, use_corrupt_question=False
    )

    # For state tracing, the target is the first state from corrupt config (different from other tracers)
    for idx, sample in enumerate(samples):
        sample["target"] = " " + corrupt_configs[idx].states[0]

    return samples


def get_memorization_exps(
    model, story_templates, all_characters, all_containers, all_states, num_samples
):
    """
    Generate samples for memorization experiments.
    """
    characters_list, containers_list, states_list = _create_base_configurations(
        all_characters, all_containers, all_states, num_samples
    )

    samples = []

    while len(samples) < num_samples:
        idx = len(samples)
        config = Sample(
            template_idx=2,
            characters=characters_list[idx],
            containers=containers_list[idx],
            states=states_list[idx],
        )

        clean_item = Dataset([config]).__getitem__(0)
        clean_prompt = clean_item["prompt"]
        clean_target = clean_item["target"]
        clean_prompt_len = len(model.tokenizer.encode(clean_prompt))

        with torch.no_grad():
            with model.trace(clean_prompt):
                pred = model.lm_head.output[0, -1].argmax(dim=-1).save()

        # If prediction on clean prompt is incorrect, skip
        pred_token = model.tokenizer.decode([pred]).lower().strip()
        clean_target_token = clean_target.lower().strip()
        if pred_token != clean_target_token:
            continue

        # Form corrupt prompt of the same length as clean prompt
        corrupt_prompt_list = []
        corrupt_prompt_list = [
            random.randint(0, model.config.vocab_size) for _ in range(clean_prompt_len)
        ]
        corrupt_prompt = model.tokenizer.decode(corrupt_prompt_list)
        corrupt_prompt_len = len(model.tokenizer.encode(corrupt_prompt))
        while corrupt_prompt_len != clean_prompt_len:
            if corrupt_prompt_len > clean_prompt_len:
                corrupt_prompt_list = corrupt_prompt_list[:-1]
            elif corrupt_prompt_len < clean_prompt_len:
                corrupt_prompt_list = corrupt_prompt_list + [
                    random.randint(0, model.config.vocab_size)
                ]
            corrupt_prompt = model.tokenizer.decode(corrupt_prompt_list)
            corrupt_prompt_len = len(model.tokenizer.encode(corrupt_prompt))

        samples.append(
            {
                "clean_prompt": clean_prompt,
                "clean_ans": clean_target,
                "corrupt_prompt": corrupt_prompt,
                "target": clean_target,
            }
        )

        if len(samples) % 10 == 0:
            print(f"Generated {len(samples)} samples")

    return samples


def load_entity_data(data_dir):
    """Load character, state, and container data"""
    all_characters = json.load(
        open(
            os.path.join(
                env_utils.DEFAULT_DATA_DIR, "synthetic_entities", "characters.json"
            ),
            "r",
        )
    )
    all_objects = json.load(
        open(
            os.path.join(
                env_utils.DEFAULT_DATA_DIR, "synthetic_entities", "bottles.json"
            ),
            "r",
        )
    )
    all_states = json.load(
        open(
            os.path.join(
                env_utils.DEFAULT_DATA_DIR, "synthetic_entities", "drinks.json"
            ),
            "r",
        )
    )

    return all_characters, all_objects, all_states


def load_model(model_name, is_remote=False, cache_dir=None):
    """Load the language model"""

    if is_remote:
        model = LanguageModel("meta-llama/Meta-Llama-3.1-405B-Instruct")
    else:
        model = LanguageModel(
            "meta-llama/Meta-Llama-3-70B-Instruct",
            device_map="auto",
            torch_dtype=torch.float16,
            dispatch=True,
        )
    return model


def find_correct_samples(model, dataset, batch_size=10, num_samples=50):
    """Find samples where both clean and corrupt prompts produce correct predictions"""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    corrects = []
    for bi, batch in enumerate(tqdm(dataloader, desc="Finding correct samples")):
        corrupt_prompt = batch["corrupt_prompt"]
        clean_prompt = batch["clean_prompt"]
        corrupt_target = batch["corrupt_ans"]
        clean_target = batch["clean_ans"]

        with torch.no_grad():
            with model.trace() as tracer:
                with tracer.invoke(clean_prompt):
                    clean_pred = model.lm_head.output[:, -1].argmax(dim=-1).save()

                with tracer.invoke(corrupt_prompt):
                    corrupt_pred = model.lm_head.output[:, -1].argmax(dim=-1).save()

        for i in range(len(clean_pred)):
            clean_pred_token = model.tokenizer.decode([clean_pred[i]]).lower().strip()
            corrupt_pred_token = (
                model.tokenizer.decode([corrupt_pred[i]]).lower().strip()
            )
            clean_target_token = clean_target[i].lower().strip()
            corrupt_target_token = corrupt_target[i].lower().strip()
            index = bi * batch_size + i
            if (
                clean_pred_token == clean_target_token
                and corrupt_pred_token == corrupt_target_token
            ):
                corrects.append(index)

        del clean_pred, corrupt_pred
        torch.cuda.empty_cache()

        if len(corrects) >= num_samples:
            corrects = corrects[:num_samples]
            break

    print(f"Found {len(corrects)} correct samples")
    return corrects


def load_or_init_tracing_results(results_path, start_token=180, start_layer=0):
    """Load existing tracing results or initialize new ones"""
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            tracing_results = json.load(f)
        start_token = int(list(tracing_results.keys())[-1])
        start_layer = int(list(tracing_results[start_token].keys())[-1]) + 1
        return defaultdict(dict, tracing_results), start_token, start_layer
    else:
        return defaultdict(dict), start_token, start_layer


def run_tracing_experiment(model, dataloader, start_token, start_layer, results_path):
    """Run the layer tracing experiment"""
    tracing_results, start_token, start_layer = load_or_init_tracing_results(
        results_path, start_token, start_layer
    )

    for t in tqdm(range(start_token, 128, -1)):
        for layer_idx in range(start_layer, model.config.num_hidden_layers):
            correct, total = 0, 0

            for bi, batch in enumerate(dataloader):
                corrupt_prompt = batch["corrupt_prompt"]
                clean_prompt = batch["clean_prompt"]
                target = batch["target"]
                batch_size = len(target)

                with torch.no_grad():
                    with model.trace() as tracer:
                        with tracer.invoke(corrupt_prompt):
                            corrupt_layer_out = (
                                model.model.layers[layer_idx].output[0][:, t].clone()
                            )

                        with tracer.invoke(clean_prompt):
                            model.model.layers[layer_idx].output[0][:, t] = (
                                corrupt_layer_out
                            )
                            pred = model.lm_head.output[:, -1].argmax(dim=-1).save()

                for i in range(batch_size):
                    pred_token = model.tokenizer.decode([pred[i]]).lower().strip()
                    target_token = target[i].lower().strip()
                    if pred_token == target_token:
                        correct += 1
                    total += 1

                del corrupt_layer_out, pred
                torch.cuda.empty_cache()

            acc = round(correct / total, 2)
            tracing_results[t][layer_idx] = acc
            print(f"Token: {t} | Layer: {layer_idx} | Accuracy: {acc}")

            # Save results periodically
            if (layer_idx + 1) % 10 == 0:
                with open(results_path, "w") as f:
                    json.dump(tracing_results, f, indent=4)

    return tracing_results
