import os
import random
import sys

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


def get_different_permutation(items: list) -> list:
    """
    Returns a permutation of `items` that differs from the original order using rotation.
    """
    if len(items) < 2:
        return items[:]

    return items[1:] + items[:1]


def error_detection(
    model: LanguageModel, dataloader: DataLoader, is_remote: bool = False
) -> tuple[float, list]:
    """
    Evaluates model performance and identifies errors by comparing predictions on both clean and counterfactual prompts.

    Args:
        model: The language model to evaluate
        dataloader: DataLoader containing clean and counterfactual prompts
        is_remote (bool): Whether to run model inference remotely

    Returns:
        tuple: (accuracy, list of error indices)
    """
    correct, total = 0, 0
    errors = []

    for bi, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        clean_prompt = batch["clean_prompt"][0]
        counterfactual_prompt = batch["counterfactual_prompt"][0]
        clean_target = batch["clean_ans"][0]
        counterfactual_target = batch["counterfactual_ans"][0]
        clean_target = batch["clean_ans"][0]

        with torch.no_grad():
            with model.trace(remote=is_remote) as tracer:
                with tracer.invoke(clean_prompt):
                    clean_pred = (
                        model.lm_head.output[0, -1].argmax(dim=-1).item().save()
                    )

                with tracer.invoke(counterfactual_prompt):
                    counterfactual_pred = (
                        model.lm_head.output[0, -1].argmax(dim=-1).item().save()
                    )

        if (
            model.tokenizer.decode([clean_pred]).lower().strip() == clean_target
            and model.tokenizer.decode([counterfactual_pred]).lower().strip()
            == counterfactual_target
        ):
            correct += 1
        else:
            errors.append(bi)
        total += 1

        del clean_pred, counterfactual_pred
        torch.cuda.empty_cache()

    return correct / total, errors


def get_reversed_sentence_counterfacts(
    all_characters: list, all_objects: list, all_states: list, n_samples: int, new_config: bool = False
) -> list:
    """
    Generates counterfactual samples by reversing the sentences and keeping the other elements the clean.

    Args:
        all_characters (list): List of available characters
        all_objects (list): List of available objects
        all_states (list): List of available states
        n_samples (int): Number of samples to generate
        new_config (bool): Whether to use the new config with 3 entities

    Returns:
        list: List of dictionaries containing clean and counterfactual samples with their configurations
    """
    clean_configs, counterfactual_configs = [], []
    samples = []

    for idx in range(n_samples):
        template_idx = 2 if not new_config else 8
        if template_idx == 2:
            characters = random.sample(all_characters, 2)
            objects = random.sample(all_objects, 2)
            states = random.sample(all_states, 2)
        else:
            characters = random.sample(all_characters, 3)
            objects = random.sample(all_objects, 3)
            states = random.sample(all_states, 3)

        sample = Sample(
            template_idx=template_idx,
            characters=characters,
            objects=objects,
            states=states,
        )
        clean_configs.append(sample)

        # To create the counterfactual config, reverse the order of characters, objects, and states.
        sample = Sample(
            template_idx=template_idx,
            characters=get_different_permutation(characters),
            objects=get_different_permutation(objects),
            states=get_different_permutation(states),
        )
        counterfactual_configs.append(sample)

    clean_dataset = Dataset(clean_configs)
    counterfactual_dataset = Dataset(counterfactual_configs)

    for idx in range(n_samples):
        if not new_config:
            random_object_idx = random.choice([0, 1])
        else:
            random_object_idx = random.choice([0, 1, 2])
        clean = clean_dataset.__getitem__(
            idx,
            set_container=random_object_idx,
            set_character=random_object_idx,
        )
        
        if not new_config:
            set_container = (random_object_idx - 1) % 2
            set_character = (random_object_idx - 1) % 2
            target = " " + clean_configs[idx].states[set_container]
        else:
            set_container = (random_object_idx - 1) % 3
            set_character = (random_object_idx - 1) % 3
            target = " " + clean_configs[idx].states[(set_container - 1) % 3]
        counterfactual = counterfactual_dataset.__getitem__(
            idx,
            set_container=set_container,
            set_character=set_character,
        )

        samples.append(
            {
                "clean_characters": clean["characters"],
                "clean_objects": clean["objects"],
                "clean_states": clean["states"],
                "clean_story": clean["story"],
                "clean_question": clean["question"],
                "clean_prompt": clean["prompt"],
                "clean_ans": clean["target"],
                "counterfactual_characters": counterfactual["characters"],
                "counterfactual_objects": counterfactual["objects"],
                "counterfactual_states": counterfactual["states"],
                "counterfactual_story": counterfactual["story"],
                "counterfactual_question": counterfactual["question"],
                "counterfactual_prompt": counterfactual["prompt"],
                "counterfactual_ans": counterfactual["target"],
                "target": target,
            }
        )

    return samples


def get_answer_lookback_payload(
    all_characters: list,
    all_objects: list,
    all_states: list,
    n_samples: int,
    new_config: bool = False,
) -> list:
    """
    Generates samples for answer lookback payload by creating clean and counterfactual configurations
    with different character-object-state mappings.

    Args:
        all_characters (list): List of available characters
        all_objects (list): List of available objects/containers
        all_states (list): List of available states
        n_samples (int): Number of samples to generate
        new_config (bool): Whether to use the new config with 3 entities

    Returns:
        list: List of dictionaries containing clean and counterfactual samples with their configurations.
    """
    clean_configs, counterfactual_configs = [], []
    samples = []

    for idx in range(n_samples):
        template_idx = 2 if not new_config else 8
        if template_idx == 2:
            characters = random.sample(all_characters, 2)
            objects = random.sample(all_objects, 2)
            states = random.sample(all_states, 2)
        else:
            characters = random.sample(all_characters, 3)
            objects = random.sample(all_objects, 3)
            states = random.sample(all_states, 3)

        sample = Sample(
            template_idx=template_idx,
            characters=characters,
            objects=objects,
            states=states,
        )
        clean_configs.append(sample)

        if template_idx == 2:
            new_states = random.sample(all_states, 2)
            while new_states[0] in states or new_states[1] in states:
                new_states = random.sample(all_states, 2)
        else:
            new_states = random.sample(all_states, 3)
            while new_states[0] in states or new_states[1] in states or new_states[2] in states:
                new_states = random.sample(all_states, 3)

        sample = Sample(
            template_idx=template_idx,
            characters=get_different_permutation(characters),
            objects=get_different_permutation(objects),
            states=new_states,
        )
        counterfactual_configs.append(sample)

    clean_dataset = Dataset(clean_configs)
    corrupt_dataset = Dataset(counterfactual_configs)

    for idx in range(n_samples):
        if not new_config:
            random_choice = random.choice([0, 1])
        else:
            random_choice = random.choice([0, 1, 2])
        clean = clean_dataset.__getitem__(
            idx,
            set_container=random_choice,
            set_character=random_choice,
        )

        if not new_config:
            set_container = (random_choice - 1) % 2
            set_character = (random_choice - 1) % 2
        else:
            set_container = (random_choice - 1) % 3
            set_character = (random_choice - 1) % 3               

        counterfactual = corrupt_dataset.__getitem__(
            idx,
            set_container=set_container,
            set_character=set_character,
        )

        samples.append(
            {
                "clean_characters": clean["characters"],
                "clean_objects": clean["objects"],
                "clean_states": clean["states"],
                "clean_story": clean["story"],
                "clean_question": clean["question"],
                "clean_prompt": clean["prompt"],
                "clean_ans": clean["target"],
                "counterfactual_characters": counterfactual["characters"],
                "counterfactual_objects": counterfactual["objects"],
                "counterfactual_states": counterfactual["states"],
                "counterfactual_story": counterfactual["story"],
                "counterfactual_question": counterfactual["question"],
                "counterfactual_prompt": counterfactual["prompt"],
                "counterfactual_ans": counterfactual["target"],
                "target": counterfactual["target"],
            }
        )

    return samples


def get_reversed_sent_diff_state_counterfacts(
    all_characters: list, all_objects: list, all_states: list, n_samples: int, new_config: bool = False
) -> list:
    """
    Generates counterfactual samples by reversing the sentence and changing the state.

    Args:
        all_characters (list): List of available characters
        all_objects (list): List of available objects/containers
        all_states (list): List of available states
        n_samples (int): Number of samples to generate
        new_config (bool): Whether to use the new config with 3 entities
    
    Returns:
        list: List of dictionaries containing clean and corrupt samples with their configurations
    """
    clean_configs, counterfactual_configs = [], []
    samples = []

    for idx in range(n_samples):
        template_idx = 2 if not new_config else 8
        if template_idx == 2:
            characters = random.sample(all_characters, 2)
            objects = random.sample(all_objects, 2)
            states = random.sample(all_states, 2)
        else:
            characters = random.sample(all_characters, 3)
            objects = random.sample(all_objects, 3)
            states = random.sample(all_states, 3)

        sample = Sample(
            template_idx=template_idx,
            characters=characters,
            objects=objects,
            states=states,
        )
        clean_configs.append(sample)

        if template_idx == 2:
            new_states = random.sample(all_states, 2)
            while new_states[0] in states or new_states[1] in states:
                new_states = random.sample(all_states, 2)
        else:
            new_states = random.sample(all_states, 3)
            while new_states[0] in states or new_states[1] in states or new_states[2] in states:
                new_states = random.sample(all_states, 3)

        sample = Sample(
            template_idx=template_idx,
            characters=get_different_permutation(characters),
            objects=get_different_permutation(objects),
            states=new_states,
        )
        counterfactual_configs.append(sample)

    clean_dataset = Dataset(clean_configs)
    corrupt_dataset = Dataset(counterfactual_configs)

    for idx in range(n_samples):
        if not new_config:
            random_choice = random.choice([0, 1])
        else:
            random_choice = random.choice([0, 1, 2])
        clean = clean_dataset.__getitem__(
            idx,
            set_container=random_choice,
            set_character=random_choice,
        )

        if not new_config:
            set_container = (random_choice - 1) % 2
            set_character = (random_choice - 1) % 2
        else:
            set_container = (random_choice - 1) % 3
            set_character = (random_choice - 1) % 3
        counterfactual = corrupt_dataset.__getitem__(
            idx,
            set_container=set_container,
            set_character=set_character,
        )

        samples.append(
            {
                "clean_characters": clean["characters"],
                "clean_objects": clean["objects"],
                "clean_states": clean["states"],
                "clean_story": clean["story"],
                "clean_question": clean["question"],
                "clean_prompt": clean["prompt"],
                "clean_ans": clean["target"],
                "counterfactual_characters": counterfactual["characters"],
                "counterfactual_objects": counterfactual["objects"],
                "counterfactual_states": counterfactual["states"],
                "counterfactual_story": counterfactual["story"],
                "counterfactual_question": counterfactual["question"],
                "counterfactual_prompt": counterfactual["prompt"],
                "counterfactual_ans": counterfactual["target"],
                "target": " " + clean_configs[idx].states[set_container],
            }
        )

    return samples


def get_query_charac_oi(
    all_characters: list,
    all_objects: list,
    all_states: list,
    n_samples,
    new_config: bool = False,
) -> list:
    """
    Generates counterfactual samples for aligning queried character OI by reversing the sentence and changing the state.
    Also, updates the object in the question.

    Args:
        all_characters (list): List of available characters
        all_objects (list): List of available objects/containers
        all_states (list): List of available states
        n_samples (int): Number of samples to generate

    Returns:
        list: List of dictionaries containing clean and counterfactual samples with their configurations.
    """
    clean_configs, counterfactual_configs = [], []
    samples = []

    for idx in range(n_samples):
        template_idx = 2 if not new_config else 8
        if template_idx == 2:
            characters = random.sample(all_characters, 2)
            objects = random.sample(all_objects, 2)
            states = random.sample(all_states, 2)
        else:
            characters = random.sample(all_characters, 3)
            objects = random.sample(all_objects, 3)
            states = random.sample(all_states, 3)

        sample = Sample(
            template_idx=template_idx,
            characters=characters,
            objects=objects,
            states=states,
        )
        clean_configs.append(sample)

        if template_idx == 2:
            new_states = random.sample(all_states, 2)
            while new_states[0] in states or new_states[1] in states:
                new_states = random.sample(all_states, 2)
        else:
            new_states = random.sample(all_states, 3)
            while new_states[0] in states or new_states[1] in states or new_states[2] in states:
                new_states = random.sample(all_states, 3)

        sample = Sample(
            template_idx=template_idx,
            characters=get_different_permutation(get_different_permutation(characters)),
            objects=get_different_permutation(get_different_permutation(objects)),
            states=new_states,
        )
        counterfactual_configs.append(sample)

    clean_dataset = Dataset(clean_configs)
    counterfactual_dataset = Dataset(counterfactual_configs)

    for idx in range(n_samples):
        if not new_config:
            random_choice = random.choice([0, 1])
        else:
            random_choice = random.choice([0, 1, 2])

        if not new_config:
            clean = clean_dataset.__getitem__(
                idx,
                set_container=1 ^ random_choice,
                set_character=random_choice,
            )
            counterfactual = counterfactual_dataset.__getitem__(
                idx,
                set_container=1 ^ random_choice,
                set_character=1 ^ random_choice,
            )
            target = " " + clean_configs[idx].states[1 ^ random_choice]
        else:
            clean = clean_dataset.__getitem__(
                idx,
                set_container=random_choice,
                set_character=(random_choice - 1) % 3,
            )
            counterfactual = counterfactual_dataset.__getitem__(
                idx,
                set_container=(random_choice) % 3,
                set_character=(random_choice) % 3,
            )
            target = " " + clean_configs[idx].states[(random_choice) % 3]

        samples.append(
            {
                "clean_characters": clean["characters"],
                "clean_objects": clean["objects"],
                "clean_states": clean["states"],
                "clean_story": clean["story"],
                "clean_question": clean["question"],
                "clean_prompt": clean["prompt"],
                "clean_ans": clean["target"],
                "counterfactual_characters": counterfactual["characters"],
                "counterfactual_objects": counterfactual["objects"],
                "counterfactual_states": counterfactual["states"],
                "counterfactual_story": counterfactual["story"],
                "counterfactual_question": counterfactual["question"],
                "counterfactual_prompt": counterfactual["prompt"],
                "counterfactual_ans": counterfactual["target"],
                "target": target,
            }
        )

    return samples


def get_query_object_oi(
    all_characters: list,
    all_objects: list,
    all_states: list,
    n_samples: int,
    new_config: bool = False,
) -> list:
    """
    Generates counterfactual samples for aligning queried object OI by reversing the sentence and changing the state.
    Also, updates the character in the question.

    Args:
        all_characters (list): List of available characters
        all_objects (list): List of available objects/containers
        all_states (list): List of available states
        n_samples (int): Number of samples to generate

    Returns:
        list: List of dictionaries containing clean and counterfactual samples with their configurations.
    """
    clean_configs, counterfactual_configs = [], []
    samples = []

    for idx in range(n_samples):
        template_idx = 2 if not new_config else 8
        if template_idx == 2:
            characters = random.sample(all_characters, 2)
            objects = random.sample(all_objects, 2)
            states = random.sample(all_states, 2)
        else:
            characters = random.sample(all_characters, 3)
            objects = random.sample(all_objects, 3)
            states = random.sample(all_states, 3)

        sample = Sample(
            template_idx=template_idx,
            characters=characters,
                objects=objects,
                states=states,
            )
        clean_configs.append(sample)

        if template_idx == 2:
            new_states = random.sample(all_states, 2)
            while new_states[0] in states or new_states[1] in states:
                new_states = random.sample(all_states, 2)
        else:
            new_states = random.sample(all_states, 3)
            while new_states[0] in states or new_states[1] in states or new_states[2] in states:
                new_states = random.sample(all_states, 3)

        sample = Sample(
            template_idx=template_idx,
            characters=get_different_permutation(characters),
            objects=get_different_permutation(objects),
            states=new_states,
        )
        counterfactual_configs.append(sample)

    clean_dataset = Dataset(clean_configs)
    counterfactual_dataset = Dataset(counterfactual_configs)

    for idx in range(n_samples):
        if not new_config:
            random_choice = random.choice([0, 1])
        else:
            random_choice = random.choice([0, 1, 2])

        if not new_config:
            clean = clean_dataset.__getitem__(
                idx,
                set_container=random_choice,
                set_character=1 ^ random_choice,
            )
            counterfactual = counterfactual_dataset.__getitem__(
                idx,
                set_container=1 ^ random_choice,
                set_character=1 ^ random_choice,
            )
            target = " " + clean_configs[idx].states[1 ^ random_choice]
        else:
            clean = clean_dataset.__getitem__(
                idx,
                set_container=random_choice,
                set_character=(random_choice - 1) % 3,
            )
            counterfactual = counterfactual_dataset.__getitem__(
                idx,
                set_container=(random_choice - 1) % 3,
                set_character=(random_choice - 1) % 3,
            )
            target = " " + clean_configs[idx].states[(random_choice - 1) % 3]

        samples.append(
            {
                "clean_characters": clean["characters"],
                "clean_objects": clean["objects"],
                "clean_states": clean["states"],
                "clean_story": clean["story"],
                "clean_question": clean["question"],
                "clean_prompt": clean["prompt"],
                "clean_ans": clean["target"],
                "counterfactual_characters": counterfactual["characters"],
                "counterfactual_objects": counterfactual["objects"],
                "counterfactual_states": counterfactual["states"],
                "counterfactual_story": counterfactual["story"],
                "counterfactual_question": counterfactual["question"],
                "counterfactual_prompt": counterfactual["prompt"],
                "counterfactual_ans": counterfactual["target"],
                "target": target,
            }
        )

    return samples


def get_object_oi_exps(
    all_characters,
    all_containers,
    all_states,
    n_samples,
    new_config: bool = False,
):
    """
    Generates samples for object OI experiments by creating clean and corrupt configurations
    with different states and object-character mappings.

    Args:
        all_characters (list): List of available characters
        all_containers (list): List of available objects/containers
        all_states (list): List of available states
        n_samples (int): Number of samples to generate

    Returns:
        list: List of dictionaries containing clean and corrupt samples with their configurations
    """
    clean_configs, counterfactual_configs = [], []
    samples = []

    for idx in range(n_samples):
        template_idx = 2 if not new_config else 8
        if template_idx == 2:
            characters = random.sample(all_characters, 2)
            containers = random.sample(all_containers, 2)
            states = random.sample(all_states, 2)
        else:
            characters = random.sample(all_characters, 3)
            containers = random.sample(all_containers, 3)
            states = random.sample(all_states, 3)

        sample = Sample(
            template_idx=template_idx,
            characters=characters,
            objects=containers,
            states=states,
        )
        clean_configs.append(sample)

        if template_idx == 2:
            new_states = random.sample(all_states, 2)
            while new_states[0] in states or new_states[1] in states:
                new_states = random.sample(all_states, 2)
        else:
            new_states = random.sample(all_states, 3)
            while new_states[0] in states or new_states[1] in states or new_states[2] in states:
                new_states = random.sample(all_states, 3)

        sample = Sample(
            template_idx=template_idx,
            characters=get_different_permutation(characters),
            objects=get_different_permutation(containers),
            states=new_states,
        )
        counterfactual_configs.append(sample)

    clean_dataset = Dataset(clean_configs)
    counterfactual_dataset = Dataset(counterfactual_configs)

    for idx in range(n_samples):
        if not new_config:
            random_choice = random.choice([0, 1])
        else:
            random_choice = random.choice([0, 1, 2])

        if not new_config:
            clean = clean_dataset.__getitem__(
                idx,
                set_container=random_choice,
                set_character=1 ^ random_choice,
            )
            counterfactual = counterfactual_dataset.__getitem__(
                idx,
                set_container=random_choice,
                set_character=random_choice,
            )
            target = " " + clean_configs[idx].states[1 ^ random_choice]
        else:
            clean = clean_dataset.__getitem__(
                idx,
                set_container=random_choice,
                set_character=(random_choice - 1) % 3,
            )
            counterfactual = counterfactual_dataset.__getitem__(
                idx,
                set_container=(random_choice + 1) % 3,
                set_character=(random_choice + 1) % 3,
            )
            target = " " + clean_configs[idx].states[(random_choice - 1) % 3]

        samples.append(
            {
                "clean_characters": clean["characters"],
                "clean_objects": clean["objects"],
                "clean_states": clean["states"],
                "clean_story": clean["story"],
                "clean_question": clean["question"],
                "clean_prompt": clean["prompt"],
                "clean_ans": clean["target"],
                "counterfactual_characters": counterfactual["characters"],
                "counterfactual_objects": counterfactual["objects"],
                "counterfactual_states": counterfactual["states"],
                "counterfactual_story": counterfactual["story"],
                "counterfactual_question": counterfactual["question"],
                "counterfactual_prompt": counterfactual["prompt"],
                "counterfactual_ans": counterfactual["target"],
                "target": target,
            }
        )

    return samples


def get_character_oi_exps(
    all_characters,
    all_objects,
    all_states,
    n_samples,
    new_config: bool = False,
):
    """
    Generates samples for character OI experiments by creating clean and counterfactual configurations
    with different states and character-object mappings.

    Args:
        all_characters (list): List of available characters
        all_objects (list): List of available objects/containers
        all_states (list): List of available states
        n_samples (int): Number of samples to generate

    Returns:
        list: List of dictionaries containing clean and counterfactual samples with their configurations
    """
    clean_configs, counterfactual_configs = [], []
    samples = []

    for idx in range(n_samples):
        template_idx = 2 if not new_config else 8
        if template_idx == 2:
            characters = random.sample(all_characters, 2)
            containers = random.sample(all_objects, 2)
            states = random.sample(all_states, 2)
        else:
            characters = random.sample(all_characters, 3)
            containers = random.sample(all_objects, 3)
            states = random.sample(all_states, 3)

        sample = Sample(
            template_idx=template_idx,
            characters=characters,
            objects=containers,
            states=states,
        )
        clean_configs.append(sample)

        if template_idx == 2:
            new_states = random.sample(all_states, 2)
            while new_states[0] in states or new_states[1] in states:
                new_states = random.sample(all_states, 2)
        else:
            new_states = random.sample(all_states, 3)
            while new_states[0] in states or new_states[1] in states or new_states[2] in states:
                new_states = random.sample(all_states, 3)

        sample = Sample(
            template_idx=template_idx,
            characters=get_different_permutation(characters),
            objects=get_different_permutation(containers),
            states=new_states,
        )
        counterfactual_configs.append(sample)

    clean_dataset = Dataset(clean_configs)
    counterfactual_dataset = Dataset(counterfactual_configs)

    for idx in range(n_samples):
        if not new_config:
            random_choice = random.choice([0, 1])
        else:
            random_choice = random.choice([0, 1, 2])

        if not new_config:
            clean = clean_dataset.__getitem__(
                idx,
                set_container=random_choice,
                set_character=(random_choice + 1) % 2,
            )
            counterfactual = counterfactual_dataset.__getitem__(
                idx,
                set_container=random_choice,
                set_character=random_choice,
            )
            target = " " + clean_configs[idx].states[random_choice]
        else:
            clean = clean_dataset.__getitem__(
                idx,
                set_container=random_choice,
                set_character=(random_choice + 1) % 3
            )
            counterfactual = counterfactual_dataset.__getitem__(
                idx,
                set_container=(random_choice) % 3,
                set_character=(random_choice) % 3,
            )
            target = " " + clean_configs[idx].states[(random_choice) % 3]

        samples.append(
            {
                "clean_characters": clean["characters"],
                "clean_objects": clean["objects"],
                "clean_states": clean["states"],
                "clean_story": clean["story"],
                "clean_question": clean["question"],
                "clean_prompt": clean["prompt"],
                "clean_ans": clean["target"],
                "counterfactual_characters": counterfactual["characters"],
                "counterfactual_objects": counterfactual["objects"],
                "counterfactual_states": counterfactual["states"],
                "counterfactual_story": counterfactual["story"],
                "counterfactual_question": counterfactual["question"],
                "counterfactual_prompt": counterfactual["prompt"],
                "counterfactual_ans": counterfactual["target"],
                "target": target,
            }
        )

    return samples
