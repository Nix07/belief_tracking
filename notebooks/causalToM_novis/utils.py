import random
import sys

import torch
from tqdm import tqdm

sys.path.append("../..")
from src.dataset import Dataset, Sample

random.seed(10)


def error_detection(model, dataloader, is_remote=False):
    """
    Evaluates model performance and identifies errors by comparing predictions on clean and corrupt prompts.

    Args:
        model: The language model to evaluate
        dataloader: DataLoader containing clean and corrupt prompts
        is_remote (bool): Whether to run model inference remotely

    Returns:
        tuple: (accuracy, list of error indices)
    """
    correct, total = 0, 0
    errors = []

    for bi, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        clean_prompt = batch["clean_prompt"][0]
        corrupt_prompt = batch["corrupt_prompt"][0]
        clean_target = batch["clean_ans"][0]
        corrupt_target = batch["corrupt_ans"][0]
        clean_target = batch["clean_ans"][0]

        with torch.no_grad():
            with model.trace(remote=is_remote) as tracer:
                with tracer.invoke(clean_prompt):
                    clean_pred = (
                        model.lm_head.output[0, -1].argmax(dim=-1).item().save()
                    )

                with tracer.invoke(corrupt_prompt):
                    corrupt_pred = (
                        model.lm_head.output[0, -1].argmax(dim=-1).item().save()
                    )

        if (
            model.tokenizer.decode([clean_pred]).lower().strip() == clean_target
            and model.tokenizer.decode([corrupt_pred]).lower().strip() == corrupt_target
        ):
            correct += 1
        else:
            errors.append(bi)
        total += 1

        del clean_pred, corrupt_pred
        torch.cuda.empty_cache()

    return correct / total, errors


def get_binding_addr_and_payload(all_characters, all_objects, all_states, n_samples):
    """
    Generates samples for binding address and payload by creating clean and corrupt configurations.

    Args:
        all_characters (list): List of available characters
        all_objects (list): List of available objects/containers
        all_states (list): List of available states
        n_samples (int): Number of samples to generate

    Returns:
        list: List of dictionaries containing clean and corrupt samples with their configurations
    """
    clean_configs, corrupt_configs = [], []
    samples = []

    for idx in range(n_samples):
        template_idx = 2
        characters = random.sample(all_characters, 2)
        containers = random.sample(all_objects, 2)
        states = random.sample(all_states, 2)

        sample = Sample(
            template_idx=template_idx,
            characters=characters,
            containers=containers,
            states=states,
        )
        clean_configs.append(sample)

        sample = Sample(
            template_idx=template_idx,
            characters=list(reversed(characters)),
            containers=list(reversed(containers)),
            states=list(reversed(states)),
        )
        corrupt_configs.append(sample)

    clean_dataset = Dataset(clean_configs)
    corrupt_dataset = Dataset(corrupt_configs)

    for idx in range(n_samples):
        random_container_idx = random.choice([0, 1])

        clean = clean_dataset.__getitem__(
            idx,
            set_container=random_container_idx,
            set_character=random_container_idx,
        )
        corrupt = corrupt_dataset.__getitem__(
            idx,
            set_container=1 ^ random_container_idx,
            set_character=1 ^ random_container_idx,
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
                "corrupt_characters": corrupt["characters"],
                "corrupt_objects": corrupt["objects"],
                "corrupt_states": corrupt["states"],
                "corrupt_story": corrupt["story"],
                "corrupt_question": corrupt["question"],
                "corrupt_prompt": corrupt["prompt"],
                "corrupt_ans": corrupt["target"],
                "target": " " + clean_configs[idx].states[1 ^ random_container_idx],
            }
        )

    return samples


def get_answer_lookback_payload(
    all_characters,
    all_objects,
    all_states,
    n_samples,
):
    """
    Generates samples for answer lookback payload by creating clean and corrupt configurations
    with different character-object-state mappings.

    Args:
        all_characters (list): List of available characters
        all_objects (list): List of available objects/containers
        all_states (list): List of available states
        n_samples (int): Number of samples to generate

    Returns:
        list: List of dictionaries containing clean and corrupt samples with their configurations
    """
    clean_configs, corrupt_configs = [], []
    samples = []

    for idx in range(n_samples):
        template_idx = 2
        characters = random.sample(all_characters, 2)
        containers = random.sample(all_objects, 2)
        states = random.sample(all_states, 2)

        sample = Sample(
            template_idx=template_idx,
            characters=characters,
            containers=containers,
            states=states,
        )
        clean_configs.append(sample)

        characters = random.sample(all_characters, 2)
        containers = random.sample(all_objects, 2)
        states = random.sample(all_states, 2)
        sample = Sample(
            template_idx=template_idx,
            characters=characters,
            containers=containers,
            states=states,
        )
        corrupt_configs.append(sample)

    clean_dataset = Dataset(clean_configs)
    corrupt_dataset = Dataset(corrupt_configs)

    for idx in range(n_samples):
        random_choice = random.choice([0, 1])
        clean = clean_dataset.__getitem__(
            idx,
            set_character=random_choice,
            set_container=1 ^ random_choice,
        )
        corrupt = corrupt_dataset.__getitem__(
            idx,
            set_character=random_choice,
            set_container=random_choice,
        )

        samples.append(
            {
                "clean_characters": clean["characters"],
                "clean_objects": clean["objects"],
                "clean_states": clean["states"],
                "clean_story": clean["story"],
                "clean_question": clean["question"],
                "clean_ans": clean["target"],
                "clean_prompt": clean["prompt"],
                "corrupt_characters": corrupt["characters"],
                "corrupt_objects": corrupt["objects"],
                "corrupt_states": corrupt["states"],
                "corrupt_story": corrupt["story"],
                "corrupt_question": corrupt["question"],
                "corrupt_ans": corrupt["target"],
                "corrupt_prompt": corrupt["prompt"],
                "target": corrupt["target"],
            }
        )

    return samples


def get_answer_lookback_pointer(all_characters, all_objects, all_states, n_samples):
    """
    Generates samples for answer lookback pointer by creating clean and corrupt configurations
    with different states.

    Args:
        all_characters (list): List of available characters
        all_objects (list): List of available objects/containers
        all_states (list): List of available states
        n_samples (int): Number of samples to generate

    Returns:
        list: List of dictionaries containing clean and corrupt samples with their configurations
    """
    clean_configs, corrupt_configs = [], []
    samples = []

    for idx in range(n_samples):
        template_idx = 2
        characters = random.sample(all_characters, 2)
        containers = random.sample(all_objects, 2)
        states = random.sample(all_states, 2)

        sample = Sample(
            template_idx=template_idx,
            characters=characters,
            containers=containers,
            states=states,
        )
        clean_configs.append(sample)

        new_states = random.sample(all_states, 2)
        while new_states[0] in states or new_states[1] in states:
            new_states = random.sample(all_states, 2)

        sample = Sample(
            template_idx=template_idx,
            characters=list(reversed(characters)),
            containers=list(reversed(containers)),
            states=new_states,
        )
        corrupt_configs.append(sample)

    clean_dataset = Dataset(clean_configs)
    corrupt_dataset = Dataset(corrupt_configs)

    for idx in range(n_samples):
        random_choice = random.choice([0, 1])

        clean = clean_dataset.__getitem__(
            idx,
            set_container=random_choice,
            set_character=random_choice,
        )
        corrupt = corrupt_dataset.__getitem__(
            idx,
            set_container=1 ^ random_choice,
            set_character=1 ^ random_choice,
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
                "corrupt_characters": corrupt["characters"],
                "corrupt_objects": corrupt["objects"],
                "corrupt_states": corrupt["states"],
                "corrupt_story": corrupt["story"],
                "corrupt_question": corrupt["question"],
                "corrupt_prompt": corrupt["prompt"],
                "corrupt_ans": corrupt["target"],
                "target": " " + clean_configs[idx].states[1 ^ random_choice],
            }
        )

    return samples


def get_query_charac_oi(
    all_characters,
    all_objects,
    all_states,
    n_samples,
):
    """
    Generates samples for aligning queried character OI by creating clean and corrupt configurations
    with different states and character-object mappings.

    Args:
        all_characters (list): List of available characters
        all_objects (list): List of available objects/containers
        all_states (list): List of available states
        n_samples (int): Number of samples to generate

    Returns:
        list: List of dictionaries containing clean and corrupt samples with their configurations
    """
    clean_configs, corrupt_configs = [], []
    samples = []

    for idx in range(n_samples):
        template_idx = 2
        characters = random.sample(all_characters, 2)
        containers = random.sample(all_objects, 2)
        states = random.sample(all_states, 2)

        sample = Sample(
            template_idx=template_idx,
            characters=characters,
            containers=containers,
            states=states,
        )
        clean_configs.append(sample)

        new_states = random.sample(all_states, 2)
        while new_states[0] in states or new_states[1] in states:
            new_states = random.sample(all_states, 2)

        sample = Sample(
            template_idx=template_idx,
            characters=list(reversed(characters)),
            containers=list(reversed(containers)),
            states=new_states,
        )
        corrupt_configs.append(sample)

    clean_dataset = Dataset(clean_configs)
    corrupt_dataset = Dataset(corrupt_configs)

    for idx in range(n_samples):
        random_choice = random.choice([0, 1])

        clean = clean_dataset.__getitem__(
            idx,
            set_container=1 ^ random_choice,
            set_character=random_choice,
        )
        corrupt = corrupt_dataset.__getitem__(
            idx,
            set_container=1 ^ random_choice,
            set_character=1 ^ random_choice,
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
                "corrupt_characters": corrupt["characters"],
                "corrupt_objects": corrupt["objects"],
                "corrupt_states": corrupt["states"],
                "corrupt_story": corrupt["story"],
                "corrupt_question": corrupt["question"],
                "corrupt_prompt": corrupt["prompt"],
                "corrupt_ans": corrupt["target"],
                "target": " " + clean_configs[idx].states[1 ^ random_choice],
            }
        )

    return samples


def get_query_object_oi(
    all_characters,
    all_objects,
    all_states,
    n_samples,
):
    """
    Generates samples for aligning queried object OI by creating clean and corrupt configurations
    with different states and object-character mappings.

    Args:
        all_characters (list): List of available characters
        all_objects (list): List of available objects/containers
        all_states (list): List of available states
        n_samples (int): Number of samples to generate

    Returns:
        list: List of dictionaries containing clean and corrupt samples with their configurations
    """
    clean_configs, corrupt_configs = [], []
    samples = []

    for idx in range(n_samples):
        template_idx = 2
        characters = random.sample(all_characters, 2)
        containers = random.sample(all_objects, 2)
        states = random.sample(all_states, 2)

        sample = Sample(
            template_idx=template_idx,
            characters=characters,
            containers=containers,
            states=states,
        )
        clean_configs.append(sample)

        new_states = random.sample(all_states, 2)
        while new_states[0] in states or new_states[1] in states:
            new_states = random.sample(all_states, 2)

        sample = Sample(
            template_idx=template_idx,
            characters=list(reversed(characters)),
            containers=list(reversed(containers)),
            states=new_states,
        )
        corrupt_configs.append(sample)

    clean_dataset = Dataset(clean_configs)
    corrupt_dataset = Dataset(corrupt_configs)

    for idx in range(n_samples):
        random_choice = random.choice([0, 1])

        clean = clean_dataset.__getitem__(
            idx,
            set_container=random_choice,
            set_character=1 ^ random_choice,
        )
        corrupt = corrupt_dataset.__getitem__(
            idx,
            set_container=1 ^ random_choice,
            set_character=1 ^ random_choice,
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
                "corrupt_characters": corrupt["characters"],
                "corrupt_objects": corrupt["objects"],
                "corrupt_states": corrupt["states"],
                "corrupt_story": corrupt["story"],
                "corrupt_question": corrupt["question"],
                "corrupt_prompt": corrupt["prompt"],
                "corrupt_ans": corrupt["target"],
                "target": " " + clean_configs[idx].states[1 ^ random_choice],
            }
        )

    return samples


def get_object_oi_exps(
    all_characters,
    all_containers,
    all_states,
    n_samples,
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
    clean_configs, corrupt_configs = [], []
    samples = []

    for idx in range(n_samples):
        template_idx = 2
        characters = random.sample(all_characters, 2)
        containers = random.sample(all_containers, 2)
        states = random.sample(all_states, 2)

        sample = Sample(
            template_idx=template_idx,
            characters=characters,
            containers=containers,
            states=states,
        )
        clean_configs.append(sample)

        new_states = random.sample(all_states, 2)
        while new_states[0] in states or new_states[1] in states:
            new_states = random.sample(all_states, 2)

        sample = Sample(
            template_idx=template_idx,
            characters=list(reversed(characters)),
            containers=list(reversed(containers)),
            states=new_states,
        )
        corrupt_configs.append(sample)

    clean_dataset = Dataset(clean_configs)
    corrupt_dataset = Dataset(corrupt_configs)

    for idx in range(n_samples):
        random_choice = random.choice([0, 1])

        clean = clean_dataset.__getitem__(
            idx,
            set_container=random_choice,
            set_character=1 ^ random_choice,
        )
        corrupt = corrupt_dataset.__getitem__(
            idx,
            set_container=1 ^ random_choice,
            set_character=1 ^ random_choice,
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
                "corrupt_characters": corrupt["characters"],
                "corrupt_objects": corrupt["objects"],
                "corrupt_states": corrupt["states"],
                "corrupt_story": corrupt["story"],
                "corrupt_question": corrupt["question"],
                "corrupt_prompt": corrupt["prompt"],
                "corrupt_ans": corrupt["target"],
                "target": " " + clean_configs[idx].states[1 ^ random_choice],
            }
        )

    return samples


def get_character_oi_exps(
    all_characters,
    all_objects,
    all_states,
    n_samples,
):
    clean_configs, corrupt_configs = [], []
    samples = []

    for idx in range(n_samples):
        template_idx = 2
        characters = random.sample(all_characters, 2)
        containers = random.sample(all_objects, 2)
        states = random.sample(all_states, 2)

        sample = Sample(
            template_idx=template_idx,
            characters=characters,
            containers=containers,
            states=states,
        )
        clean_configs.append(sample)

        new_states = random.sample(all_states, 2)
        while new_states[0] in states or new_states[1] in states:
            new_states = random.sample(all_states, 2)

        sample = Sample(
            template_idx=template_idx,
            characters=list(reversed(characters)),
            containers=list(reversed(containers)),
            states=new_states,
        )
        corrupt_configs.append(sample)

    clean_dataset = Dataset(clean_configs)
    corrupt_dataset = Dataset(corrupt_configs)

    for idx in range(n_samples):
        random_choice = random.choice([0, 1])

        clean = clean_dataset.__getitem__(
            idx,
            set_container=1 ^ random_choice,
            set_character=random_choice,
        )
        corrupt = corrupt_dataset.__getitem__(
            idx,
            set_container=1 ^ random_choice,
            set_character=1 ^ random_choice,
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
                "corrupt_characters": corrupt["characters"],
                "corrupt_objects": corrupt["objects"],
                "corrupt_states": corrupt["states"],
                "corrupt_story": corrupt["story"],
                "corrupt_question": corrupt["question"],
                "corrupt_prompt": corrupt["prompt"],
                "corrupt_ans": corrupt["target"],
                "target": " " + clean_configs[idx].states[1 ^ random_choice],
            }
        )

    return samples