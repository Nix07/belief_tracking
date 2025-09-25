import random
import sys

import torch
from tqdm import tqdm

sys.path.append("../..")
from src.dataset import Dataset, Sample

random.seed(10)


def error_detection(model, dataloader, is_remote=False):
    """
    Evaluates model performance and identifies errors by comparing predictions on clean and counterfactual prompts.

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


def get_visibility_lookback_data(
    all_characters: list,
    all_objects: list,
    all_states: list,
    n_samples: int,
) -> list:
    """
    Generates samples for the visibility lookback experiment.
    In clean sample characters can observe each other, but in counterfactual sample they cannot.

    Args:
        all_characters (list): List of available characters
        all_objects (list): List of available objects/containers
        all_states (list): List of available states
        n_samples (int): Number of samples to generate

    Returns:
        list: List of dictionaries containing clean and corrupt samples with their configurations
    """
    clean_configs, counterfactual_configs, samples = [], [], []

    for idx in range(n_samples):
        characters = random.sample(all_characters, 2)
        objects = random.sample(all_objects, 2)
        states = random.sample(all_states, 2)

        no_vis_sample = Sample(
            template_idx=0,
            characters=characters,
            objects=objects,
            states=states,
        )

        new_states = random.sample(all_states, 2)
        new_characters = random.sample(all_characters, 2)
        new_objects = random.sample(all_objects, 2)
        while new_states[0] in states or new_states[1] in states:
            new_states = random.sample(all_states, 2)
        while new_characters[0] in characters or new_characters[1] in characters:
            new_characters = random.sample(all_characters, 2)
        while new_objects[0] in objects or new_objects[1] in objects:
            new_objects = random.sample(all_objects, 2)

        vis_sample = Sample(
            template_idx=1,
            characters=new_characters,
            objects=new_objects,
            states=new_states,
        )

        clean_configs.append(no_vis_sample)
        counterfactual_configs.append(vis_sample)

    clean_dataset = Dataset(clean_configs)
    counterfactual_dataset = Dataset(counterfactual_configs)

    for idx in range(n_samples):
        clean = clean_dataset.__getitem__(
            idx,
            set_character=0,
            set_container=1,
        )
        counterfactual = counterfactual_dataset.__getitem__(
            idx,
            set_character=0,
            set_container=1,
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
                "target": " " + clean_configs[idx].states[1],
            }
        )
    return samples
