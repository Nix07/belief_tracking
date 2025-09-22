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


def get_visibility_lookback_data(
    all_characters,
    all_objects,
    all_states,
    n_samples,
):
    clean_configs, corrupt_configs, samples = [], [], []

    for idx in range(n_samples):
        characters = random.sample(all_characters, 2)
        containers = random.sample(all_objects, 2)
        states = random.sample(all_states, 2)

        no_vis_sample = Sample(
            template_idx=0,
            characters=characters,
            containers=containers,
            states=states,
        )

        new_states = random.sample(all_states, 2)
        new_characters = random.sample(all_characters, 2)
        new_containers = random.sample(all_objects, 2)
        while new_states[0] in states or new_states[1] in states:
            new_states = random.sample(all_states, 2)
        while new_characters[0] in characters or new_characters[1] in characters:
            new_characters = random.sample(all_characters, 2)
        while new_containers[0] in containers or new_containers[1] in containers:
            new_containers = random.sample(all_objects, 2)

        vis_sample = Sample(
            template_idx=1,
            characters=new_characters,
            containers=new_containers,
            states=new_states,
        )

        clean_configs.append(no_vis_sample)
        corrupt_configs.append(vis_sample)

    clean_dataset = Dataset(clean_configs)
    corrupt_dataset = Dataset(corrupt_configs)

    for idx in range(n_samples):
        clean = clean_dataset.__getitem__(
            idx,
            set_character=0,
            set_container=1,
        )
        corrupt = corrupt_dataset.__getitem__(
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
                "corrupt_characters": corrupt["characters"],
                "corrupt_objects": corrupt["objects"],
                "corrupt_states": corrupt["states"],
                "corrupt_story": corrupt["story"],
                "corrupt_question": corrupt["question"],
                "corrupt_prompt": corrupt["prompt"],
                "corrupt_ans": corrupt["target"],
                "target": " " + clean_configs[idx].states[1],
            }
        )
    return samples
