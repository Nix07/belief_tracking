import json
import os
import sys
from collections import defaultdict
from typing import Literal

import fire
import torch
from nnsight import LanguageModel
from run_patching_exp_utils import (
    exp_to_intervention_positions,
    exp_to_vec_type,
    free_gpu_cache,
    get_bigtom_intervention_positions,
    load_basis_directions,
    prepare_bigtom_dataset,
    prepare_dataset,
    set_seed,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src import global_utils

os.environ["NDIF_KEY"] = global_utils.load_env_var("NDIF_KEY")
os.environ["HF_TOKEN"] = global_utils.load_env_var("HF_WRITE")

query_object_indices = [-5, -4]
query_character_indices = [-8, -7]


@torch.inference_mode()
def validate(
    exp_name: Literal[
        "answer_lookback-pointer",
        "answer_lookback-payload",
        "binding_lookback-pointer_object",
        "binding_lookback-pointer_character",
        "binding_lookback-address_and_payload",
        "visibility_lookback-source",
        "visibility_lookback-payload",
        "visibility_lookback-address_and_pointer",
        "binding_lookback-pointer_character_and_object",
    ],
    lm: LanguageModel,
    layer_idx: int,
    validation_loader: DataLoader,
    projection: torch.Tensor | dict[str, torch.Tensor] | None = None,
    verbose: bool = False,
    save_outputs: bool = False,
    projection_type: Literal["full_rank", "singular_vector"] = "full_rank",
    bigtom: bool = False,
    remote: bool = False,
) -> float:
    if (
        not bigtom
        or exp_name == "answer_lookback-pointer"
        or exp_name == "answer_lookback-payload"
    ):
        intervention_positions = exp_to_intervention_positions[exp_name]
        patch_to_cache_map = {
            k: v
            for k, v in zip(
                intervention_positions["patch"], intervention_positions["cache"]
            )
        }
    else:
        # Intervention positions are dynamic for bigtom samples
        intervention_positions, patch_to_cache_map = None, None

    correct, total = 0, 0
    for batch_idx, batch in tqdm(
        enumerate(validation_loader), total=len(validation_loader)
    ):
        alt_prompts = batch["counterfactual_prompt"]
        org_prompts = batch["clean_prompt"]
        targets = (
            batch["target"] if "target" in batch else batch["counterfactual_target"]
        )
        target_tokens = lm.tokenizer(targets, return_tensors="pt").input_ids[:, -1]
        batch_size = target_tokens.size(0)
        alt_acts, org_acts_state = defaultdict(dict), defaultdict(dict)

        if bigtom and not (
            exp_name == "answer_lookback-pointer"
            or exp_name == "answer_lookback-payload"
        ):
            intervention_positions, prompt_struct = get_bigtom_intervention_positions(
                exp_name, lm, org_prompts, alt_prompts
            )
            patch_to_cache_map = {
                k: v
                for k, v in zip(
                    intervention_positions["patch"], intervention_positions["cache"]
                )
            }

        def bigtom_nnsight_request(return_logits: bool = False):
            with lm.session(remote=remote) as session:
                with lm.trace(alt_prompts[0]):
                    for t in intervention_positions["cache"]:
                        alt_acts[t] = lm.model.layers[layer_idx].output[0][:, t].clone()

                with lm.generate(
                    org_prompts[0],
                    max_new_tokens=2,
                    do_sample=False,
                    num_return_sequences=1,
                    pad_token_id=lm.tokenizer.pad_token_id,
                    eos_token_id=lm.tokenizer.eos_token_id,
                ):
                    for t in intervention_positions["patch"]:
                        curr_output = lm.model.layers[layer_idx].output[0][:, t].clone()
                        if projection is not None:
                            if isinstance(projection, dict):
                                if t in query_object_indices:
                                    proj = projection["query_obj_ordering_id"]
                                elif t in query_character_indices:
                                    proj = projection["query_charac_ordering_id"]
                                else:
                                    raise ValueError("Invalid projection type")
                            else:
                                proj = projection
                            alt_proj = torch.matmul(
                                alt_acts[patch_to_cache_map[t]], proj
                            )
                            org_proj = torch.matmul(curr_output, proj)
                            patch = curr_output - org_proj + alt_proj

                            del alt_proj, org_proj
                            free_gpu_cache()
                        else:
                            patch = alt_acts[patch_to_cache_map[t]]

                        lm.model.layers[layer_idx].output[0][:, t] = patch

                    out = lm.generator.output.save()

            pred = out[0][prompt_struct["org_prompt_len"] : -1]

            return pred

        def nnsight_request():
            with lm.trace(remote=remote) as tracer:
                with tracer.invoke(alt_prompts):
                    for t in intervention_positions["cache"]:
                        alt_acts[t] = lm.model.layers[layer_idx].output[0][:, t].clone()

                with tracer.invoke(org_prompts):
                    for t in intervention_positions["patch"]:
                        curr_output = lm.model.layers[layer_idx].output[0][:, t].clone()
                        if projection is not None:
                            if isinstance(projection, dict):
                                if t in query_object_indices:
                                    proj = projection["query_obj_ordering_id"]
                                elif t in query_character_indices:
                                    proj = projection["query_charac_ordering_id"]
                                else:
                                    raise ValueError("Invalid projection type")
                            else:
                                proj = projection
                            alt_proj = torch.matmul(
                                alt_acts[patch_to_cache_map[t]], proj
                            )
                            org_proj = torch.matmul(curr_output, proj)
                            patch = curr_output - org_proj + alt_proj

                            del alt_proj, org_proj
                            free_gpu_cache()
                        else:
                            patch = alt_acts[patch_to_cache_map[t]]

                        lm.model.layers[layer_idx].output[0][:, t] = patch

                    logits = lm.lm_head.output[:, -1]
                    pred = torch.argmax(logits, dim=-1).save()

            return pred

        pred = (
            nnsight_request()
            if not bigtom
            or exp_name == "answer_lookback-pointer"
            or exp_name == "answer_lookback-payload"
            else bigtom_nnsight_request()
        )

        pred = pred.cpu()

        for i in range(batch_size):
            pred_token = lm.tokenizer.decode(pred[i])
            if not bigtom:
                is_correct = pred_token.lower().strip() == targets[i].lower().strip()
            else:
                is_correct = pred_token.lower().strip() in targets[i].lower().strip()
            if verbose:
                print(
                    f"Correct: {is_correct} | Predicted: {pred_token.lower().strip()} | Target: {targets[i].lower().strip()}"
                )
            correct += int(is_correct)

            total += 1

        del alt_acts, alt_prompts, org_prompts, targets, target_tokens, pred
        free_gpu_cache()

    return correct / total


def get_low_rank_projection(
    exp_name: Literal[
        "answer_lookback-pointer",
        "answer_lookback-payload",
        "binding_lookback-pointer_object",
        "binding_lookback-pointer_character",
        "binding_lookback-address_and_payload",
        "visibility_lookback-source",
        "visibility_lookback-payload",
        "visibility_lookback-address_and_pointer",
        "binding_lookback-pointer_character_and_object",
    ],
    lm: LanguageModel,
    layer_idx: int,
    train_loader: DataLoader,
    basis_directions: torch.Tensor | dict[str, torch.Tensor],
    learning_rate: float = 0.1,
    n_epochs: int = 1,
    lamb: float = 0.1,
    verbose: bool = False,
    bigtom: bool = False,
    remote: bool = False,
) -> tuple[torch.Tensor, dict]:
    if remote:
        raise NotImplementedError("Training not tested for remote yet")

    if (
        not bigtom
        or exp_name == "answer_lookback-pointer"
        or exp_name == "answer_lookback-payload"
    ):
        intervention_positions = exp_to_intervention_positions[exp_name]
        patch_to_cache_map = {
            k: v
            for k, v in zip(
                intervention_positions["patch"], intervention_positions["cache"]
            )
        }
    else:
        # Intervention positions are dynamic for bigtom samples
        intervention_positions, patch_to_cache_map = None, None

    if not isinstance(basis_directions, dict):
        basis_indices = list(range(basis_directions.size(0)))
        mask = torch.ones(
            len(basis_indices), requires_grad=True, device="cuda", dtype=torch.bfloat16
        )
        basis_directions = basis_directions.to("cuda")
        optimizer = torch.optim.Adam([mask], lr=learning_rate)
    else:
        masks = {}
        for key in basis_directions:
            basis_indices = list(range(basis_directions[key].size(0)))
            masks[key] = torch.ones(
                len(basis_indices),
                requires_grad=True,
                device="cuda",
                dtype=torch.bfloat16,
            )
            basis_directions[key] = basis_directions[key].to("cuda")

        optimizer = torch.optim.Adam([{"params": masks.values()}], lr=learning_rate)

    # training loop
    for epoch in range(n_epochs):
        epoch_loss = 0

        for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            alt_prompts = batch["counterfactual_prompt"]
            org_prompts = batch["clean_prompt"]
            targets = (
                batch["target"] if "target" in batch else batch["counterfactual_target"]
            )
            if not bigtom:
                target_tokens = lm.tokenizer(targets, return_tensors="pt").input_ids[
                    :, -1
                ]
            else:
                target_tokens = lm.tokenizer(targets, return_tensors="pt").input_ids[
                    :, 1:
                ]
            batch_size = target_tokens.size(0)
            alt_acts, org_acts_state = defaultdict(dict), defaultdict(dict)

            if bigtom and not (
                exp_name == "answer_lookback-pointer"
                or exp_name == "answer_lookback-payload"
            ):
                intervention_positions, prompt_struct = (
                    get_bigtom_intervention_positions(
                        exp_name, lm, org_prompts, alt_prompts
                    )
                )
                patch_to_cache_map = {
                    k: v
                    for k, v in zip(
                        intervention_positions["patch"], intervention_positions["cache"]
                    )
                }

            if not isinstance(basis_directions, dict):
                masked_directions = basis_directions * mask.unsqueeze(-1)
                proj_matrix = torch.matmul(masked_directions.T, masked_directions).to(
                    lm.dtype
                )
            else:
                proj_matrix = {}
                for key in basis_directions:
                    masked_directions = basis_directions[key] * masks[key].unsqueeze(-1)
                    proj_matrix[key] = torch.matmul(
                        masked_directions.T, masked_directions
                    ).to(lm.dtype)

            with lm.trace() as tracer:
                with tracer.invoke(alt_prompts):
                    for t in intervention_positions["cache"]:
                        alt_acts[t] = lm.model.layers[layer_idx].output[0][:, t].clone()

                with tracer.invoke(org_prompts):
                    for t in intervention_positions["patch"]:
                        if not isinstance(basis_directions, dict):
                            proj = proj_matrix
                        else:
                            if t in query_object_indices:
                                proj = proj_matrix["query_obj_ordering_id"]
                            elif t in query_character_indices:
                                proj = proj_matrix["query_charac_ordering_id"]
                            else:
                                raise ValueError("Invalid projection type")

                        curr_output = lm.model.layers[layer_idx].output[0][:, t].clone()
                        alt_proj = torch.matmul(alt_acts[patch_to_cache_map[t]], proj)
                        org_proj = torch.matmul(curr_output, proj)
                        lm.model.layers[layer_idx].output[0][:, t] = (
                            curr_output - org_proj + alt_proj
                        )

                    logits = lm.lm_head.output[:, -1].save()

                    del alt_acts, org_proj
                    free_gpu_cache()

            target_logit = logits[torch.arange(batch_size), target_tokens]
            if not bigtom:
                task_loss = -torch.mean(target_logit)
            else:
                task_loss = -target_logit.sum()
            if not isinstance(basis_directions, dict):
                l1_loss = lamb * torch.norm(mask, p=1)
            else:
                l1_loss = 0
                for key in masks:
                    l1_loss += lamb * torch.norm(masks[key], p=1)
            loss = task_loss + l1_loss.to(task_loss.device)

            if verbose:
                if not isinstance(basis_directions, dict):
                    mask_data = mask.data.clone().clamp(0, 1).round()
                    cur_rank = mask_data.sum().item()
                else:
                    cur_rank = {}
                    for key in masks:
                        mask_data = masks[key].data.clone().clamp(0, 1).round()
                        cur_rank[key] = mask_data.sum().item()

                print(
                    f"Epoch: {epoch}, Batch: {batch_idx}, Rank: {cur_rank}, Loss: {loss.item()} | l_task: {task_loss.item()}, l1: {l1_loss.item()}"
                )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Clamp after optimizer step
            with torch.no_grad():
                if not isinstance(basis_directions, dict):
                    mask.data.clamp_(0, 1)
                else:
                    for key in masks:
                        masks[key].data.clamp_(0, 1)

            epoch_loss += loss.item()

            del logits, proj_matrix
            free_gpu_cache()

    # build the projection after training
    if not isinstance(basis_directions, dict):
        mask_data = mask.data.clone()
        mask_data.clamp_(0, 1)
        rounded = torch.round(mask_data)

        masked_directions = basis_directions * rounded.unsqueeze(-1)
        proj_matrix = torch.matmul(masked_directions.T, masked_directions).to(lm.dtype)

        metadata = {"mask": rounded.tolist(), "rank": rounded.sum().item()}

    else:
        rounded = {}
        proj_matrix = {}
        for key in masks:
            mask_data = masks[key].data.clone()
            mask_data.clamp_(0, 1)
            rounded[key] = torch.round(mask_data)

            masked_directions = basis_directions[key] * rounded[key].unsqueeze(-1)
            proj_matrix[key] = torch.matmul(masked_directions.T, masked_directions).to(
                lm.dtype
            )

        metadata = {
            "mask": {k: v.tolist() for k, v in rounded.items()},
            "rank": {k: v.sum().item() for k, v in rounded.items()},
        }

    return proj_matrix, metadata


def run_experiment(
    experiment_name: Literal[
        "answer_lookback-pointer",
        "answer_lookback-payload",
        "binding_lookback-pointer_object",
        "binding_lookback-pointer_character",
        "binding_lookback-address_and_payload",
        "visibility_lookback-source",
        "visibility_lookback-payload",
        "visibility_lookback-address_and_pointer",
        "binding_lookback-pointer_character_and_object",
    ],
    lm: LanguageModel,
    layers: list[int] = list(range(34, 60, 2)),
    train_size: int = 80,
    validation_size: int = 80,
    batch_size: int = 4,
    verbose: bool = False,
    learning_rate: float = 0.1,
    n_epochs: int = 1,
    lamb: float = 0.1,
    save_path: str = "results",
    save_outputs_on_val: bool = False,
    bigtom: bool = False,
    remote: bool = False,
    skip_subspace_patching: bool = True,
):
    print("#" * 30)
    print(f"Running experiment: {experiment_name}")
    print("#" * 30)

    lm_shorthand = lm.config._name_or_path.split("/")[-1]

    save_path = os.path.join(
        save_path,
        lm_shorthand,
        experiment_name.split("-")[0],
        experiment_name.split("-")[1],
    )
    os.makedirs(save_path, exist_ok=True)

    if not bigtom:
        train_dataloader, valid_dataloader = prepare_dataset(
            lm=lm,
            experiment_name=experiment_name,
            train_size=train_size,
            valid_size=validation_size,
            batch_size=batch_size,
            remote=remote,
        )
    else:
        train_dataloader, valid_dataloader = prepare_bigtom_dataset(
            experiment_name=experiment_name,
            train_size=train_size,
            valid_size=validation_size,
            batch_size=batch_size,
        )

    singular_vectors = None
    exclude_projections = []

    skip_low_rank_projection = (
        (experiment_name in exclude_projections)
        or ("405B" in lm.config._name_or_path)
        or skip_subspace_patching
    )

    if not skip_low_rank_projection:
        if (
            not experiment_name.startswith("vis")
            or experiment_name == "visibility_lookback-source"
        ):
            direction_type = exp_to_vec_type[experiment_name]
            path = "bigToM" if bigtom else "causalToM"
            if isinstance(direction_type, list):
                directions = {
                    d: load_basis_directions(
                        direction_type="singular_vecs", vector_type=d, path=path
                    )
                    for d in direction_type
                }
                singular_vectors = {
                    l: {d: directions[d][l] for d in direction_type} for l in layers
                }
            else:
                singular_vectors = load_basis_directions(
                    direction_type="singular_vecs",
                    vector_type=exp_to_vec_type[experiment_name],
                    path=path,
                )

    print("Running experiment === > ", experiment_name)
    progress_bar = tqdm(layers)
    for layer in progress_bar:
        progress_bar.set_description(f"Layer: {layer}")
        print(f"\n{'-' * 10} Layer: {layer} {'-' * 10}")

        layer_performance = {}

        # full state patching
        full_acc = validate(
            experiment_name,
            lm,
            layer,
            valid_dataloader,
            verbose=verbose,
            save_outputs=save_outputs_on_val,
            projection_type="full_rank",
            bigtom=bigtom,
            remote=remote,
        )
        print("-" * 30)
        print(f"Full state patching val: {full_acc}")
        print("-" * 30)
        layer_performance["full_rank"] = {"accuracy": full_acc, "rank": None}

        if singular_vectors is not None:
            # singular vector patching
            training_metadata = {
                "learning_rate": learning_rate,
                "n_epochs": n_epochs,
                "lamb": lamb,
            }
            print(f"Training singular vectors with {training_metadata}")
            singular_projection, singular_metadata = get_low_rank_projection(
                exp_name=experiment_name,
                lm=lm,
                layer_idx=layer,
                train_loader=train_dataloader,
                basis_directions=singular_vectors[layer],
                learning_rate=learning_rate,
                n_epochs=n_epochs,
                lamb=lamb,
                verbose=verbose,
                bigtom=bigtom,
                remote=remote,
            )

            print("validating ...")
            singular_acc = validate(
                exp_name=experiment_name,
                lm=lm,
                layer_idx=layer,
                validation_loader=valid_dataloader,
                projection=singular_projection,
                verbose=verbose,
                save_outputs=save_outputs_on_val,
                projection_type="singular_vector",
                remote=remote,
                bigtom=bigtom,
            )
            print("-" * 30)
            print(
                f"Singular vector patching val: {singular_acc} | Rank: {singular_metadata['rank']}"
            )
            print("-" * 30)

            layer_performance["singular_vector"] = {
                "accuracy": singular_acc,
                "rank": singular_metadata["rank"],
                "metadata": {
                    "training_args": training_metadata,
                    "mask": singular_metadata["mask"],
                },
            }

        # save results after each layer
        with open(os.path.join(save_path, f"{layer}.json"), "w") as f:
            json.dump(
                layer_performance,
                f,
                indent=4,
            )


experiment_layers = {
    "meta-llama/Meta-Llama-3-70B-Instruct": {
        "answer_lookback-pointer": list(range(30, 60, 1)),
        "answer_lookback-payload": list(range(50, 80, 1)),
        "binding_lookback-pointer_object": list(range(10, 40, 1)),
        "binding_lookback-pointer_character": list(range(10, 40, 1)),
        "binding_lookback-address_and_payload": list(range(25, 45, 1)),
        "visibility_lookback-source": list(range(0, 40, 1)),
        "visibility_lookback-payload": list(range(0, 60, 1)),
        "visibility_lookback-address_and_pointer": list(range(0, 60, 1)),
        "vis_2nd_to_1st_and_ques": list(range(0, 60, 1)),
        "pointer": list(range(10, 40, 1)),
    },
    "meta-llama/Meta-Llama-3.1-405B-Instruct": {
        "answer_lookback-payload": list(range(60, 80, 2)),
        "answer_lookback-pointer": list(range(40, 80, 1)),
        "binding_lookback-pointer_object": list(range(10, 50, 2)),
        "binding_lookback-pointer_character": list(range(10, 50, 2)),
        "binding_lookback-address_and_payload": list(range(20, 70, 2)),
        "visibility_lookback-source": list(range(0, 50, 2)),
        "visibility_lookback-payload": list(range(40, 80, 2)),
        "visibility_lookback-address_and_pointer": list(range(0, 80, 2)),
    },
}


def main(
    experiment: Literal[
        "answer_lookback-pointer",
        "answer_lookback-payload",
        "binding_lookback-pointer_object",
        "binding_lookback-pointer_character",
        "binding_lookback-address_and_payload",
        "binding_lookback-character_oi",
        "binding_lookback-object_oi",
        "visibility_lookback-source",
        "visibility_lookback-payload",
        "visibility_lookback-address_and_pointer",
        "pointer",
    ],
    model_key: str = "meta-llama/Meta-Llama-3-70B-Instruct",
    layers: list[int] = None,
    train_size: int = 80,
    validation_size: int = 80,
    batch_size: int = 1,
    verbose: bool = False,
    learning_rate: float = 0.1,  # 0.005 for BigToM
    n_epochs: int = 1,
    lamb: float = None,
    save_path: str = "experiments/causalToM_novis/results",
    save_outputs: bool = False,
    bigtom: bool = False,
    remote: bool = False,
    skip_subspace_patching: bool = True,
):
    """
    Run single layer patching experiments.

    Args:
        experiment: Type of experiment to run
        model_key: Model to use
        layers: List of layers to run experiments on
        train_size: Number of training samples
        validation_size: Number of validation samples
        batch_size: Batch size for training and validation
        verbose: Whether to print verbose output
        learning_rate: Learning rate for training
        n_epochs: Number of training epochs
        lamb: L1 regularization parameter
        save_path: Path to save results
        save_outputs: Whether to save outputs
        bigtom: Whether to run experiments on BigToM
        verbose: Whether to print verbose output
        remote: Whether to run experiments remotely
    """
    if lamb is None:
        if experiment == "visibility_lookback-source":
            lamb = 0.03
        else:
            lamb = 0.1

    if bigtom:
        # save_path = "experiments/bigToM/results"
        if experiment not in [
            "answer_lookback-pointer",
            "answer_lookback-payload",
            "binding_lookback-pointer_character",
            "visibility_lookback-source",
            "visibility_lookback-payload",
        ]:
            raise ValueError(f"Experiment {experiment} is not supported for BigToM")

    print(f"Running experiment: {experiment}")
    print(f"Model: {model_key}")
    print(f"Layers: {layers}")
    print(f"Train size: {train_size}")
    print(f"Validation size: {validation_size}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {n_epochs}")
    print(f"Lambda: {lamb}")
    print(f"Save path: {save_path}")
    print(f"Save outputs: {save_outputs}")
    print(f"BigToM: {bigtom}")
    print(f"Remote: {remote}")
    print(f"Verbose: {verbose}")
    print(f"Skip subspace patching: {skip_subspace_patching}")
    if remote:
        lm = LanguageModel("meta-llama/Meta-Llama-3.1-405B-Instruct")
    else:
        lm = LanguageModel(
            "meta-llama/Meta-Llama-3-70B-Instruct",
            device_map="auto",
            dtype=torch.float16,
            dispatch=True,
        )

    if layers is None:
        n_layer = lm.config.num_hidden_layers
        layers = sorted(
            list(
                set(
                    list(range(0, n_layer, 10))
                    + experiment_layers[model_key].get(experiment, [])
                    + [n_layer - 1]
                )
            )
        )

    set_seed(10)
    run_experiment(
        experiment_name=experiment,
        lm=lm,
        layers=layers,
        train_size=train_size,
        validation_size=validation_size,
        batch_size=batch_size,
        verbose=verbose,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        lamb=lamb,
        save_path=save_path,
        save_outputs_on_val=save_outputs and not remote,
        bigtom=bigtom,
        remote=remote,
        skip_subspace_patching=skip_subspace_patching,
    )


if __name__ == "__main__":
    fire.Fire(main)
