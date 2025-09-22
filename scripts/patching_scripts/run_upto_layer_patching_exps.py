import json
import os
import sys
from collections import defaultdict
from typing import Literal

import fire
import torch
from nnsight import LanguageModel
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from run_patching_exp_utils import (
    Accuracy,
    exp_to_vec_type,
    free_gpu_cache,
    load_basis_directions,
    prepare_dataset,
    set_seed,
)

from src import env_utils

os.environ["NDIF_KEY"] = env_utils.load_env_var("NDIF_KEY")
os.environ["HF_TOKEN"] = env_utils.load_env_var("HF_WRITE")

charac_indices = [131, 133, 146, 147, 158, 159]
reversed_charac_indices = [133, 131, 158, 159, 146, 147]

object_indices = [150, 151, 162, 163]
reversed_object_indices = [162, 163, 150, 151]

state_indices = [155, 156, 167, 168]
query_character_indices = [-8, -7]
query_object_indices = [-5, -4]

retain_full_indices = {
    "binding_lookback-object_oi": state_indices,
    "binding_lookback-character_oi": object_indices + state_indices,
    "binding_lookback-source_1": state_indices,
    "binding_lookback-source_2": [],
}
retain_upto_indices = {
    "binding_lookback-object_oi": query_object_indices,
    "binding_lookback-character_oi": query_character_indices,
    "binding_lookback-source_1": [],
    "binding_lookback-source_2": [],
}
patch_indices = {
    "binding_lookback-object_oi": reversed_object_indices,
    "binding_lookback-character_oi": reversed_charac_indices,
    "binding_lookback-source_1": reversed_charac_indices + reversed_object_indices,
    "binding_lookback-source_2": reversed_charac_indices + reversed_object_indices,
}


def is_mixed_projections(experiment_name):
    return experiment_name in ["source_1", "source_2"]


@torch.inference_mode()
def validate(
    exp_name: Literal[
        "binding_lookback-object_oi",
        "binding_lookback-character_oi",
        "binding_lookback-source_1",
        "binding_lookback-source_2",
    ],
    lm: LanguageModel,
    layer_idx: int,
    validation_loader: DataLoader,
    projections: (
        dict[str, torch.Tensor] | dict[str, dict[int, torch.Tensor]] | None
    ) = None,
    verbose: bool = False,
    restore_state: bool = True,
    save_outputs: bool = True,
    projection_type: Literal["full_rank", "singular_vector"] = "full_rank",
    remote: bool = False,
) -> float:
    save_outputs = save_outputs and not remote

    cuda_last_index = torch.cuda.device_count() - 1
    device_p = f"cuda:{cuda_last_index}"

    if save_outputs:
        exp_dir = exp_name
        if restore_state is False:
            exp_dir += "_wo_state"
        save_path = os.path.join(
            "results",
            lm.config._name_or_path.split("/")[-1],
            "lm_pred_on_val_set",
            exp_dir,
            f"layer_{layer_idx}",
            projection_type,
        )
        os.makedirs(save_path, exist_ok=True)
        valid_data = validation_loader.dataset

    interesting_positions = (
        charac_indices
        + object_indices
        + state_indices
        + query_character_indices
        + query_object_indices
    )

    patch_to_cache_map = {
        k: v
        for k, v in zip(
            charac_indices + object_indices,
            reversed_charac_indices + reversed_object_indices,
        )
    }

    correct, total = 0, 0
    for batch_idx, batch in tqdm(
        enumerate(validation_loader), total=len(validation_loader)
    ):
        alt_prompts = batch["corrupt_prompt"]
        org_prompts = batch["clean_prompt"]
        targets = batch["target"] if "target" in batch else batch["corrupt_target"]
        target_tokens = lm.tokenizer(targets, return_tensors="pt").input_ids[:, -1]
        batch_size = target_tokens.size(0)
        alt_acts, org_acts = defaultdict(dict), defaultdict(dict)

        def nnsight_request(return_logits: bool = False):
            with lm.trace(remote=remote) as tracer:
                with tracer.invoke(alt_prompts):
                    for layer in range(0, layer_idx + 1):
                        for t in interesting_positions:
                            alt_acts[layer][t] = (
                                lm.model.layers[layer].output[0][:, t].clone()
                            )

                with tracer.invoke(org_prompts):
                    for layer in range(0, lm.config.num_hidden_layers):
                        for t in interesting_positions:
                            org_acts[layer][t] = (
                                lm.model.layers[layer].output[0][:, t].clone()
                            )

                with tracer.invoke(org_prompts):
                    for layer in range(0, layer_idx + 1):
                        if projections is not None:
                            if is_mixed_projections(exp_name):
                                projection = {
                                    k: v[layer].to(device=device_p)
                                    for k, v in projections.items()
                                }
                            else:
                                projection = projections[layer].to(device=device_p)
                        if exp_name == "object_position":
                            for t in reversed_charac_indices:
                                lm.model.layers[layer].output[0][:, t] = alt_acts[
                                    layer
                                ][patch_to_cache_map[t]]

                        for t in patch_indices[exp_name]:
                            curr_output = lm.model.layers[layer].output[0][:, t].clone()
                            if projections is not None:
                                if is_mixed_projections(exp_name):
                                    if t in charac_indices:
                                        proj = projection["character_ordering_id"]
                                    elif t in object_indices:
                                        proj = projection["object_ordering_id"]
                                    else:
                                        raise ValueError("Invalid patch index")
                                else:
                                    proj = projection

                                alt_proj = torch.matmul(
                                    alt_acts[layer][patch_to_cache_map[t]], proj
                                )
                                org_proj = torch.matmul(curr_output, proj)
                                patch = curr_output - org_proj + alt_proj
                            else:
                                patch = alt_acts[layer][patch_to_cache_map[t]]

                            lm.model.layers[layer].output[0][:, t] = patch

                        for t in retain_upto_indices[exp_name]:
                            lm.model.layers[layer].output[0][:, t] = org_acts[layer][t]

                    for layer in range(lm.config.num_hidden_layers):
                        for t in retain_full_indices[exp_name]:
                            if not restore_state:
                                if t in state_indices:
                                    continue
                            lm.model.layers[layer].output[0][:, t] = org_acts[layer][t]

                    logits = lm.lm_head.output[:, -1]
                    logits = logits.save() if return_logits else logits
                    pred = torch.argmax(logits, dim=-1).save()

            return logits, pred

        logits, pred = nnsight_request(return_logits=not remote)

        for i in range(batch_size):
            total += 1
            pred_token = lm.tokenizer.decode(pred[i])
            is_correct = pred_token.lower().strip() == targets[i].lower().strip()
            if verbose:
                print(
                    f"Predicted: {pred_token.lower().strip()}, Target: {targets[i].lower().strip()}"
                )
            correct += int(is_correct)

            if save_outputs:
                sample = valid_data[batch_idx * batch_size + i]
                with open(
                    os.path.join(save_path, f"sample_{total}.json"),
                    "w",
                ) as f:
                    json.dump(
                        {
                            "sample": sample,
                            "pred": {
                                "token": pred_token,
                                "token_id": pred[i].item(),
                                "logit": logits[i, pred[i]].item(),
                            },
                            "logit_distribution": logits[i].tolist(),
                            "is_correct": is_correct,
                        },
                        f,
                        indent=4,
                    )

        del (alt_acts, alt_prompts, org_prompts, targets, target_tokens, logits, pred)

        # if projections is not None:
        #     try:
        #         del projection
        #     except:
        #         pass
        free_gpu_cache()

    return correct / total


def get_low_rank_projection(
    exp_name: Literal[
        "binding_lookback-object_oi",
        "binding_lookback-character_oi",
        "binding_lookback-source_1",
        "binding_lookback-source_2",
    ],
    lm: LanguageModel,
    layer_idx: int,
    train_loader: DataLoader,
    all_basis_directions: dict[int, torch.Tensor] | dict[str, dict[int, torch.Tensor]],
    learning_rate: float = 0.1,
    n_epochs: int = 1,
    lamb: float = 0.01,
    verbose: bool = False,
    restore_state: bool = True,
    remote: bool = False,
) -> tuple[torch.Tensor, dict]:
    if remote is True:
        raise NotImplementedError("Training not tested for remote yet")

    cuda_last_index = torch.cuda.device_count() - 1
    device_p = f"cuda:{cuda_last_index}"

    interesting_positions = (
        charac_indices
        + object_indices
        + state_indices
        + query_character_indices
        + query_object_indices
    )

    patch_to_cache_map = {
        k: v
        for k, v in zip(
            charac_indices + object_indices,
            reversed_charac_indices + reversed_object_indices,
        )
    }

    if not is_mixed_projections(exp_name):
        basis_indices = list(range(all_basis_directions[0].size(0)))
        mask = torch.ones(
            (layer_idx + 1, len(basis_indices)),
            requires_grad=True,
            device=device_p,
            dtype=torch.bfloat16,
        )
        optimizer = torch.optim.Adam([mask], lr=learning_rate)
    else:
        mask = {}
        for key in all_basis_directions:
            basis_indices = list(range(all_basis_directions[key][0].size(0)))
            mask[key] = torch.ones(
                (layer_idx + 1, len(basis_indices)),
                requires_grad=True,
                device=device_p,
                dtype=torch.bfloat16,
            )
        optimizer = torch.optim.Adam([{"params": mask.values()}], lr=learning_rate)

    for epoch in range(n_epochs):
        epoch_loss = 0

        for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            alt_prompts = batch["corrupt_prompt"]
            org_prompts = batch["clean_prompt"]
            targets = batch["target"] if "target" in batch else batch["corrupt_target"]
            target_tokens = lm.tokenizer(targets, return_tensors="pt").input_ids[:, -1]
            batch_size = target_tokens.size(0)
            alt_acts, org_acts = defaultdict(dict), defaultdict(dict)

            with lm.trace() as tracer:
                with tracer.invoke(alt_prompts):
                    for layer in range(0, layer_idx + 1):
                        for t in interesting_positions:
                            alt_acts[layer][t] = (
                                lm.model.layers[layer].output[0][:, t].clone()
                            )

                with tracer.invoke(org_prompts):
                    for layer in range(0, lm.config.num_hidden_layers):
                        for t in interesting_positions:
                            org_acts[layer][t] = (
                                lm.model.layers[layer].output[0][:, t].clone()
                            )

                with tracer.invoke(org_prompts):
                    for layer in range(0, layer_idx + 1):
                        if exp_name == "object_position":
                            for t in reversed_charac_indices:
                                lm.model.layers[layer].output[0][:, t] = alt_acts[
                                    layer
                                ][patch_to_cache_map[t]]

                        if not is_mixed_projections(exp_name):
                            masked_directions = all_basis_directions[layer].to(
                                device=device_p
                            ) * mask[layer].unsqueeze(-1)
                            projection = torch.matmul(
                                masked_directions.T, masked_directions
                            ).half()
                        else:
                            projection = {}
                            for key in all_basis_directions:
                                masked_directions = all_basis_directions[key][layer].to(
                                    device=device_p
                                ) * mask[key][layer].unsqueeze(-1)
                                projection[key] = torch.matmul(
                                    masked_directions.T, masked_directions
                                ).half()

                        for t in patch_indices[exp_name]:
                            if is_mixed_projections(exp_name):
                                if t in charac_indices:
                                    proj = projection["character_ordering_id"]
                                elif t in object_indices:
                                    proj = projection["object_ordering_id"]
                                else:
                                    raise ValueError("Invalid patch index")
                            else:
                                proj = projection

                            curr_output = lm.model.layers[layer].output[0][:, t].clone()
                            alt_proj = torch.matmul(
                                alt_acts[layer][patch_to_cache_map[t]], proj
                            )
                            org_proj = torch.matmul(curr_output, proj)
                            patch = curr_output - org_proj + alt_proj
                            lm.model.layers[layer].output[0][:, t] = patch

                        for t in retain_upto_indices[exp_name]:
                            lm.model.layers[layer].output[0][:, t] = org_acts[layer][t]

                    for layer in range(lm.config.num_hidden_layers):
                        for t in retain_full_indices[exp_name]:
                            if not restore_state:
                                if t in state_indices:
                                    continue
                            lm.model.layers[layer].output[0][:, t] = org_acts[layer][t]

                    logits = lm.lm_head.output[:, -1].save()

            target_logit = logits[torch.arange(batch_size), target_tokens]
            task_loss = -torch.mean(target_logit)
            if not is_mixed_projections(exp_name):
                l1_loss = lamb * torch.norm(mask, p=1)
            else:
                l1_loss = 0
                for key in mask:
                    l1_loss += lamb * torch.norm(mask[key], p=1)
            loss = task_loss + l1_loss.to(task_loss.device)

            if verbose:
                print(
                    f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()} |>> l_task: {task_loss.item()}, l1: {l1_loss.item()}"
                )
                if not is_mixed_projections(exp_name):
                    mask_data = mask.data.clone().clamp(0, 1).round()
                    print(
                        {
                            l: int(mask_data[l].sum().item())
                            for l in range(mask_data.size(0))
                        }
                    )
                else:
                    for key in mask:
                        print(key)
                        mask_data = mask[key].data.clone().clamp(0, 1).round()
                        print(
                            {
                                l: int(mask_data[l].sum().item())
                                for l in range(mask_data.size(0))
                            }
                        )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Clamp after optimizer step
            with torch.no_grad():
                if not is_mixed_projections(exp_name):
                    mask.data.clamp_(0, 1)
                else:
                    for key in mask:
                        mask[key].data.clamp_(0, 1)

            epoch_loss += loss.item()

            del (
                alt_acts,
                alt_prompts,
                org_prompts,
                targets,
                target_tokens,
                logits,
                projection,
            )
            free_gpu_cache()

    # build the projections after training
    if not is_mixed_projections(exp_name):
        projections = {}
        meta_data = {}
        for layer in range(layer_idx + 1):
            mask_data = mask[layer].data.clone().clamp(0, 1).round()
            masked_directions = all_basis_directions[layer].to(
                device=device_p
            ) * mask_data.unsqueeze(-1)
            projections[layer] = (
                torch.matmul(masked_directions.T, masked_directions).half().cpu()
            )
            # projections[layer] = projection.cpu()
            meta_data[layer] = {
                "mask": mask_data.tolist(),
                "rank": mask_data.sum().item(),
            }
    else:
        projections = {}
        meta_data = {}
        for key in mask:
            projections[key] = {}
            meta_data[key] = {}
            for layer in range(layer_idx + 1):
                mask_data = mask[key][layer].data.clone().clamp(0, 1).round()
                masked_directions = all_basis_directions[key][layer].to(
                    device=device_p
                ) * mask_data.unsqueeze(-1)
                projections[key][layer] = (
                    torch.matmul(masked_directions.T, masked_directions).half().cpu()
                )
                meta_data[key][layer] = {
                    "mask": mask_data.tolist(),
                    "rank": mask_data.sum().item(),
                }

    for param in lm.model.parameters():
        param.grad = None
    free_gpu_cache()

    return projections, meta_data


def run_experiment(
    experiment_name: Literal[
        "binding_lookback-object_oi",
        "binding_lookback-character_oi",
        "binding_lookback-source_1",
        "binding_lookback-source_2",
    ],
    lm: LanguageModel,
    layers: list[int] = list(range(34, 60, 2)),
    train_size: int = 80,
    validation_size: int = 80,
    batch_size: int = 4,
    verbose: bool = False,
    learning_rate: float = 0.1,
    n_epochs: int = 1,
    lamb: float = 0.01,
    save_path: str = "results/",
    restore_state: bool = True,
    save_outputs_on_val: bool = False,
    remote: bool = False,
):
    print("#" * 30)
    print(f"Running experiment: {experiment_name}")
    print("#" * 30)

    exp_subdir = experiment_name
    if not restore_state:
        exp_subdir += "_wo_state"
    lm_shorthand = lm.config._name_or_path.split("/")[-1]
    if remote:
        lm_shorthand = f"{lm_shorthand}-8bit-remote"
    save_path = os.path.join(save_path, lm_shorthand, experiment_name)
    os.makedirs(save_path, exist_ok=True)

    train_dataloader, valid_dataloader = prepare_dataset(
        lm=lm,
        experiment_name=experiment_name,
        train_size=train_size,
        valid_size=validation_size,
        batch_size=batch_size,
        remote=remote,
    )

    singular_vectors = None
    exclude_projections = []

    skip_low_rank_projection = (experiment_name in exclude_projections) or (
        "405B" in lm.config._name_or_path
    )

    if not skip_low_rank_projection:
        direction_type = exp_to_vec_type[experiment_name]
        if isinstance(direction_type, list):
            singular_vectors = {
                d: load_basis_directions("singular_vecs", d) for d in direction_type
            }
        else:
            singular_vectors = load_basis_directions("singular_vecs", direction_type)

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
            projection_type="full_rank",
            save_outputs=save_outputs_on_val,
            restore_state=restore_state,
            remote=remote,
        )
        print("-" * 30)
        print(f"Full state patching val: {full_acc}")
        print("-" * 30)
        layer_performance["full_rank"] = Accuracy(
            accuracy=full_acc, rank=None
        ).to_dict()

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
                all_basis_directions=singular_vectors,
                learning_rate=learning_rate,
                n_epochs=n_epochs,
                lamb=lamb,
                verbose=verbose,
                remote=remote,
            )

            print("validating ...")
            singular_acc = validate(
                exp_name=experiment_name,
                lm=lm,
                layer_idx=layer,
                validation_loader=valid_dataloader,
                projections=singular_projection,
                verbose=verbose,
                save_outputs=save_outputs_on_val,
                restore_state=restore_state,
                projection_type="singular_vector",
                remote=remote,
            )
            print("-" * 30)
            print(f"Singular vector patching val: {singular_acc}")
            print("-" * 30)

            layer_performance["singular_vector"] = Accuracy(
                accuracy=singular_acc,
                metadata={
                    "training_args": training_metadata,
                    "metadata": singular_metadata,
                },
            ).to_dict()

        # save results after each layer
        with open(os.path.join(save_path, f"{layer}.json"), "w") as f:
            json.dump(
                layer_performance,
                f,
                indent=4,
            )


experiment_layers = {
    "meta-llama/Meta-Llama-3-70B-Instruct": {
        "binding_lookback-object_oi": list(range(10, 40, 1)),
        "binding_lookback-character_oi": list(range(10, 40, 1)),
        "binding_lookback-source_1": list(range(10, 40, 1)),
        "binding_lookback-source_2": list(range(10, 40, 1)),
    },
    "meta-llama/Meta-Llama-3.1-405B-Instruct": {
        "binding_lookback-source_1": list(range(20, 80, 2)),
        "binding_lookback-source_2": list(range(20, 80, 2)),
        "binding_lookback-object_oi": list(range(20, 80, 2)),
        "binding_lookback-character_oi": list(range(20, 80, 2)),
    },
}


def main(
    experiment: Literal[
        "binding_lookback-object_oi",
        "binding_lookback-character_oi",
        "binding_lookback-source_1",
        "binding_lookback-source_2",
    ],
    model_key: str = "meta-llama/Meta-Llama-3-70B-Instruct",
    layers: list[int] = None,
    train_size: int = 80,
    validation_size: int = 80,
    batch_size: int = 1,
    verbose: bool = False,
    learning_rate: float = 0.1,
    n_epochs: int = 1,
    lamb: float = 0.01,
    save_path: str = "experiments/causalToM_novis/results",
    save_outputs: bool = False,
    restore_state: bool = True,
    no_restore_state: bool = False,
    remote: bool = False,
):
    if remote:
        lm = LanguageModel("meta-llama/Meta-Llama-3.1-405B-Instruct")
    else:
        lm = LanguageModel(
            "meta-llama/Meta-Llama-3-70B-Instruct",
            device_map="auto",
            torch_dtype=torch.float16,
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
        restore_state=restore_state,
        save_outputs_on_val=save_outputs,
        remote=remote,
    )


if __name__ == "__main__":
    fire.Fire(main)
