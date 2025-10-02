import ctypes
import gc
import inspect
import json
import os
import random
import sys
import threading
from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
import torch
from dataclasses_json import DataClassJsonMixin
from nnsight import LanguageModel
from torch.utils.data import DataLoader
from tqdm import tqdm

curr_file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(curr_file_dir)))
from notebooks.bigToM.utils import (
    get_answer_lookback_payload_exps,
    get_answer_lookback_pointer_exps,
    get_binding_lookback_pointer_exps,
    get_prompt_token_len,
    get_ques_start_token_idx,
    get_visibility_lookback_exps,
    get_visitibility_sent_start_idx,
)
from notebooks.causalToM_novis.utils import (
    get_answer_lookback_payload,
    get_character_oi_exps,
    get_object_oi_exps,
    get_query_charac_oi,
    get_query_object_oi,
    get_reversed_sent_diff_state_counterfacts,
    get_reversed_sentence_counterfacts,
)
from notebooks.causalToM_vis.utils import get_visibility_lookback_data
from src import global_utils


def free_gpu_cache():
    gc.collect()
    torch.cuda.empty_cache()


def _async_raise(tid, exctype):
    """Raises an exception in the threads with id tid"""
    if not inspect.isclass(exctype):
        raise TypeError("Only types can be raised (not instances)")
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_long(tid), ctypes.py_object(exctype)
    )
    if res == 0:
        raise ValueError("Invalid thread id")
    elif res != 1:
        # If it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


class ThreadWithId(threading.Thread):
    def get_id(self):
        # Returns the thread ID
        if hasattr(self, "_thread_id"):
            return self._thread_id
        for tid, thread in threading._active.items():
            if thread is self:
                self._thread_id = tid
                return tid
        return None

    def terminate(self):
        # Terminates the thread by raising an exception in it
        thread_id = self.get_id()
        if thread_id is not None:
            _async_raise(thread_id, SystemExit)


def send_request_to_ndif(nnsight_request: Callable, timeout=72000, n_try=5) -> Any:
    """
    Execute a remote request with timeout handling and retries.
    This implementation forcibly terminates the worker thread if it exceeds the timeout.

    Args:
        nnsight_request: Callable function that executes a request on a remote server
        timeout: Maximum time in seconds to wait for the request to complete (default value: 5 minutes)
        n_try: Maximum number of retry attempts

    Returns:
        The result of the successful nnsight_request call

    Raises:
        RuntimeError: If the request fails after n_try attempts
    """
    for attempt in range(1, n_try + 1):
        result_container = []
        exception_container = []

        def worker_function():
            try:
                result = nnsight_request()
                result_container.append(result)
            except Exception as e:
                exception_container.append(e)

        worker_thread = ThreadWithId(target=worker_function)
        worker_thread.daemon = (
            True  # Allow the thread to be terminated when main thread exits
        )
        worker_thread.start()

        # Wait for the thread to complete or timeout
        worker_thread.join(timeout=timeout)

        # Check if the thread is still alive after the timeout
        if worker_thread.is_alive():
            print(f"Request timed out (attempt {attempt}/{n_try})")
            # Forcibly terminate the thread
            worker_thread.terminate()
            if attempt == n_try:
                raise RuntimeError(
                    f"Request failed after {n_try} attempts (timeout: {timeout}s)"
                )
        else:
            # Thread completed within the timeout
            if exception_container:
                # An exception occurred in the worker thread
                print(
                    f"Request failed with error: {str(exception_container[0])} (attempt {attempt}/{n_try})"
                )
                if attempt == n_try:
                    raise RuntimeError(
                        f"Request failed after {n_try} attempts: {str(exception_container[0])}"
                    )
            else:
                # Thread completed successfully
                return result_container[0]

    raise RuntimeError(
        f"END OF FUNCTION (shouldn't reach): Request failed after {n_try} attempts"
    )


exp_to_ds_func_map = {
    "answer_lookback-pointer": get_reversed_sent_diff_state_counterfacts,
    "answer_lookback-payload": get_answer_lookback_payload,
    "binding_lookback-pointer_object": get_query_object_oi,
    "binding_lookback-pointer_character": get_query_charac_oi,
    "binding_lookback-object_oi": get_object_oi_exps,
    "binding_lookback-character_oi": get_character_oi_exps,
    "binding_lookback-address_and_payload": get_reversed_sentence_counterfacts,
    "visibility_lookback-payload": get_visibility_lookback_data,
    "visibility_lookback-source": get_visibility_lookback_data,
    "visibility_lookback-address_and_payload": get_visibility_lookback_data,
    "vis_2nd_to_1st_and_ques": get_visibility_lookback_data,
    "binding_lookback-source_1": get_reversed_sent_diff_state_counterfacts,
    "binding_lookback-source_2": get_reversed_sent_diff_state_counterfacts,
    "binding_lookback-pointer_charac_and_object": get_reversed_sent_diff_state_counterfacts,
}

bigtom_exp_to_ds_func_map = {
    "answer_lookback-pointer": get_answer_lookback_pointer_exps,
    "answer_lookback-payload": get_answer_lookback_payload_exps,
    "binding_lookback-pointer_character": get_binding_lookback_pointer_exps,
    "visibility_lookback-source": get_visibility_lookback_exps,
    "visibility_lookback-payload": get_visibility_lookback_exps,
    "visibility_lookback-address_and_payload": get_visibility_lookback_exps,
}

exp_to_vec_type = {
    "answer_lookback-pointer": "last_token",
    "answer_lookback-payload": "last_token",
    "binding_lookback-pointer_object": "query_obj_tokens",
    "binding_lookback-pointer_character": "query_charac_tokens",
    "binding_lookback-address_and_payload": "state_tokens",
    "binding_lookback-object_oi": "object_tokens",
    "binding_lookback-character_oi": "character_tokens",
    "visibility_lookback-payload": None,
    "visibility_lookback-source": "second_visibility_sent",
    "visibility_lookback-address_and_payload": None,
    "vis_2nd_to_1st_and_ques": None,
    "binding_lookback-source_1": ["object_tokens", "character_tokens"],
    "binding_lookback-source_2": ["object_tokens", "character_tokens"],
    "binding_lookback-pointer_charac_and_object": [
        "query_obj_tokens",
        "query_charac_tokens",
    ],
}

bigtom_exp_to_intervention_positions = {
    "binding_lookback-pointer_character": {
        "cache": ["alt_ques_start_idx", "alt_ques_start_idx"],
        "patch": ["org_ques_start_idx", "org_ques_start_idx"],
        "cache_offset": [3, 5],
        "patch_offset": [3, 5],
    },
    "visibility_lookback-source": {
        "cache": ["alt_vis_sent_start_idx", "alt_ques_start_idx"],
        "patch": ["org_vis_sent_start_idx", "org_ques_start_idx"],
        "cache_offset": [0, 0],
        "patch_offset": [0, 0],
    },
    "visibility_lookback-payload": {
        "cache": ["alt_ques_start_idx", "alt_prompt_len"],
        "patch": ["org_ques_start_idx", "org_prompt_len"],
        "cache_offset": [0, 0],
        "patch_offset": [0, 0],
    },
    "visibility_lookback-address_and_payload": {
        "cache": ["alt_ques_start_idx", "alt_prompt_len"],
        "patch": ["org_ques_start_idx", "org_prompt_len"],
        "cache_offset": [0, 0],
        "patch_offset": [0, 0],
    },
}

exp_to_intervention_positions = {
    "answer_lookback-pointer": {
        "cache": [-1],
        "patch": [-1],
    },
    "answer_lookback-payload": {
        "cache": [-1],
        "patch": [-1],
    },
    "binding_lookback-pointer_object": {
        "cache": [-5, -4],
        "patch": [-5, -4],
    },
    "binding_lookback-pointer_character": {
        "cache": [-8, -7],
        "patch": [-8, -7],
    },
    "binding_lookback-address_and_payload": {
        "cache": [155, 156, 167, 168],
        "patch": [167, 168, 155, 156],
    },
    "visibility_lookback-payload": {
        "cache": [i for i in range(176, 183)],
        "patch": [i for i in range(176, 183)],
    },
    "visibility_lookback-source": {
        "cache": [i for i in range(183, 195)],
        "patch": [i for i in range(183, 195)],
    },
    "visibility_lookback-address_and_payload": {
        "cache": [i for i in range(176, 195)],  # second visibility + question
        "patch": [i for i in range(176, 195)],  # second visibility + question
    },
    "vis_2nd_to_1st_and_ques": {
        "cache": [i for i in range(176, 183)]
        + [i for i in range(183, 195)],  # second visibility + question
        "patch": [i for i in range(169, 176)]
        + [i for i in range(183, 195)],  # first visibility + question
    },
    "binding_lookback-pointer_charac_and_object": {
        "cache": [-5, -4]
        + [
            -8,
            -7,
        ],  # binding_lookback-pointer_object + binding_lookback-pointer_character
        "patch": [-5, -4] + [-8, -7],
    },
}


def set_seed(seed: int) -> None:
    """Globally set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_bigtom_intervention_positions(
    exp_name: str, lm: LanguageModel, org_prompts: str, alt_prompts: str
) -> tuple[dict, dict]:
    prompt_struct = {
        "org_ques_start_idx": get_ques_start_token_idx(lm.tokenizer, org_prompts),
        "alt_ques_start_idx": get_ques_start_token_idx(lm.tokenizer, alt_prompts),
        "org_vis_sent_start_idx": get_visitibility_sent_start_idx(
            lm.tokenizer, org_prompts
        ),
        "alt_vis_sent_start_idx": get_visitibility_sent_start_idx(
            lm.tokenizer, alt_prompts
        ),
        "org_prompt_len": get_prompt_token_len(lm.tokenizer, org_prompts),
        "alt_prompt_len": get_prompt_token_len(lm.tokenizer, alt_prompts),
    }

    meta_intervention_positions = bigtom_exp_to_intervention_positions[exp_name]
    cache_start = (
        prompt_struct[meta_intervention_positions["cache"][0]]
        + meta_intervention_positions["cache_offset"][0]
    )
    cache_end = (
        prompt_struct[meta_intervention_positions["cache"][1]]
        + meta_intervention_positions["cache_offset"][1]
    )
    patch_start = (
        prompt_struct[meta_intervention_positions["patch"][0]]
        + meta_intervention_positions["patch_offset"][0]
    )
    patch_end = (
        prompt_struct[meta_intervention_positions["patch"][1]]
        + meta_intervention_positions["patch_offset"][1]
    )
    intervention_positions = {
        "cache": [i for i in range(cache_start, cache_end)],
        "patch": [i for i in range(patch_start, patch_end)],
    }
    return intervention_positions, prompt_struct


def load_basis_directions(
    direction_type: Literal["singular_vecs"],
    vector_type: (
        Literal[
            "last_token",
            "state_tokens",
            "object_tokens",
            "character_tokens",
            "query_charac_tokens",
            "query_obj_tokens",
            "second_visibility_sent",
        ]
        | None
    ),
    path: str = "CausalToM",
    prefix: str = "",
) -> dict[int, torch.Tensor]:
    if vector_type is None:
        print("WARNING: No direction type specified")
        return None
    path = os.path.join(prefix, "svd", path, vector_type, direction_type)
    directions = {}
    for file in os.listdir(path):
        idx = int(file.split(".")[0])
        directions[idx] = torch.load(os.path.join(path, file)).cpu()

    print(
        f"""Loaded {len(directions)} {direction_type} directions for {vector_type} | shape: {directions[0].shape}"""
    )

    return directions


@torch.inference_mode()
def check_lm_on_sample(lm, sample, remote=False):
    lm.tokenizer.padding_side = "left"
    # lm.model.eval()

    prompts = [sample["counterfactual_prompt"], sample["clean_prompt"]]
    answers = [
        sample["counterfactual_ans"]
        if "counterfactual_ans" in sample
        else sample["counterfactual_target"],
        sample["clean_ans"] if "clean_ans" in sample else sample["clean_target"],
    ]
    inputs = lm.tokenizer(prompts, return_tensors="pt", padding=True)
    if not remote:
        inputs = inputs.to(lm.device)

    def nnsight_request():
        # print(inputs)
        with lm.trace(inputs, remote=remote):
            logits = lm.lm_head.output[:, -1]
            predicted = torch.argmax(logits, dim=-1).save()
        return predicted

    predicted = (
        nnsight_request()
        if not remote
        else send_request_to_ndif(nnsight_request, timeout=100, n_try=5)
    )

    predicted = predicted.value if remote else predicted
    predicted = [lm.tokenizer.decode(pred) for pred in predicted]

    ok = True
    for ans, pred in zip(answers, predicted):
        # print(f'Predicted: "{pred}", Target: "{ans}"')
        if pred.lower().strip() != ans.lower().strip():
            ok = False

    return ok


@torch.inference_mode()
def filter_dataset_on_lm(lm, dataset, size=None, remote=False):
    filtered_dataset = []
    size = size if size is not None else len(dataset)
    correct, total = 0, 0
    progress_bar = tqdm(dataset)
    for sample in progress_bar:
        total += 1
        if check_lm_on_sample(lm, sample, remote=remote):
            filtered_dataset.append(sample)
            correct += 1
        if len(filtered_dataset) == size:
            break
        progress_bar.set_description(
            f"LM accuracy: {correct / total:.2f}({correct}/{total})"
        )
    return filtered_dataset, correct / total


def prepare_dataset(
    experiment_name: str,
    train_size: int = 80,
    valid_size: int = 80,
    batch_size: int = 4,
    lm: LanguageModel | None = None,
    remote: bool = False,
):
    all_characters = json.load(
        open(
            os.path.join(
                global_utils.DEFAULT_DATA_DIR, "synthetic_entities", "characters.json"
            ),
            "r",
        )
    )
    all_objects = json.load(
        open(
            os.path.join(
                global_utils.DEFAULT_DATA_DIR, "synthetic_entities", "bottles.json"
            ),
            "r",
        )
    )
    all_states = json.load(
        open(
            os.path.join(
                global_utils.DEFAULT_DATA_DIR, "synthetic_entities", "drinks.json"
            ),
            "r",
        )
    )

    dataset = exp_to_ds_func_map[experiment_name](
        all_characters,
        all_objects,
        all_states,
        2 * (train_size + valid_size),
    )

    if lm is not None:
        print("Filtering dataset on LM...")
        dataset, lm_acc = filter_dataset_on_lm(
            lm, dataset, train_size + valid_size, remote=remote
        )
        print("-" * 30)
        print(f"Len: {len(dataset)} | LM accuracy: {lm_acc:.2f}")
        print("-" * 30)
        if len(dataset) < train_size + valid_size:
            print("WARNING: Dataset is smaller than train_size + valid_size")
        if len(dataset) < train_size + valid_size // 2:
            raise ValueError("Dataset is too small")

    train_dataset = dataset[:train_size]
    valid_dataset = dataset[train_size:]

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, valid_dataloader


def prepare_bigtom_dataset(
    experiment_name: str,
    train_size: int = 80,
    valid_size: int = 80,
    batch_size: int = 4,
):
    df_false = pd.read_csv(
        os.path.join(
            global_utils.DEFAULT_DATA_DIR,
            "bigtom",
            "0_forward_belief_false_belief",
            "stories.csv",
        ),
        delimiter=";",
    )

    df_true = pd.read_csv(
        os.path.join(
            global_utils.DEFAULT_DATA_DIR,
            "bigtom",
            "0_forward_belief_true_belief",
            "stories.csv",
        ),
        delimiter=";",
    )

    dataset = bigtom_exp_to_ds_func_map[experiment_name](
        df_false,
        df_true,
        train_size + valid_size,
    )

    train_dataset = dataset[:train_size]
    valid_dataset = dataset[train_size:]

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, valid_dataloader


@dataclass(frozen=False)
class Accuracy(DataClassJsonMixin):
    accuracy: float
    rank: int = None  # full rank if None
    metadata: dict = None
