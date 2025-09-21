#!/usr/bin/env python3
"""
Memorization Analysis Script

This script performs causal mediation analysis to understand memorization in language models.
It implements Intervention-Induced Accuracy (IIA) analysis by corrupting specific tokens
and measuring the impact on model predictions across different layers.

Based on the memorization.ipynb notebook.
"""

import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from nnsight import LanguageModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import get_memorization_exps


def load_environment_variables():
    """Load environment variables from env.yml file."""
    env_file = "../../env.yml"
    if os.path.exists(env_file):
        with open(env_file, "r") as f:
            for line in f:
                if "HF_TOKEN" in line or "NDIF_API_KEY" in line:
                    key, value = line.split(":", 1)
                    os.environ[key.strip()] = value.strip()


def initialize_model():
    """Initialize the language model with appropriate configuration."""
    model = LanguageModel(
        "meta-llama/Meta-Llama-3-70B-Instruct",
        device_map="auto",
        dtype=torch.float16,
        dispatch=True,
    )
    return model


def load_data():
    """Load all required data files."""
    all_characters = json.load(
        open("/disk/u/nikhil/mind/data/synthetic_entities/characters.json", "r")
    )
    all_containers = json.load(
        open("/disk/u/nikhil/mind/data/synthetic_entities/bottles.json", "r")
    )
    all_states = json.load(
        open("/disk/u/nikhil/mind/data/synthetic_entities/drinks.json", "r")
    )
    story_templates = json.load(
        open("/disk/u/nikhil/mind/data/story_templates.json", "r")
    )

    return all_characters, all_containers, all_states, story_templates


def create_dataset_and_dataloader(
    model,
    story_templates,
    all_characters,
    all_containers,
    all_states,
    num_samples=1,
    batch_size=1,
):
    """Create dataset and dataloader for the analysis."""
    dataset = get_memorization_exps(
        model, story_templates, all_characters, all_containers, all_states, num_samples
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    prompt_len = len(model.tokenizer.encode(dataset[0]["clean_prompt"]))

    return dataset, dataloader, prompt_len


def run_iia_analysis(model, dataloader, prompt_len) -> dict:
    """
    Run noising intervention analysis to identify causally relevant tokens.

    This function performs causal mediation analysis by:
    1. Corrupting specific tokens in the input
    2. Measuring the impact on model predictions across different layers
    3. Computing accuracy scores for each token-layer combination

    Returns:
        dict: IIA results
    """
    iia = defaultdict(dict)
    intervened_logits = torch.zeros(
        prompt_len, model.config.num_hidden_layers, len(dataloader.dataset)
    )

    with torch.no_grad():
        for token_idx in range(prompt_len - 1, 129, -1):
            for layer_idx in tqdm(
                range(0, model.config.num_hidden_layers, 1),
                desc="Layers for token: {}".format(token_idx),
            ):
                correct, total = 0, 0

                for bi, batch in enumerate(dataloader):
                    corrupt_prompt = batch["corrupt_prompt"]
                    clean_prompt = batch["clean_prompt"]
                    clean_ans = batch["clean_ans"]
                    batch_size = len(clean_ans)

                    # Get corrupted layer output
                    with model.trace() as tracer:
                        barrier = tracer.barrier(2)

                        with tracer.invoke(corrupt_prompt):
                            corrupt_layer_out = (
                                model.model.layers[layer_idx]
                                .output[:, token_idx]
                                .clone()
                            )
                            barrier()

                        # Apply intervention and get prediction
                        with tracer.invoke(clean_prompt):
                            barrier()
                            model.model.layers[layer_idx].output[:, token_idx] = (
                                corrupt_layer_out
                            )
                            logits = model.lm_head.output[:, -1].save()

                    # Evaluate predictions
                    for i in range(batch_size):
                        pred = logits[i].argmax(dim=-1)
                        pred_token = model.tokenizer.decode([pred]).lower().strip()
                        clean_answer_token = clean_ans[i].lower().strip()
                        intervened_logits[token_idx][layer_idx][bi * batch_size + i] = (
                            logits[i, pred].save()
                        )

                        if pred_token != clean_answer_token:
                            correct += 1
                        total += 1

                    del pred
                    torch.cuda.empty_cache()

                iia[token_idx][layer_idx] = correct / total

                # Save intermediate results
                with open("iia.json", "w") as f:
                    json.dump(iia, f, indent=4)

                # Save intervened logits as a tensor to the disk
                torch.save(intervened_logits, "intervened_logits.pt")

    return iia


def sort_iia_results(iia):
    """Sort IIA results by token and layer indices."""
    # Sort by token index
    iia = dict(sorted(iia.items(), key=lambda x: x[0]))
    # Sort by layer index
    iia = {k: dict(sorted(v.items(), key=lambda x: x[0])) for k, v in iia.items()}
    return iia


def visualize_iia_results(iia, save_path=None):
    """
    Create a heatmap visualization of IIA results.

    Args:
        iia: Dictionary containing IIA results
        save_path: Optional path to save the plot
    """
    # Convert to DataFrame
    iia_df = pd.DataFrame(iia)

    # Reverse the order of the columns for better visualization
    iia_df = iia_df.iloc[::-1]

    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(iia_df, cmap="YlGnBu")
    plt.title("Intervention-Induced Accuracy (IIA) Heatmap")
    plt.xlabel("Token Index")
    plt.ylabel("Layer Index")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Heatmap saved to {save_path}")

    plt.show()


def main():
    """Main function to run the memorization analysis."""
    print("Starting Memorization Analysis...")

    # Load environment variables
    load_environment_variables()

    # Initialize model
    print("Initializing model...")
    model = initialize_model()

    # Load data
    print("Loading data...")
    all_characters, all_containers, all_states, story_templates = load_data()

    # Create dataset and dataloader
    print("Creating dataset...")
    num_samples = 50
    batch_size = 50
    _, dataloader, prompt_len = create_dataset_and_dataloader(
        model,
        story_templates,
        all_characters,
        all_containers,
        all_states,
        num_samples,
        batch_size,
    )

    print(f"Dataset size: {len(dataloader.dataset)}")
    print(f"Dataset created with prompt length: {prompt_len}")

    # Run IIA analysis
    print("Running IIA analysis...")
    iia = run_iia_analysis(model, dataloader, prompt_len)

    # Sort results
    iia = sort_iia_results(iia)

    # Visualize results
    print("Creating visualization...")
    visualize_iia_results(iia, save_path="iia_heatmap.png")

    # Save final results
    with open("iia_final.json", "w") as f:
        json.dump(iia, f, indent=2)

    print("Analysis complete! Results saved to iia_final.json and iia_heatmap.png")

    return iia


if __name__ == "__main__":
    results = main()
