import random

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

random.seed(10)


def get_bigtom_samples(
    df_false: pd.DataFrame,
    df_true: pd.DataFrame,
    n_samples: int,
    belief_type: str = "false_belief",
) -> list:
    """
    Get the samples for the bigtom dataset.

    Args:
        df_false: The dataframe for the false belief scenario.
        df_true: The dataframe for the true belief scenario.
        n_samples: The number of samples to get.
        belief_type: The type of belief to get.

    Returns:
        list: List of dictionaries containing clean and counterfactual samples with their configurations.
    """
    true_stories, false_stories = [], []
    for i in range(len(df_true)):
        story = df_true.iloc[i]["story"]
        question = df_true.iloc[i]["question"]
        answer = df_true.iloc[i]["answer"]
        distractor = df_true.iloc[i]["distractor"]
        true_stories.append(
            {
                "story": story,
                "question": question,
                "answer": answer,
                "distractor": distractor,
            }
        )

    for i in range(len(df_false)):
        story = df_false.iloc[i]["story"]
        question = df_true.iloc[i]["question"]
        answer = df_false.iloc[i]["answer"]
        distractor = df_false.iloc[i]["distractor"]
        false_stories.append(
            {
                "story": story,
                "question": question,
                "answer": answer,
                "distractor": distractor,
            }
        )

    samples = []
    instruction = "1. Track the belief of each character as described in the story. 2. A character's belief is formed only when they perform an action themselves or can observe the action taking place. 3. A character does not have any belief about the container or its content which they cannot observe directly. 4. To answer the question, predict only the final state of the queried container in fewest tokens possible, strictly based on the belief of the character, mentioned in the question. 5. Do not predict the entire sentence with character or container as the final output."

    for idx in range(n_samples):
        tb_ans = true_stories[idx]["answer"].split()
        fb_ans = false_stories[idx]["answer"].split()

        # Find the index of first word which is different in both answers
        diff_idx = 0
        for i, (v, j) in enumerate(zip(tb_ans, fb_ans)):
            if v != j:
                diff_idx = i
                break

        tb_ans = " ".join(tb_ans[diff_idx:])[:-1]
        fb_ans = " ".join(fb_ans[diff_idx:])[:-1]

        if belief_type == "true_belief":
            question = true_stories[idx]["question"]
            tb_prompt = f"Instructions: {instruction}\n\nStory: {true_stories[idx]['story']}\nQuestion: {question}\nAnswer:"
            samples.append({"prompt": tb_prompt, "answer": tb_ans})
        else:
            question = false_stories[idx]["question"]
            fb_prompt = f"Instructions: {instruction}\n\nStory: {false_stories[idx]['story']}\nQuestion: {question}\nAnswer:"
            samples.append({"prompt": fb_prompt, "answer": fb_ans})

    return samples


def get_tb_fb_answer(tb_ans: list, fb_ans: list) -> tuple:
    """
    Get the answer for the true belief and false belief scenario.

    Args:
        tb_ans: The answer for the true belief scenario.
        fb_ans: The answer for the false belief scenario.

    Returns:
        tuple: The answer for the true belief and false belief scenario.
    """
    diff_idx = 0
    for i, (v, j) in enumerate(zip(tb_ans, fb_ans)):
        if v != j:
            diff_idx = i
            break

    tb_ans = " ".join(tb_ans[diff_idx:])[:-1]
    fb_ans = " ".join(fb_ans[diff_idx:])[:-1]

    return tb_ans, fb_ans


def get_answer_lookback_pointer_exps(
    df_false: pd.DataFrame,
    df_true: pd.DataFrame,
    n_samples: int,
) -> list:
    """
    Get the samples for the answer lookback pointer experiment.

    Args:
        df_false: The dataframe for the false belief scenario.
        df_true: The dataframe for the true belief scenario.
        n_samples: The number of samples to get.

    Returns:
        list: List of dictionaries containing clean and counterfactual samples with their configurations.
    """
    true_stories, false_stories = [], []
    for i in range(len(df_true)):
        story = df_true.iloc[i]["story"]
        question = df_true.iloc[i]["question"]
        answer = df_true.iloc[i]["answer"]
        distractor = df_true.iloc[i]["distractor"]
        true_stories.append(
            {
                "story": story,
                "question": question,
                "answer": answer,
                "distractor": distractor,
            }
        )

    for i in range(len(df_false)):
        story = df_false.iloc[i]["story"]
        question = df_true.iloc[i]["question"]
        answer = df_false.iloc[i]["answer"]
        distractor = df_false.iloc[i]["distractor"]
        false_stories.append(
            {
                "story": story,
                "question": question,
                "answer": answer,
                "distractor": distractor,
            }
        )

    samples = []
    instruction = "1. Track the belief of each character as described in the story. 2. A character's belief is formed only when they perform an action themselves or can observe the action taking place. 3. A character does not have any belief about the container or its content which they cannot observe directly. 4. To answer the question, predict only the final state of the queried container in fewest tokens possible, strictly based on the belief of the character, mentioned in the question. 5. Do not predict the entire sentence with character or container as the final output."

    for idx in range(n_samples):
        org_tb_ans = true_stories[idx]["answer"].split()
        org_fb_ans = false_stories[idx]["answer"].split()
        org_tb_ans, org_fb_ans = get_tb_fb_answer(org_tb_ans, org_fb_ans)

        question = true_stories[idx]["question"]
        org_prompt = f"Instructions: {instruction}\n\nStory: {true_stories[idx]['story']}\nQuestion: {question}\nAnswer:"
        org_ans = org_tb_ans

        # Sample a different story
        random_sample_idx = random.choice(range(len(false_stories)))
        while random_sample_idx == idx:
            random_sample_idx = random.choice(range(len(false_stories)))

        alt_tb_ans = true_stories[random_sample_idx]["answer"].split()
        alt_fb_ans = false_stories[random_sample_idx]["answer"].split()
        alt_tb_ans, alt_fb_ans = get_tb_fb_answer(alt_tb_ans, alt_fb_ans)

        question = false_stories[random_sample_idx]["question"]
        alt_prompt = f"Instructions: {instruction}\n\nStory: {false_stories[random_sample_idx]['story']}\nQuestion: {question}\nAnswer:"
        alt_ans = alt_fb_ans

        samples.append(
            {
                "clean_story": false_stories[idx]["story"],
                "clean_question": false_stories[idx]["question"],
                "clean_prompt": org_prompt,
                "clean_ans": org_ans,
                "counterfactual_story": true_stories[random_sample_idx]["story"],
                "counterfactual_question": true_stories[random_sample_idx]["question"],
                "counterfactual_prompt": alt_prompt,
                "counterfactual_ans": alt_ans,
                "target": org_fb_ans,
            }
        )

    return samples


def get_answer_lookback_payload_exps(
    df_false: pd.DataFrame,
    df_true: pd.DataFrame,
    n_samples: int,
) -> list:
    """
    Get the samples for the answer lookback payload experiment.

    Args:
        df_false: The dataframe for the false belief scenario.
        df_true: The dataframe for the true belief scenario.
        n_samples: The number of samples to get.

    Returns:
        list: List of dictionaries containing clean and counterfactual samples with their configurations.
    """
    true_stories, false_stories = [], []
    for i in range(len(df_true)):
        story = df_true.iloc[i]["story"]
        question = df_true.iloc[i]["question"]
        answer = df_true.iloc[i]["answer"]
        distractor = df_true.iloc[i]["distractor"]
        true_stories.append(
            {
                "story": story,
                "question": question,
                "answer": answer,
                "distractor": distractor,
            }
        )

    for i in range(len(df_false)):
        story = df_false.iloc[i]["story"]
        question = df_true.iloc[i]["question"]
        answer = df_false.iloc[i]["answer"]
        distractor = df_false.iloc[i]["distractor"]
        false_stories.append(
            {
                "story": story,
                "question": question,
                "answer": answer,
                "distractor": distractor,
            }
        )

    samples = []
    instruction = "1. Track the belief of each character as described in the story. 2. A character's belief is formed only when they perform an action themselves or can observe the action taking place. 3. A character does not have any belief about the container or its content which they cannot observe directly. 4. To answer the question, predict only the final state of the queried container in fewest tokens possible, strictly based on the belief of the character, mentioned in the question. 5. Do not predict the entire sentence with character or container as the final output."

    for idx in range(n_samples):
        tb_ans = true_stories[idx]["answer"].split()
        fb_ans = false_stories[idx]["answer"].split()
        tb_ans, fb_ans = get_tb_fb_answer(tb_ans, fb_ans)

        question = true_stories[idx]["question"]
        org_prompt = f"Instructions: {instruction}\n\nStory: {true_stories[idx]['story']}\nQuestion: {question}\nAnswer:"
        org_ans = tb_ans

        # Sample a different story
        random_sample_idx = random.choice(range(len(true_stories)))
        while random_sample_idx == idx:
            random_sample_idx = random.choice(range(len(true_stories)))

        tb_ans = true_stories[random_sample_idx]["answer"].split()
        fb_ans = false_stories[random_sample_idx]["answer"].split()
        tb_ans, fb_ans = get_tb_fb_answer(tb_ans, fb_ans)

        question = true_stories[random_sample_idx]["question"]
        alt_prompt = f"Instructions: {instruction}\n\nStory: {true_stories[random_sample_idx]['story']}\nQuestion: {question}\nAnswer:"
        alt_ans = tb_ans

        samples.append(
            {
                "clean_story": true_stories[idx]["story"],
                "clean_question": true_stories[idx]["question"],
                "clean_prompt": org_prompt,
                "clean_ans": org_ans,
                "counterfactual_story": true_stories[random_sample_idx]["story"],
                "counterfactual_question": true_stories[random_sample_idx]["question"],
                "counterfactual_prompt": alt_prompt,
                "counterfactual_ans": alt_ans,
                "target": alt_ans,
            }
        )

    return samples


def get_binding_lookback_pointer_exps(
    df_false: pd.DataFrame,
    df_true: pd.DataFrame,
    n_samples: int,
) -> list:
    """
    Get the samples for the binding lookback pointer experiment.

    Args:
        df_false: The dataframe for the false belief scenario.
        df_true: The dataframe for the true belief scenario.
        n_samples: The number of samples to get.

    Returns:
        list: List of dictionaries containing clean and counterfactual samples with their configurations.
    """
    true_stories, false_stories = [], []
    for i in range(len(df_true)):
        story = df_true.iloc[i]["story"]
        question = df_true.iloc[i]["question"]
        answer = df_true.iloc[i]["answer"]
        distractor = df_true.iloc[i]["distractor"]
        true_stories.append(
            {
                "story": story,
                "question": question,
                "answer": answer,
                "distractor": distractor,
            }
        )

    for i in range(len(df_false)):
        story = df_false.iloc[i]["story"]
        question = df_true.iloc[i]["question"]
        answer = df_false.iloc[i]["answer"]
        distractor = df_false.iloc[i]["distractor"]
        false_stories.append(
            {
                "story": story,
                "question": question,
                "answer": answer,
                "distractor": distractor,
            }
        )

    samples = []
    instruction = "1. Track the belief of each character as described in the story. 2. A character's belief is formed only when they perform an action themselves or can observe the action taking place. 3. A character does not have any belief about the container or its content which they cannot observe directly. 4. To answer the question, predict only the final state of the queried container in fewest tokens possible, strictly based on the belief of the character, mentioned in the question. 5. Do not predict the entire sentence with character or container as the final output."

    for idx in range(n_samples):
        alt_tb_ans = true_stories[idx]["answer"].split()
        alt_fb_ans = false_stories[idx]["answer"].split()
        alt_tb_ans, alt_fb_ans = get_tb_fb_answer(alt_tb_ans, alt_fb_ans)

        alt_question = true_stories[idx]["question"]
        alt_prompt = f"Instructions: {instruction}\n\nStory: {true_stories[idx]['story']}\nQuestion: {alt_question}\nAnswer:"
        alt_ans = alt_tb_ans

        # Sample a different story
        random_sample_idx = random.choice(range(len(true_stories)))
        while random_sample_idx == idx:
            random_sample_idx = random.choice(range(len(true_stories)))

        org_tb_ans = true_stories[random_sample_idx]["answer"].split()
        org_fb_ans = false_stories[random_sample_idx]["answer"].split()
        org_tb_ans, org_fb_ans = get_tb_fb_answer(org_tb_ans, org_fb_ans)

        org_question = true_stories[random_sample_idx]["question"]
        # Replace the second word in org_question with the second word in org_question
        org_question = org_question.split()
        org_question[1] = alt_question.split()[1]
        org_question = " ".join(org_question)
        org_prompt = f"Instructions: {instruction}\n\nStory: {true_stories[random_sample_idx]['story']}\nQuestion: {org_question}\nAnswer:"
        org_ans = "unknown"

        samples.append(
            {
                "clean_story": true_stories[random_sample_idx]["story"],
                "clean_question": org_question,
                "clean_prompt": org_prompt,
                "clean_ans": org_ans,
                "counterfactual_story": true_stories[idx]["story"],
                "counterfactual_question": alt_question,
                "counterfactual_prompt": alt_prompt,
                "counterfactual_ans": alt_ans,
                "target": org_tb_ans,
            }
        )

    return samples


def get_visibility_lookback_exps(
    df_false: pd.DataFrame,
    df_true: pd.DataFrame,
    n_samples: int,
) -> list:
    """
    Get the samples for the visibility lookback experiment.

    Args:
        df_false: The dataframe for the false belief scenario.
        df_true: The dataframe for the true belief scenario.
        n_samples: The number of samples to get.

    Returns:
        list: List of dictionaries containing clean and counterfactual samples with their configurations.
    """
    true_stories, false_stories = [], []
    for i in range(len(df_true)):
        story = df_true.iloc[i]["story"]
        question = df_true.iloc[i]["question"]
        answer = df_true.iloc[i]["answer"]
        distractor = df_true.iloc[i]["distractor"]
        true_stories.append(
            {
                "story": story,
                "question": question,
                "answer": answer,
                "distractor": distractor,
            }
        )

    for i in range(len(df_false)):
        story = df_false.iloc[i]["story"]
        question = df_true.iloc[i]["question"]
        answer = df_false.iloc[i]["answer"]
        distractor = df_false.iloc[i]["distractor"]
        false_stories.append(
            {
                "story": story,
                "question": qu