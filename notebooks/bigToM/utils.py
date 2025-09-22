import random

import torch

random.seed(10)


def get_bigtom_samples(df_false, df_true, n_samples, belief_type="false_belief"):
    """
    Get the samples for the bigtom dataset.

    Args:
        df_false: The dataframe for the false belief scenario.
        df_true: The dataframe for the true belief scenario.
        n_samples: The number of samples to get.
        belief_type: The type of belief to get.
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


def get_tb_fb_answer(tb_ans, fb_ans):
    """
    Get the answer for the true belief and false belief scenario.

    Args:
        tb_ans: The answer for the true belief scenario.
        fb_ans: The answer for the false belief scenario.
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
    df_false,
    df_true,
    n_samples,
):
    """
    Get the samples for the answer lookback pointer experiment.

    Args:
        df_false: The dataframe for the false belief scenario.
        df_true: The dataframe for the true belief scenario.
        n_samples: The number of samples to get.
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
                "corrupt_story": true_stories[random_sample_idx]["story"],
                "corrupt_question": true_stories[random_sample_idx]["question"],
                "corrupt_prompt": alt_prompt,
                "corrupt_ans": alt_ans,
                "target": org_fb_ans,
            }
        )

    return samples


def get_answer_lookback_payload_exps(
    df_false,
    df_true,
    n_samples,
):
    """
    Get the samples for the answer lookback payload experiment.

    Args:
        df_false: The dataframe for the false belief scenario.
        df_true: The dataframe for the true belief scenario.
        n_samples: The number of samples to get.
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
                "corrupt_story": true_stories[random_sample_idx]["story"],
                "corrupt_question": true_stories[random_sample_idx]["question"],
                "corrupt_prompt": alt_prompt,
                "corrupt_ans": alt_ans,
                "target": alt_ans,
            }
        )

    return samples


def get_binding_lookback_pointer_exps(df_false, df_true, n_samples):
    """
    Get the samples for the binding lookback pointer experiment.

    Args:
        df_false: The dataframe for the false belief scenario.
        df_true: The dataframe for the true belief scenario.
        n_samples: The number of samples to get.
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
                "corrupt_story": true_stories[idx]["story"],
                "corrupt_question": alt_question,
                "corrupt_prompt": alt_prompt,
                "corrupt_ans": alt_ans,
                "target": org_tb_ans,
            }
        )

    return samples


def get_visibility_lookback_exps(
    df_false,
    df_true,
    n_samples,
):
    """
    Get the samples for the visibility lookback experiment.

    Args:
        df_false: The dataframe for the false belief scenario.
        df_true: The dataframe for the true belief scenario.
        n_samples: The number of samples to get.
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

        alt_story = true_stories[idx]["story"]
        alt_question = true_stories[idx]["question"]
        alt_prompt = f"Instructions: {instruction}\n\nStory: {alt_story}\nQuestion: {alt_question}\nAnswer:"
        alt_ans = tb_ans

        org_story = false_stories[idx]["story"]
        org_question = false_stories[idx]["question"]
        org_prompt = f"Instructions: {instruction}\n\nStory: {org_story}\nQuestion: {org_question}\nAnswer:"
        org_ans = fb_ans

        samples.append(
            {
                "clean_story": org_story,
                "clean_question": org_question,
                "clean_prompt": org_prompt,
                "clean_ans": org_ans,
                "corrupt_story": alt_story,
                "corrupt_question": alt_question,
                "corrupt_prompt": alt_prompt,
                "corrupt_ans": alt_ans,
                "target": alt_ans,
            }
        )

    return samples


def get_ques_start_token_idx(tokenizer, prompt):
    """
    Get the start index of the question.

    Args:
        tokenizer: The tokenizer to use.
        prompt: The prompt to get the start index of the question of.
    """
    input_tokens = tokenizer(prompt, return_tensors="pt").input_ids.squeeze()
    corrolary_token = tokenizer(":", return_tensors="pt").input_ids.squeeze()[-1].item()
    ques_start_idx = (input_tokens == corrolary_token).nonzero()[2].item()

    return ques_start_idx - 1


def get_visitibility_sent_start_idx(tokenizer, prompt):
    """
    Get the start index of the visitibility sentence.

    Args:
        tokenizer: The tokenizer to use.
        prompt: The prompt to get the start index of the visitibility sentence of.
    """
    input_tokens = tokenizer(prompt, return_tensors="pt").input_ids.squeeze()

    story_start_idx = (input_tokens == 18422).nonzero()[0].item()
    period_idx = (input_tokens == 13).nonzero(as_tuple=True)[0]
    period_idx = period_idx[period_idx > story_start_idx]

    return period_idx[-1] + 1


def get_prompt_token_len(tokenizer, prompt):
    """
    Get the token length of the prompt.

    Args:
        tokenizer: The tokenizer to use.
        prompt: The prompt to get the token length of.
    """
    input_tokens = tokenizer(prompt, return_tensors="pt").input_ids.squeeze()
    return len(input_tokens)


def check_pred(model, pred, target, verbose=False):
    """
    Check if the prediction is correct or not.

    Args:
        model: The model to use.
        pred: The prediction to check.
        target: The ground truth.
        verbose: Whether to print the prompt.

    Returns:
        "Yes" if the prediction is correct, "No" otherwise.
    """
    prompt = f"Instruction: Check if the following ground truth and prediction are same or different. If they are the same, then predict 'Yes', else 'No' \n\nGround truth: {target}\nPrediction: {pred}\nAnswer:"
    if verbose:
        print(prompt)

    with torch.no_grad():
        with model.generate(
            prompt,
            max_new_tokens=2,
            do_sample=False,
            num_return_sequences=1,
            pad_token_id=model.tokenizer.pad_token_id,
        ):
            out = model.generator.output.save()

    prompt_len = get_prompt_token_len(model.tokenizer, prompt)

    return model.tokenizer.decode(out[0][prompt_len:-1]).strip()
