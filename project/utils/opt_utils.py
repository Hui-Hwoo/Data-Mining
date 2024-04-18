import collections
import json
import os
import pickle
import re
import sys

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

sys.path.insert(0, ROOT_PATH)

import numpy as np
from utils import eval_utils
import pandas as pd


def extract_string_in_square_brackets(input_string):
    raw_result = re.findall(r"\[.*?\]", input_string)
    if raw_result:
        return raw_result[0][1:-1]
    else:
        return ""


def gen_ins_and_score_pairs_substr(
    old_instructions_and_scores,
    old_instruction_score_threshold=0.1,
    max_num_instructions=1000,
    return_str_only=False,
    num_score_buckets=np.inf,
):
    """Generate the string that includes instruction-score pairs."""
    print(f"hui old_instructions_and_scores: {old_instructions_and_scores}")

    old_instructions_and_scores_str = ""
    old_instructions_and_scores = sorted(
        old_instructions_and_scores, key=lambda x: x[1]
    )[-max_num_instructions:]
    old_instructions_and_scores_in_meta_prompt = []
    for instruction, score, i_step in old_instructions_and_scores:
        if (
            not old_instruction_score_threshold
            or score >= old_instruction_score_threshold
        ):
            old_instructions_and_scores_in_meta_prompt.append(
                (instruction, score, i_step)
            )
            if num_score_buckets == np.inf:
                score_to_show = round(score, 3)
            else:
                score_to_show = round(score * num_score_buckets)
            old_instructions_and_scores_str += (
                f"\ntext:\n{instruction}\nscore:\n{score_to_show}\n"
            )
    if return_str_only:
        return old_instructions_and_scores_str
    else:
        return (
            old_instructions_and_scores_str,
            old_instructions_and_scores_in_meta_prompt,
        )


def gen_meta_prompt(
    old_instructions_and_scores,
    optimizer_llm_name,
    old_instruction_score_threshold=0.1,
    max_num_instructions=1000,
    data=None,
    few_shot_index_list=None,
    num_score_buckets=np.inf,
):
    """Generate meta prompt for instruction rewriting."""
    meta_prompt = ""

    if optimizer_llm_name.lower() in {"gpt-3.5-turbo", "gpt-4"}:
        meta_prompt_old_instruction_part = (
            "Your task is to generate the answer starting sentence <Start>."
            " Below are some previous starting sentences with their scores."
            " The score ranges from 0 to 100.\n"
        )
    else:
        assert optimizer_llm_name.lower() == "text-bison"
        meta_prompt_old_instruction_part = (
            "I have some texts along with their corresponding scores."
            " The texts are arranged in ascending order based on their scores,"
            " where higher scores indicate better quality.\n\n"
        )
    # add old instructions
    old_instructions_and_scores_str = gen_ins_and_score_pairs_substr(
        old_instructions_and_scores=old_instructions_and_scores,
        old_instruction_score_threshold=0.1 or old_instruction_score_threshold,
        max_num_instructions=max_num_instructions,
        return_str_only=True,
        num_score_buckets=num_score_buckets,
    )

    meta_prompt_old_instruction_part += old_instructions_and_scores_str
    meta_prompt_exemplar_part = ""

    if optimizer_llm_name.lower() in {"gpt-3.5-turbo", "gpt-4"}:
        meta_prompt_exemplar_part += "Below are some problems.\n"
    else:
        meta_prompt_exemplar_part += (
            "The following exemplars show how to apply your text: you replace"
            " <INS> in each input with your text, then read the input and give"
            " an output. We say your output is wrong if your output is"
            " different from the given output, and we say your output is"
            " correct if they are the same. When replacing <INS> with an old"
            " piece of text above, we get wrong outputs on the following"
            " inputs.\n\n"
        )
    for idx in few_shot_index_list:
        question = data.iloc[idx, 0]
        true_answer = data.iloc[idx, 1]

        meta_prompt_exemplar_part += f"\ninput:\nQ: <INS>\n{question}\nA:"

        if optimizer_llm_name.lower() in {"gpt-3.5-turbo", "gpt-4"}:
            meta_prompt_exemplar_part += f"\nGround truth answer:\n{true_answer}\n"
        else:
            meta_prompt_exemplar_part += f"\noutput: {true_answer}\n"

    meta_prompt += meta_prompt_old_instruction_part + "\n\n" + meta_prompt_exemplar_part

    if optimizer_llm_name.lower() in {"gpt-3.5-turbo", "gpt-4"}:
        meta_prompt += (
            "\n\nGenerate a starting sentence that is different from all the"
            " <Start> sentences above, and has a higher score than all the"
            " <Start> sentences above. The starting sentence should begin with"
            " <Start> and end with </Start>. The starting sentence should be"
            " concise, effective, and generally applicable to all QA pairs"
            " above."
        )
    else:
        meta_prompt += (
            "\n\nWrite your new text that is different from the old ones and"
            " has a score as high as possible. Write the text in square brackets([])."
        )
    return meta_prompt


def run_evolution(**kwargs):
    """The function for evolution."""
    # ================= experiment configurations =============================
    model_config = kwargs["model_config"]
    raw_data = kwargs["raw_data"]
    instruction_config = kwargs["instruction_config"]
    scorer_llm_dict = kwargs["scorer_llm_dict"]
    optimizer_llm_dict = kwargs["optimizer_llm_dict"]
    save_folder = kwargs["save_folder"]
    verbose = kwargs["verbose"] if "verbose" in kwargs else False

    # =================== save configurations to json file ====================
    configs_dict = {
        "model_config": model_config,
        "instruction_config": instruction_config,
    }

    with open(
        os.path.join(save_folder["result_by_instruction_folder"], "configs_dict.json"),
        "w",
    ) as f:
        json.dump(configs_dict, f, indent=4)

    # ================ split data into train/val/test ==========================
    train_ratio = model_config["train_ratio"]
    eval_ratio = model_config["eval_ratio"]
    test_ratio = 1 - train_ratio - eval_ratio
    num_examples = raw_data.shape[0]
    np.random.seed(0)
    train_index = np.sort(
        np.array(
            np.random.choice(
                num_examples, size=int(train_ratio * num_examples), replace=False
            )
        )
    )
    eval_and_test_index = np.sort(
        np.array(list(set(np.arange(num_examples)) - set(train_index)))
    )
    eval_index = np.sort(
        np.array(
            np.random.choice(
                eval_and_test_index,
                size=int(eval_ratio * num_examples),
                replace=False,
            )
        )
    )

    generated_ins_on_few_shot_results_dict = dict()
    old_ins_on_few_shot_results_dict = dict()
    eval_results = []  # format: [(i_step, instruction, detailed_results_df)]
    old_instructions_and_scores_raw = []  # format: [(instruction, score, step_index)
    old_instructions_and_scores = []
    meta_prompts = []  # format: [(meta_prompt, step_index)]
    instruction_score_dict = dict()  # {instruction: score}
    # key: step index; value: the list of few-shot indices in that step
    few_shot_index_list_by_step_dict = dict()
    detailed_results_df_by_instruction_dict = dict()
    wrong_questions_from_start_counter = collections.Counter()
    # EVAL results
    eval_detailed_results_df_dict = dict()  # {instruction: detailed_results_df}
    instruction_eval_score_dict = dict()  # {instruction: eval_score}
    old_instruction_md5_hashstrings_set = set()

    print(
        f"train_ratio: {train_ratio}, num of training points: {int(num_examples * train_ratio)}"
    )
    print(
        f"eval_ratio: {eval_ratio}, number of eval points: "
        f"{int(num_examples * eval_ratio)}"
    )
    print(
        f"test_ratio: {test_ratio}, number of test points: "
        f"{int(num_examples * test_ratio)}"
    )
    print(
        f"optimizer llm temperature: {optimizer_llm_dict['temperature']}, schedule:"
        f" {model_config['optimizer_llm_temperature_schedule']}"
    )
    print(
        f"generating {instruction_config['num_generated_instructions_in_each_step']} instructions in"
        f" each step, run for {model_config['num_search_steps']} steps"
    )
    print(
        "discarding generated instructions with score less than:"
        f" {instruction_config['old_instruction_score_threshold']} (old_instruction_score_threshold)"
    )
    print(f"num_score_buckets: {instruction_config['num_score_buckets']}")

    prev_saved_instructions = set()

    # evaluate initial instructions
    print("\n============== evaluating initial instructions ===============")
    for instruction in instruction_config["initial_instructions"]:
        print(f"""computing the score of "{instruction}" by prompting""")

        detailed_results_df = eval_utils.evaluate_single_instruction(
            data=raw_data,
            instruction=instruction,
            eval_index_all=train_index,
            call_server_func=scorer_llm_dict["call_server_func"],
            extract_final_answer_by_prompting_again=False,
            prediction_num_decimals=0,
            verbose=verbose,
        )

        detailed_results_df_by_instruction_dict[instruction] = detailed_results_df
        scores = detailed_results_df["accuracy"]
        average_score = np.average(scores)
        print(f"instruction: {instruction}, score: {average_score}")
        filename = eval_utils.instruction_to_filename(instruction)
        file_path = os.path.join(
            save_folder["result_by_instruction_folder"], f"{filename}.csv"
        )
        detailed_results_df.to_csv(file_path, index=True, header=True)
        print(
            f"""saving results of "{instruction}" to {'/'.join(file_path.split("/")[-2:])}"""
        )
        old_instructions_and_scores.append((instruction, average_score, -1))
        old_instructions_and_scores_raw.append((instruction, average_score, -1))
        instruction_score_dict[instruction] = average_score

        # increment the counter on wrong questions
        wrong_question_indices_set = set(
            list(
                detailed_results_df.iloc[
                    np.where(detailed_results_df.accuracy == 0.0)[0], :
                ].index
            )
        )
        for idx in wrong_question_indices_set:
            wrong_questions_from_start_counter[idx] += 1

    # evolution
    num_search_steps = model_config["num_search_steps"]
    optimizer_llm_temperature = optimizer_llm_dict["temperature"]
    for i_step in range(num_search_steps):
        print(f"\n================== Step {i_step} =====================")
        if not i_step % 10:
            print(f"old_instructions_and_scores: {old_instructions_and_scores}")

        optimizer_llm_temperature_curr = optimizer_llm_temperature
        print(f"current optimizer_llm_temperature: {optimizer_llm_temperature_curr}")

        # generate new instructions
        num_few_shot_questions_for_instruction_refinement = instruction_config[
            "num_few_shot_questions_for_instruction_refinement"
        ]

        np.random.seed(i_step)
        few_shot_index_list = np.sort(
            np.random.choice(
                train_index,
                num_few_shot_questions_for_instruction_refinement,
                replace=False,
            )
        ).tolist()

        few_shot_index_list_by_step_dict[i_step] = few_shot_index_list

        meta_prompt = gen_meta_prompt(
            old_instructions_and_scores=old_instructions_and_scores,
            optimizer_llm_name=optimizer_llm_dict["name"],
            old_instruction_score_threshold=instruction_config[
                "old_instruction_score_threshold"
            ],
            max_num_instructions=instruction_config["max_num_instructions"],
            data=raw_data,
            few_shot_index_list=few_shot_index_list,
            num_score_buckets=instruction_config["num_score_buckets"],
        )

        print(f"\nmeta_prompt: \n\n{meta_prompt}\n")
        meta_prompts.append((meta_prompt, i_step))
        remaining_num_instructions_to_generate = instruction_config[
            "num_generated_instructions_in_each_step"
        ]
        generated_instructions_raw = []

        while remaining_num_instructions_to_generate > 0:
            optimizer_llm_input_text = meta_prompt
            # generate instructions
            print(f"current temperature: {optimizer_llm_temperature_curr}")
            raw_outputs = optimizer_llm_dict["call_optimizer_server_func"](
                optimizer_llm_input_text,
                temperature=optimizer_llm_temperature_curr,
            )

            raw_outputs = raw_outputs[:remaining_num_instructions_to_generate]
            generated_instructions_raw += [
                extract_string_in_square_brackets(string) for string in raw_outputs
            ]

            remaining_num_instructions_to_generate -= optimizer_llm_dict["batch_size"]

        generated_instructions_raw = list(
            map(eval_utils.polish_sentence, generated_instructions_raw)
        )
        print(f"\ninitially generated instructions: {generated_instructions_raw}\n")

        # do not evaluate old instructions again
        generated_instructions = []  # the new instructions generated in this step
        for ins in generated_instructions_raw:
            ins_md5_hashstring = eval_utils.instruction_to_filename(
                ins, md5_hashing=True
            )
            if ins_md5_hashstring not in old_instruction_md5_hashstrings_set:
                generated_instructions.append(ins)
                old_instruction_md5_hashstrings_set.add(ins_md5_hashstring)
            else:
                print(f"already evaluated '{ins}' previously")
        generated_instructions = list(set(generated_instructions))

        to_evaluate_instructions = []
        for instruction in generated_instructions:
            if (
                len(instruction) > 500
                or any(char.isdigit() for char in instruction)
                or "INS" in instruction
            ):
                continue
            to_evaluate_instructions.append(instruction)
        print(f"\nto-evaluate generated instructions: {to_evaluate_instructions}\n")

        # evaluate newly generated instructions on the training set
        for instruction in to_evaluate_instructions:
            if instruction not in prev_saved_instructions:
                print(f"""computing the score of "{instruction}" by prompting""")
                detailed_results_df = eval_utils.evaluate_single_instruction(
                    data=raw_data,
                    instruction=instruction,
                    eval_index_all=train_index,
                    call_server_func=scorer_llm_dict["call_server_func"],
                    extract_final_answer_by_prompting_again=False,
                    verbose=verbose,
                )
                prev_saved_instructions.add(instruction)
            else:
                # do not re-evaluate instructions that had been evaluated previously
                detailed_results_df = pd.read_csv(
                    os.path.join(
                        save_folder["result_by_instruction_folder"],
                        f"{instruction}.csv",
                    ),
                    index_col=0,
                    header=0,
                )
                print(f"""reading previously saved "{instruction}" information""")

            scores = detailed_results_df["accuracy"]
            average_score = np.average(scores)
            print(f"Step {i_step}, instruction: {instruction}, score: {average_score}")

            # increment the counter on wrong questions
            wrong_question_indices_set = set(
                list(detailed_results_df[detailed_results_df["accuracy"] == 0.0].index)
            )
            for idx in wrong_question_indices_set:
                wrong_questions_from_start_counter[idx] += 1

            filename = eval_utils.instruction_to_filename(instruction)
            file_path = os.path.join(
                save_folder["result_by_instruction_folder"], f"""{filename}.csv"""
            )
            detailed_results_df.to_csv(file_path, index=True, header=True)
            print(f"saving results to {file_path}")

            detailed_results_df_by_instruction_dict[instruction] = detailed_results_df
            old_instructions_and_scores.append((instruction, average_score, i_step))
            instruction_score_dict[instruction] = average_score

        # record all generated instructions
        for instruction in generated_instructions_raw:
            if instruction in instruction_score_dict:
                average_score = instruction_score_dict[instruction]
            else:
                average_score = np.nan
            old_instructions_and_scores_raw.append((instruction, average_score, i_step))

        # =============================== eval ====================================
        # every eval_interval steps, evaluate the instructions that were generated
        # in the current step and were not skipped
        if not i_step % model_config["eval_interval"]:
            for instruction in generated_instructions_raw:
                # if the instruction wasn't skipped in any step
                if instruction in instruction_score_dict:
                    if instruction not in instruction_eval_score_dict:
                        detailed_results_df = eval_utils.evaluate_single_instruction(
                            data=raw_data,
                            instruction=instruction,
                            eval_index_all=eval_index,
                            call_server_func=scorer_llm_dict["call_server_func"],
                            extract_final_answer_by_prompting_again=False,
                            prediction_num_decimals=0,
                            verbose=verbose,
                        )
                        eval_score = np.average(detailed_results_df["accuracy"])
                        eval_detailed_results_df_dict[instruction] = detailed_results_df
                        instruction_eval_score_dict[instruction] = eval_score
                    else:
                        eval_score = instruction_eval_score_dict[instruction]
                    print(
                        f"EVAL: \nStep {i_step}, instruction: {instruction}, eval score:"
                        f" {eval_score:.2f}"
                    )
                    eval_results.append((i_step, instruction, eval_score))

        # ===================== save up-to-date results ===========================
        results_dict = dict()
        results_dict["meta_prompts"] = meta_prompts
        results_dict["old_instructions_and_scores"] = list(old_instructions_and_scores)
        results_dict["old_instructions_and_scores_raw"] = list(
            old_instructions_and_scores_raw
        )
        results_dict["generated_ins_on_few_shot_results_dict"] = (
            generated_ins_on_few_shot_results_dict
        )
        results_dict["old_ins_on_few_shot_results_dict"] = (
            old_ins_on_few_shot_results_dict
        )
        results_dict["few_shot_index_list_by_step_dict"] = (
            few_shot_index_list_by_step_dict
        )
        results_dict["eval_results"] = eval_results
        results_dict["eval_detailed_results_df_dict"] = eval_detailed_results_df_dict
        with open(
            os.path.join(
                save_folder["result_by_instruction_folder"], "results_dict.pkl"
            ),
            "wb",
        ) as fp:
            pickle.dump(results_dict, fp)
        print(f"\nsaved all results to\n{save_folder['result_by_instruction_folder']}")
