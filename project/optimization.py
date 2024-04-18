import datetime
import functools
import os
import sys

from absl import app
import pandas as pd
from utils import prompt_utils, opt_utils, constant
import google.generativeai as palm
import openai

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, ROOT_PATH)
ROOT_DATA_FOLDER_PATH = os.path.join(ROOT_PATH, "data")


def main(_):
    # Config
    openai_api_key = constant.OPENAI_API_KEY
    palm_api_key = constant.PALM_API_KEY
    scorer_llm_name = "text-bison"  # "text-bison", "gpt-3.5-turbo", "gpt-4",
    optimizer_llm_name = "text-bison"
    task_name = "train"  # "test"

    optimizer_finetuned_palm_dict = {
        "num_decodes": 8,
        "max_decode_steps": 1024,
    }
    optimizer_llm_dict = {
        "max_decode_steps": 512,
        "num_decodes": 1,
        "batch_size": 1,
    }

    scorer_finetuned_palm_dict = {
        "max_decode_steps": 1024,
    }
    scorer_gpt_dict = {
        "max_decode_steps": 1024,
        "num_decodes": 1,
    }

    # =================== check the input arguments ==========================
    # make sure the scorer and optimizer models are callable
    if scorer_llm_name in {"gpt-3.5-turbo", "gpt-4"}:
        assert openai_api_key, "The OpenAI API key must be provided."
        openai.api_key = openai_api_key
    else:
        assert scorer_llm_name == "text-bison"
        assert (
            palm_api_key
        ), "A PaLM API key is needed when prompting the text-bison model."
        palm.configure(api_key=palm_api_key)

    if optimizer_llm_name in {"gpt-3.5-turbo", "gpt-4"}:
        assert openai_api_key, "The OpenAI API key must be provided."
        openai.api_key = openai_api_key
    else:
        assert optimizer_llm_name == "text-bison"
        assert (
            palm_api_key
        ), "A PaLM API key is needed when prompting the text-bison model."
        palm.configure(api_key=palm_api_key)

    # TODO: print info

    # =================== create the result directory ==========================
    datetime_str = (
        str(datetime.datetime.now().replace(microsecond=0))
        .replace(" ", "-")
        .replace(":", "-")
    )

    save_folder = os.path.join(
        ROOT_PATH,
        "outputs",
        "optimization-results",
        f"gsm8k-{task_name}-s-{scorer_llm_name}-o-{optimizer_llm_name}-{datetime_str}/",
    )
    result_by_instruction_folder = os.path.join(save_folder, "result_by_instruction")
    root_data_folder_path = os.path.join(ROOT_DATA_FOLDER_PATH, "gsm_data")
    os.makedirs(result_by_instruction_folder)
    print(
        f"\nresult directory: {f'gsm8k-{task_name}-s-{scorer_llm_name}-o-{optimizer_llm_name}-{datetime_str}/'}"
    )

    # ====================== scorer model configs ==============================
    # difference between num_decodes and batch_size:
    # - num_decodes: how many outputs we actually want for each input
    # - batch_size: the batch size in model serving, should equal to that in model serving config

    if scorer_llm_name == "text-bison":
        # when prompting text-bison with Cloud API
        call_scorer_finetuned_palm_server_func = functools.partial(
            prompt_utils.call_palm_server_from_cloud,
            model="text-bison-001",
            temperature=0.0,
            max_decode_steps=scorer_finetuned_palm_dict["max_decode_steps"],
        )

        scorer_llm_dict = {
            "name": "text-bison",
            "model_type": scorer_llm_name.lower(),
            "temperature": 0.0,
            "batch_size": 1,
        }
        scorer_llm_dict.update(scorer_finetuned_palm_dict)
        call_scorer_server_func = call_scorer_finetuned_palm_server_func
        scorer_llm_dict["call_server_func"] = call_scorer_server_func
    else:
        assert scorer_llm_name.lower() in {"gpt-3.5-turbo", "gpt-4"}

        scorer_llm_dict = {
            "temperature": 0.0,
            "name": scorer_llm_name,
            "model_type": scorer_llm_name.lower(),
            "batch_size": 1,
        }
        scorer_llm_dict.update(scorer_gpt_dict)
        call_scorer_server_func = functools.partial(
            prompt_utils.call_openai_server_func,
            model=scorer_llm_name.lower(),
            max_decode_steps=scorer_gpt_dict["max_decode_steps"],
            temperature=0.0,
        )
        scorer_llm_dict["call_server_func"] = call_scorer_server_func
    

    # ====================== optimizer model configs ============================
    if optimizer_llm_name.lower() == "text-bison":
        # when prompting text-bison with Cloud API

        call_optimizer_finetuned_palm_server_func = functools.partial(
            prompt_utils.call_palm_server_from_cloud,
            model="text-bison-001",
            temperature=1.0,
            max_decode_steps=optimizer_finetuned_palm_dict["max_decode_steps"],
        )

        optimizer_llm_dict = {
            "model_type": optimizer_llm_name.lower(),
            "temperature": 0.8,
            "name": "text-bison",
            "batch_size": 1,
        }
        optimizer_llm_dict.update(optimizer_finetuned_palm_dict)
        call_optimizer_server_func = call_optimizer_finetuned_palm_server_func
        optimizer_llm_dict["call_optimizer_server_func"] = call_optimizer_server_func

    else:
        assert optimizer_llm_name in {"gpt-3.5-turbo", "gpt-4"}
        optimizer_llm_dict["name"] = optimizer_llm_name
        call_optimizer_server_func = functools.partial(
            prompt_utils.call_openai_server_func,
            model=optimizer_llm_name,
            max_decode_steps=optimizer_llm_dict["max_decode_steps"],
            temperature=0.8,
        )
        optimizer_llm_dict["call_optimizer_server_func"] = call_optimizer_server_func

    # ====================== read data ============================
    print("\n================ prompt optimization settings ==============")
    # from https://github.com/hendrycks/test/blob/master/categories.py

    raw_data = pd.DataFrame()
    f_gsm = os.path.join(root_data_folder_path, f"gsm_{task_name}.tsv")
    single_task_df = pd.read_csv(f_gsm, sep="\t", header=None)
    raw_data = pd.concat([raw_data, single_task_df])

    train_ratio = 0.7  # 0.035
    eval_ratio = 0.1

    # ========== set other optimization experiment hyperparameters ==============

    initial_instructions = [
        "Let's solve the problem.",
        # "",
        # "The answer is",
    ]

    # ===================== run prompt optimization ======================

    evolution_kwargs = {
        "model_config": {
            "num_search_steps": 10,  # 200
            "train_ratio": train_ratio,
            "eval_ratio": eval_ratio,
            "eval_interval": 3,  # every this number of steps, compute the accuracies of current-step instructions on the validation set
            "optimizer_llm_temperature_schedule": "constant",
            "optimizer_llm_temperature_end": 1.0,
        },
        "raw_data": raw_data,
        "instruction_config": {
            "max_num_instructions": 20,
            "num_score_buckets": 100,
            "initial_instructions": initial_instructions,
            "instruction_type": "text",
            "num_generated_instructions_in_each_step": 4,  # 8
            "num_few_shot_questions_for_instruction_refinement": 3,
            "old_instruction_score_threshold": (
                0.15 if scorer_llm_name == "text-bison" else 0.3
            ),
            "include_qa": False,
            "few_shot_qa_pairs": True,
            "few_shot_selection_criteria": "random",  # one of {'current_most_frequent', 'random'}
        },
        "scorer_llm_dict": scorer_llm_dict,
        "optimizer_llm_dict": optimizer_llm_dict,
        "save_folder": {
            "save_folder": save_folder,
            "result_by_instruction_folder": result_by_instruction_folder,
        }
    }

    opt_utils.run_evolution(**evolution_kwargs)


if __name__ == "__main__":
    app.run(main)
