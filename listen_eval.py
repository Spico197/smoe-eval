"""
nohup python listen_eval.py > eval_listen_no_ad.log 2>&1 &
"""

import time
import json
import subprocess
from pathlib import Path
from typing import Tuple, List

from dotenv import load_dotenv
from loguru import logger
from smoe.utils.notification import send_to_wechat


load_dotenv()
logger.add("listen.log")


SETTINGS = {
    "mmlu": {
        "task_name": "mmlu-5shot",
        "tasks": "hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions",
        "fewshot": 5,
    },
    "arc": {
        "task_name": "arc_challenge-25shot",
        "tasks": "arc_challenge",
        "fewshot": 25,
    },
    "hellaswag": {
        "task_name": "hellaswag-10shot",
        "tasks": "hellaswag",
        "fewshot": 10,
    },
    "triviaqa": {
        "task_name": "triviaqa-5shot",
        "tasks": "triviaqa",
        "fewshot": 5,
    },
    "gsm8k": {
        "task_name": "gsm8k-4shot",
        "tasks": "gsm8k",
        "fewshot": 4,
    },
    "truthfulqa": {
        "task_name": "truthfulqa-0shot",
        "tasks": "truthfulqa_mc",
        "fewshot": 0,
    },
}


def run_command(command):
    try:
        logger.info(f"Running cmd: {command}")
        subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"An error occurred: {e}")


def listen(
    abbr: str,
    folder: str,
    tasks: List[str] = ["arc", "hellaswag"],
    evaluated: List[Tuple[str, str]] = None,
):
    """
    Args:
        evaluated: list of (ckpt_id, task) representing the evaluated tasks.
    """
    sleep_interval = 5 * 60
    batch_size = 2
    model_type = "llama-moe-causal"  # llama-moe-causal, hf-causal-experimental
    results_folder = Path(f"results/{abbr}")
    log_dir = "logs"

    notified = []
    if evaluated is None:
        evaluated = []
    for ckpt_id, task in evaluated:
        notified.append((ckpt_id, task))
    Path(log_dir).mkdir(exist_ok=True, parents=True)
    results_folder.mkdir(exist_ok=True, parents=True)
    folder = Path(folder)
    while True:
        available_folders = list(folder.glob("checkpoint-*"))
        for ckpt_folder in available_folders:
            ckpt_id = ckpt_folder.name.split("-")[-1]
            for task in tasks:
                out_path = results_folder / f"{ckpt_id}-{SETTINGS[task]['task_name']}.json"

                # notification
                if out_path.exists() and (ckpt_id, task) not in notified and (ckpt_id, task) in evaluated:
                    logger.info(f"Results detected: Task {task} for {abbr}, folder: {str(ckpt_folder)}")
                    time.sleep(3)
                    results = json.load(out_path.open("r", encoding="utf8"))
                    results_str = json.dumps(results, ensure_ascii=False, indent=2)
                    logger.info(results_str)
                    send_to_wechat(f"Abbr: {abbr}, Task: {task}, Ckpt: {ckpt_id} " + results_str)
                    notified.append((ckpt_id, task))
                if (ckpt_id, task) in evaluated:
                    continue

                logger.info(
                    f"Evaluating task {task} for {abbr}, folder: {str(ckpt_folder)}"
                )
                logger.info(
                    f"Sleep for {sleep_interval} seconds to avoid incomplete dumping"
                )
                time.sleep(sleep_interval)
                logger.info(f"Dest path: {str(out_path)}")

                cmd_args = [
                    f"--model='{model_type}'",
                    f"--model_args='pretrained={str(ckpt_folder)},use_accelerate=True'",
                    f"--tasks='{SETTINGS[task]['tasks']}'",
                    f"--num_fewshot={SETTINGS[task]['fewshot']}",
                    f"--batch_size={batch_size}",
                    "--no_cache",
                    f"--output_path='{str(out_path)}'",
                    "--device='cuda:0'",
                ]
                log_path = f"{log_dir}/{abbr}-{ckpt_id}-{task}.log"
                run_command(
                    "nohup srun -p MoE -n1 -N1 --gres=gpu:1 --quotatype=auto "
                    + f"--output={log_path} "
                    + f"--error={log_path} "
                    + "python main.py "
                    + " ".join(cmd_args)
                    + f" 1>{log_path} 2>&1 &"
                )
                evaluated.append((ckpt_id, task))

        time.sleep(sleep_interval)


if __name__ == "__main__":
    listen(
        "sheared_llama_portion_no_ad",
        "/mnt/petrelfs/share_data/quxiaoye/runs/llama2_random_scale4_112gpus_dynamic_data/outputs/cpt-llama2_random_scale4_112gpus_dynamic_data-2323339/",
        tasks=["arc", "hellaswag"],
        evaluated=[("340", "arc"), ("340", "hellaswag")],
    )
