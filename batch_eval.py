import time
import subprocess
from pathlib import Path

from loguru import logger

from listen_eval import SETTINGS


def run_command(command, wait=True):
    logger.info(f"Running command: {command}")
    proc = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    pid = proc.pid
    if wait:
        proc.wait()
    return pid


def download_to_cache(remote_url, cache_dir):
    # aws s3 cp s3://moe_checkpoints/zhutong/models/sheared_fluency_8_2/3740/ models/sheared_fluency_8_2/3740/ --recursive --exclude *global_step*
    # logger.info(f"downloading ckpt from {remote_url} to {cache_dir}")
    # run_command(f"aws s3 cp {remote_url} {cache_dir} --recursive")
    # logger.info("downloaded")
    raise NotImplementedError("buggy downloading")


def eval_one(
    abbr: str,
    folder: str,
    partition: str = "MoE",
    tasks: list[str] = ["arc", "hellaswag"],
    model_type: str = "llama-moe-causal",  # llama-moe-causal, hf-causal-experimental
    results_folder: str = "results",
    log_dir: str = "logs",
    batch_size: int = 2,
):
    results_folder = Path(results_folder)
    log_dir_p = Path(log_dir)
    results_folder.mkdir(exist_ok=True, parents=True)
    log_dir_p.mkdir(exist_ok=True, parents=True)

    out_path_list = []
    for task in tasks:
        out_path = results_folder / f"{abbr}-{task}.json"
        out_path_list.append(out_path)

        logger.info(f"Evaluating task {task} for {abbr}, folder: {str(folder)}")
        logger.info(f"Dest path: {str(out_path)}")

        cmd_args = [
            f"--model='{model_type}'",
            f"--model_args='pretrained={str(folder)},use_accelerate=True'",
            f"--tasks='{SETTINGS[task]['tasks']}'",
            f"--num_fewshot={SETTINGS[task]['fewshot']}",
            f"--batch_size={batch_size}",
            "--no_cache",
            f"--output_path='{str(out_path)}'",
            "--device='cuda:0'",
        ]
        log_path = f"{log_dir}/{abbr}-{task}.log"
        run_command(
            f"nohup srun -p {partition} -n1 -N1 --gres=gpu:1 --quotatype=auto "
            + f"--output={log_path} "
            + f"--error={log_path} "
            + "python main.py "
            + " ".join(cmd_args)
            + f" 1>{log_path} 2>&1 &"
        )

    # while len(out_path_list) > 0:
    #     out_path: Path = out_path_list.pop(0)
    #     while not out_path.exists():
    #         time.sleep(60)
    #     logger.info(f"Task {out_path.name} finished, results saved to {str(out_path)}")


def remove_dir(folder: str):
    if folder.startswith("/") or folder.startswith("~"):
        raise ValueError("folder should be relative path")
    run_command(f"rm -r {folder}")


if __name__ == "__main__":
    cache_dir = "models/cache"
    tasks = [
        {
            # "model_dir": "s3://moe_checkpoints/zhutong/models/sheared_fluency_8_2/340",
            "model_dir": "models/sheared_fluency_8_2/340",
            "task": "arc,hellaswag",
        },
        {
            # "model_dir": "s3://moe_checkpoints/zhutong/models/sheared_fluency_8_2/3740",
            "model_dir": "models/sheared_fluency_8_2/3740",
            "task": "hellaswag",
        },
        {
            # "model_dir": "s3://moe_checkpoints/zhutong/models/sheared_fluency_8_2/9180",
            "model_dir": "models/sheared_fluency_8_2/9180",
            "task": "arc,hellaswag",
        },
        {
            # "model_dir": "s3://moe_checkpoints/zhutong/models/sheared_fluency_8_2/10540",
            "model_dir": "models/sheared_fluency_8_2/10540",
            "task": "arc",
        },
        {
            # "model_dir": "s3://moe_checkpoints/zhutong/models/sheared_fluency_8_2/13260",
            "model_dir": "models/sheared_fluency_8_2/13260",
            "task": "arc,hellaswag",
        },
    ]
    for task in tasks:
        # download_to_cache(task["model_dir"], cache_dir)
        model_dir_p = Path(task["model_dir"])
        # sheared_fluency_8_2-9180
        abbr = f"{model_dir_p.parent.name}-{model_dir_p.name}"
        eval_one(abbr, str(model_dir_p), tasks=task["task"].split(","))
        # remove_dir(cache_dir)
