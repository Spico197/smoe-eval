import re
import subprocess
import time
import argparse
from pathlib import Path


def get_latest_ckpt_folder(folder, regex, ignore=None):
    """
    Check if there is a new checkpoint dir and the model is ready to be loaded.
    """

    # get all checkpoint dirs
    checkpoint_dirs = [str(d) for d in Path(folder).glob("*") if re.search(regex, str(d))]

    # filter out ignored dirs
    if ignore is not None:
        ignore = re.compile(ignore)
        checkpoint_dirs = [d for d in checkpoint_dirs if not ignore.search(d)]

    if len(checkpoint_dirs) == 0:
        return None

    # sort by number
    checkpoint_dirs = sorted(checkpoint_dirs, key=lambda d: int(re.search(r"\d+", d).group(0)))

    # get the latest checkpoint dir
    latest_checkpoint_dir = checkpoint_dirs[-1]

    return latest_checkpoint_dir


def submit_eval(ckpt_folder, args):
    ret_string = subprocess.check_output(f"python main.py --model llama-moe-causal --model_args \"{ckpt_folder}\" --tasks {args.tasks} --batch_size 1 --output_path {args.output_filepath}").read()



def main(args):
    visited = set()

    while True:
        latest_ckpt_folder = get_latest_ckpt_folder(args.watch_folder, args.regex_for_checkpoint_dir, args.ignore)
        time.sleep(args.time_interval)
        if latest_ckpt_folder not in visited:
            job_id = submit_eval(latest_ckpt_folder, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("watch_folder")
    parser.add_argument("-r", "--regex-for-checkpoint-dir", default="checkpoint-\\d+")
    parser.add_argument("-i", "--ignore", default=None, help="Regex for model folders to ignore")
    parser.add_argument("-t", "--tokenizer-dir", default=None, help="Path to tokenizer dir. If not set, use checkpoint dirs.")
    parser.add_argument("-m", "--last-dir", default=None, help="Path to the last model ckpt directory. This is used for determing when to stop evaluation.")
    parser.add_argument("--tasks", default="arc,mmlu")
    parser.add_argument("--time-interval", default=300, type=int, help="Time interval in seconds to check for new checkpoints.")
    parser.add_argument("-o", "--output-filepath", default=None, help="Filepath to write eval results to.")
    args = parser.parse_args()

    if args.output_filepath is None:
        args.output_filepath = Path(args.watch_folder) / "eval_results.txt"

    main(args)
