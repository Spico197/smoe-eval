import json
import argparse
import statistics as sts
from pathlib import Path
from typing import List, Dict, Optional, Union


RESULT_FILENAMES_AND_KEYS = [["arc_challenge-25shot", "acc_norm"], ["hellaswag-10shot", "acc_norm"], ["mmlu-5shot", "acc_norm"], ["truthfulqa-0shot", "mc2"]]
# RESULT_FILENAMES_AND_KEYS = [["arc_challenge-25shot", "acc_norm"], ["hellaswag-10shot", "acc_norm"], ["mmlu-5shot", "acc_norm"], ["truthfulqa-0shot", "mc2"], ["gsm8k-8shot", "acc"]]


def load_json(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)
    return data


def calc_mean(scores: Union[List[Dict[str, float]], List[float]], key: Optional[str] = "acc_norm") -> float:
    if key:
        mean = sts.mean([v[key] for v in scores])
    else:
        mean = sts.mean(scores)
    return mean


def main(args):
    if args.mmlu_result_filepath:
        data = load_json(args.mmlu_result_filepath)
        mean_score = calc_mean(data["results"].values())
        print(f"Mean acc_norm: {mean_score:.4f}")
    if args.results_dir:
        folder = Path(args.results_dir)
        scores = []
        for filename, key in RESULT_FILENAMES_AND_KEYS:
            filepath = folder / filename
            if filepath.exists():
                results = load_json(filepath)
                mean_score = calc_mean(results["results"].values(), key=key)
                scores.append(mean_score)
                print(f"{filename}: {100 * mean_score:.2f} %")
        print(f"Average: {100 * sts.mean(scores):.2f} %")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmlu_result_filepath", "-f")
    parser.add_argument("--results_dir", "-d")
    args = parser.parse_args()
    main(args)
