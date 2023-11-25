import json
import random
from pathlib import Path
from collections import defaultdict

from lm_eval.tasks import get_task_dict


def task_val():
    num_ins_per_task = 200
    mmlu_tot_num_ins = 200
    out_dir = "val_set"
    task_type_list = ["arc_challenge", "gsm8k", "hellaswag", *"hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions".split(",")]

    task_to_contents = defaultdict(list)
    tasks = get_task_dict(task_type_list)
    for name, task in tasks.items():
        if task.has_validation_docs() and len(list(task.validation_docs())):
            docs = task.validation_docs()
            src = "val"
        elif task.has_training_docs() and len(list(task.training_docs())):
            docs = task.training_docs()
            src = "train"
        elif task.has_test_docs() and len(list(task.test_docs())):
            docs = task.test_docs()
            src = "test"
        else:
            raise ValueError
        print(name, task, src)

        for idx, doc in enumerate(docs):
            if name.startswith("arc_"):
                content = task.doc_to_text(doc) + f" {doc['choices'][doc['gold']]}"
            elif name == "gsm8k":
                content = task.doc_to_text(doc) + task.doc_to_target(doc)
            elif name == "hellaswag":
                content = task.doc_to_text(doc) + f" {doc['choices'][doc['gold']]}"
            elif name.startswith("hendrycksTest-"):
                content = task.doc_to_text(doc) + f" {doc['choices'][doc['gold']]}"
            else:
                raise NotImplementedError

            task_to_contents[name].append({"task": name, "content": content})

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(exist_ok=True)
    mmlu_ins_list = []
    for name, contents in task_to_contents.items():
        if name.startswith("hendrycksTest-"):
            mmlu_ins_list.extend(contents)
        with open(out_dir_p / f"{name}.txt", "w", encoding="utf8") as f:
            for _ in range(20):
                random.shuffle(contents)
            for content in contents[:num_ins_per_task]:
                f.write(json.dumps(content, ensure_ascii=False) + "\n")

    if len(mmlu_ins_list) > 0:
        for _ in range(20):
            random.shuffle(mmlu_ins_list)
        with open(out_dir_p / "mmlu.txt", "w", encoding="utf8") as f:
            for content in mmlu_ins_list[:mmlu_tot_num_ins]:
                f.write(json.dumps(content, ensure_ascii=False) + "\n")

    """
    $ srun -p MoE python build_val_set.py
    phoenix-srun: job 2081582 queued and waiting for resources
    phoenix-srun: job 2081582 has been allocated resources
    phoenix-srun: Job 2081582 scheduled successfully!
    Current QUOTA_TYPE is [reserved], which means the job has occupied quota in RESERVED_TOTAL under your partition.
    Current PHX_PRIORITY is normal

    [2023-10-12 16:27:33,101] [INFO] [real_accelerator.py:133:get_accelerator] Setting ds_accelerator to cuda (auto detect)
    arc_challenge <lm_eval.tasks.arc.ARCChallenge object at 0x7f29e4434510> val
    gsm8k <lm_eval.tasks.gsm8k.GradeSchoolMath8K object at 0x7f2726237110> train
    hellaswag <lm_eval.tasks.hellaswag.HellaSwag object at 0x7f2726208890> val
    hendrycksTest-abstract_algebra <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f272f99e410> val
    hendrycksTest-anatomy <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f2726250a10> val
    hendrycksTest-astronomy <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f29eb879090> val
    hendrycksTest-business_ethics <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f27262b7f90> val
    hendrycksTest-clinical_knowledge <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f2726236e90> val
    hendrycksTest-college_biology <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f2726270290> val
    hendrycksTest-college_chemistry <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f272628aa10> val
    hendrycksTest-college_computer_science <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f27262af790> val
    hendrycksTest-college_mathematics <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f2726286c10> val
    hendrycksTest-college_medicine <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f2726b1f550> val
    hendrycksTest-college_physics <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f29e43acb90> val
    hendrycksTest-computer_security <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f2728be8c90> val
    hendrycksTest-conceptual_physics <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f2721e26150> val
    hendrycksTest-econometrics <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f2726bcf010> val
    hendrycksTest-electrical_engineering <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f2721e26090> val
    hendrycksTest-elementary_mathematics <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f272626ab50> val
    hendrycksTest-formal_logic <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f2721e27150> val
    hendrycksTest-global_facts <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f27262d9490> val
    hendrycksTest-high_school_biology <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f2721e464d0> val
    hendrycksTest-high_school_chemistry <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f2721e27f10> val
    hendrycksTest-high_school_computer_science <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f2721d60e50> val
    hendrycksTest-high_school_european_history <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f2721bf8790> val
    hendrycksTest-high_school_geography <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f272903c690> val
    hendrycksTest-high_school_government_and_politics <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f2726f59550> val
    hendrycksTest-high_school_macroeconomics <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f2726208f50> val
    hendrycksTest-high_school_mathematics <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f2721dae410> val
    hendrycksTest-high_school_microeconomics <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f268807f0d0> val
    hendrycksTest-high_school_physics <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f2726fa5b50> val
    hendrycksTest-high_school_psychology <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f2721dd3750> val
    hendrycksTest-high_school_statistics <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f2721dacf90> val
    hendrycksTest-high_school_us_history <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f2721dd2750> val
    hendrycksTest-high_school_world_history <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f272621cf10> val
    hendrycksTest-human_aging <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f26880cd950> val
    hendrycksTest-human_sexuality <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f2726f59a50> val
    hendrycksTest-international_law <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f26880cde10> val
    hendrycksTest-jurisprudence <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f2726f82310> val
    hendrycksTest-logical_fallacies <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f2726fc89d0> val
    hendrycksTest-machine_learning <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f2721dd1bd0> val
    hendrycksTest-management <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f2728c09f50> val
    hendrycksTest-marketing <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f2721e05d10> val
    hendrycksTest-medical_genetics <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f27262376d0> val
    hendrycksTest-miscellaneous <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f2726f59410> val
    hendrycksTest-moral_disputes <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f27262580d0> val
    hendrycksTest-moral_scenarios <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f2728c09e10> val
    hendrycksTest-nutrition <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f268806a5d0> val
    hendrycksTest-philosophy <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f2721e6b5d0> val
    hendrycksTest-prehistory <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f2688068310> val
    hendrycksTest-professional_accounting <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f2721e10d50> val
    hendrycksTest-professional_law <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f2688090510> val
    hendrycksTest-professional_medicine <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f2726242490> val
    hendrycksTest-professional_psychology <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f2721e13010> val
    hendrycksTest-public_relations <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f2726af7750> val
    hendrycksTest-security_studies <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f2721d0d210> val
    hendrycksTest-sociology <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f2726b1f290> val
    hendrycksTest-us_foreign_policy <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f2721e44a50> val
    hendrycksTest-virology <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f2726236e10> val
    hendrycksTest-world_religions <lm_eval.tasks.hendrycks_test.create_task.<locals>.HendrycksTest object at 0x7f272830f6d0> val
    """


def pt_val():
    import os
    import shutil
    from smoe.utils.io import load_jsonlines, dump_jsonlines

    src_folder = Path("/mnt/petrelfs/share_data/quxiaoye/SlimPajama_processed/")
    dest_folder = Path("/mnt/petrelfs/share_data/quxiaoye/data/llama1_7B_val_set_tokenized/")
    dest_folder.mkdir(exist_ok=True)
    num_ins_per_type = 200
    for folder in src_folder.glob("*"):
        data_type = folder.stem
        data = []
        used_data_file_to_data = {}
        # (filepath, used, total)
        used_data_files = []
        # if data is less than num_ins_per_type, then use an extra file, else save the left data back
        for file in folder.glob("*.jsonl"):
            _data = load_jsonlines(file)
            random.shuffle(_data)
            used_data_file_to_data[file] = _data
            if len(data) < num_ins_per_type:
                used_data = _data[:num_ins_per_type - len(data)]
                data.extend(used_data)
                used_data_files.append((file, len(used_data), len(_data)))
            else:
                break
        assert len(data) == num_ins_per_type
        _dest = dest_folder / f"{data_type}.jsonl"
        dump_jsonlines(data, _dest)
        print(f"dump data ({len(data)}) jsonlines: {_dest}")
        for file, used, total in used_data_files:
            if used >= total:
                os.remove(file)
                print(f"remove {file}")
            else:
                remained_data = used_data_file_to_data[file][used:]
                dump_jsonlines(remained_data, file)
                print(f"data ({used}/{total}) dump remained data ({len(remained_data)}): {file}")


if __name__ == "__main__":
    pt_val()
