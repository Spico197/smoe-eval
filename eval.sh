#!/usr/bin/bash

#SBATCH --job-name=eval-moe-fpt-20k
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log

#SBATCH --partition=MoE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
##SBATCH --mem=0
#SBATCH -x SH-IDCA1404-10-140-54-116

#SBATCH --nodes=1
#SBATCH --gres=gpu:8

source ~/anaconda3/bin/activate smoe

num_nodes=1         # should match with --nodes
num_gpu_per_node=8  # should match with --gres

# #cpu/#num_gpu_per_node
export OMP_NUM_THREADS=4
export NCCL_DEBUG=INFO
export LOGLEVEL=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export TORCH_SHOW_CPP_STACKTRACES=1
# export CUDA_LAUNCH_BLOCKING=1

{
    # model_type="hf-causal-experimental"
    # out_name="llama_reproduction"
    # model_dir="/mnt/petrelfs/share_data/quxiaoye/models/llama_7B"

    # model_type="llama-moe-causal"
    # out_name="16select4_woCPT"
    # model_dir="/mnt/petrelfs/share_data/quxiaoye/models/llama_7B_MoE_16Select4-l2_norm"

    model_type="llama-moe-causal"
    out_name="clustering-l2-l2_norm-llama2-7b-8select2-gate_proj"
    model_dir="/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Graph-l2_norm-l2_norm/llama_7B-16Select4-up_proj"
    # model_dir="/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-test/Clustering-l2-l2_norm/llama2_7B-8Select2-gate_proj"

    # model_type="llama-moe-causal"
    # out_name="16select4_woCPT_13B"
    # model_dir="/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-no-softmax/Clustering-l2-l2_norm/llama_13B-16Select4-gate_proj"

    # model_type="llama-moe-causal"
    # out_name="16select4_16card_bs16_checkpoint15000_onlywiki"
    # # model_dir="/mnt/petrelfs/share_data/quxiaoye/continual_train_moe_models/16select4_16card_bs16_checkpoint15000"
    # model_dir=/mnt/petrelfs/share_data/quxiaoye/continual_train_moe_models/16select4_16card_bs16_checkpoint15000_onlywiki

    # model_type="llama-moe-causal"
    # out_name="16select4_64card_bs16_2_checkpoint12000"
    # model_dir="/mnt/petrelfs/share_data/quxiaoye/continual_train_moe_models/16select4_64card_bs16_2_checkpoint12000"
    # out_name="16select4_64card_bs16_2_checkpoint20000"
    # model_dir="/mnt/petrelfs/zhutong/smoe/outputs/cpt-moe-fpt-64gpus-bs16_2-zero1default-1600316/checkpoint-20000"
    # out_name="16select4_64card_bs16_2_checkpoint22000"
    # model_dir="/mnt/petrelfs/zhutong/smoe/outputs/cpt-moe-fpt-64gpus-bs16_2-zero1default-1600316/checkpoint-22000"
    # out_name="16select4_64card_bs16_2_checkpoint23000"
    # model_dir="/mnt/petrelfs/zhutong/smoe/outputs/cpt-moe-fpt-64gpus-bs16_2-zero1default-1600316/checkpoint-23000"

    # --------------------------------------------------------------------------------------------------------------------

    model_type="llama-moe-causal"
    model_dir=$1
    out_name=$(python -c "import sys; print('-'.join(sys.argv[1].split('/')[-2:]))" $model_dir)

    task_type=$2

    case $task_type in 
        "mmlu")        
            task_name="mmlu-5shot"
            tasks="hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions"
            fewshot=5
            ;;
        "arc")
            task_name="arc_challenge-25shot"
            tasks="arc_challenge"
            fewshot=25
            ;;
        "hellaswag")
            task_name="hellaswag-10shot"
            tasks="hellaswag"
            fewshot=10
            ;;
        "triviaqa")
            task_name="triviaqa-5shot"
            tasks="triviaqa"
            fewshot=5
            ;;
        "truthfulqa")
            task_name="truthfulqa-0shot"
            tasks="truthfulqa_mc"
            fewshot=0
            ;;
        *)
            echo "$task_type task not supported!"
            exit 1

    # --------------------------------------------------------------------------------------------------------------------

    # task_name="mmlu-5shot"
    # tasks="hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions"
    # fewshot=5

    # task_name="arc_challenge-25shot"
    # tasks="arc_challenge"
    # fewshot=25

    # task_name="gsm8k-8shot"
    # tasks="gsm8k"
    # fewshot=8

    # task_name="hellaswag-10shot"
    # tasks="hellaswag"
    # fewshot=10

    # task_name="triviaqa-5shot"
    # tasks="triviaqa"
    # fewshot=5

    # task_name="truthfulqa-0shot"
    # tasks="truthfulqa_mc"
    # fewshot=0

    # --------------------------------------------------------------------------------------------------------------------

    python main.py \
        --model=${model_type} \
        --model_args="pretrained=${model_dir},use_accelerate=True" \
        --tasks=${tasks} \
        --num_fewshot=${fewshot} \
        --batch_size=2 \
        --no_cache \
        --output_path="results/${out_name}/${task_name}"
}
