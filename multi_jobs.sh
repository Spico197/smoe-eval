folders=(
    # "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-l2-l2_norm/llama2_7B-16Select4-up_proj"
    # "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-l2-l2_norm/llama2_7B-8Select2-up_proj"
    # "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-l2-l2_norm/llama_7B-16Select4-up_proj"
    # "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-l2-l2_norm/llama_7B-8Select2-up_proj"
    # "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Random-l2_norm/llama2_7B-16Select4-up_proj"
    # "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Random-l2_norm/llama2_7B-8Select2-up_proj"
    # "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Random-l2_norm/llama_7B-8Select2-up_proj"
    # "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Random-l2_norm/llama_7B-16Select4-up_proj"
    # "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Graph-l2_norm-l2_norm/llama2_7B-16Select4-gate_proj"
    # "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Graph-l2_norm-l2_norm/llama2_7B-8Select2-gate_proj"
    # "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Graph-l2_norm-l2_norm/llama_7B-8Select2-up_proj"
    # "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Graph-l2_norm-l2_norm/llama2_7B-16Select4-up_proj"
    # "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-cos-l2_norm/llama2_7B-8Select2-up_proj"
    # "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-cos-l2_norm/llama_7B-8Select2-up_proj"
    # "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-cos-l2_norm/llama_7B-16Select4-up_proj"
    # "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-cos-l2_norm/llama2_7B-16Select4-up_proj"
    # "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Graph-l1_norm-l1_norm/llama2_7B-8Select2-up_proj"
    # "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Graph-l1_norm-l1_norm/llama2_7B-8Select2-gate_proj"
    # "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Graph-l1_norm-l1_norm/llama_7B-8Select2-up_proj"
    # "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Graph-l1_norm-l1_norm/llama2_7B-16Select4-gate_proj"
    # "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Graph-l1_norm-l1_norm/llama_7B-16Select4-up_proj"
    # "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Graph-l1_norm-l1_norm/llama2_7B-16Select4-up_proj"
    # "/mnt/petrelfs/zhutong/smoe/outputs/random_16select4_moe"
    # "/mnt/petrelfs/zhutong/smoe/outputs/random_gate_16select4_moe"
    # "/mnt/petrelfs/zhutong/smoe/outputs/cpt-moe-fpt-7b-random-64gpus-bs16_2-zero1default-1708772/checkpoint-2000"
    "outputs/cpt-moe-fpt-7b-random-64gpus-bs16_2-zero1default-1708772/checkpoint-3000"
)

tasks=(
    "arc"
    "truthfulqa"
    "hellaswag"
    "gsm8k"
    "mmlu"
)

num_jobs=$(echo "${#folders[@]} * ${#tasks[@]}" | bc)

i=0
for model_dir in ${folders[@]}; do
    for task_type in ${tasks[@]}; do
        job_id=$(sbatch eval.sh "$model_dir" "$task_type")
        i=$((i + 1))
        echo "Job $i/$num_jobs - $job_id - $task_type: $model_dir"
        sleep 1
    done
done

# Submitted batch job 1665102
# Submitted 1: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-l2-l2_norm/llama2_7B-16Select4-up_proj
# Submitted batch job 1665103
# Submitted 2: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-l2-l2_norm/llama2_7B-8Select2-up_proj
# Submitted batch job 1665104
# Submitted 3: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-l2-l2_norm/llama_7B-16Select4-up_proj
# Submitted batch job 1665105
# Submitted 4: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-l2-l2_norm/llama_7B-8Select2-up_proj
# Submitted batch job 1665106
# Submitted 5: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Random-l2_norm/llama2_7B-16Select4-up_proj
# Submitted batch job 1665107
# Submitted 6: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Random-l2_norm/llama2_7B-8Select2-up_proj
# Submitted batch job 1665108
# Submitted 7: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Random-l2_norm/llama_7B-8Select2-up_proj
# Submitted batch job 1665109
# Submitted 8: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Random-l2_norm/llama_7B-16Select4-up_proj
# Submitted batch job 1665110
# Submitted 9: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Graph-l2_norm-l2_norm/llama2_7B-16Select4-gate_proj
# Submitted batch job 1665111
# Submitted 10: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Graph-l2_norm-l2_norm/llama2_7B-8Select2-gate_proj
# Submitted batch job 1665112
# Submitted 11: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Graph-l2_norm-l2_norm/llama_7B-8Select2-up_proj
# Submitted batch job 1665113
# Submitted 12: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Graph-l2_norm-l2_norm/llama2_7B-16Select4-up_proj
# Submitted batch job 1665114
# Submitted 13: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-cos-l2_norm/llama2_7B-8Select2-up_proj
# Submitted batch job 1665115
# Submitted 14: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-cos-l2_norm/llama_7B-8Select2-up_proj
# Submitted batch job 1665116
# Submitted 15: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-cos-l2_norm/llama_7B-16Select4-up_proj
# Submitted batch job 1665117
# Submitted 16: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-cos-l2_norm/llama2_7B-16Select4-up_proj
# Submitted batch job 1665118
# Submitted 17: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Graph-l1_norm-l1_norm/llama2_7B-8Select2-up_proj
# Submitted batch job 1665119
# Submitted 18: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Graph-l1_norm-l1_norm/llama2_7B-8Select2-gate_proj
# Submitted batch job 1665120
# Submitted 19: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Graph-l1_norm-l1_norm/llama_7B-8Select2-up_proj
# Submitted batch job 1665121
# Submitted 20: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Graph-l1_norm-l1_norm/llama2_7B-16Select4-gate_proj
# Submitted batch job 1665122
# Submitted 21: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Graph-l1_norm-l1_norm/llama_7B-16Select4-up_proj
# Submitted batch job 1665123
# Submitted 22: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Graph-l1_norm-l1_norm/llama2_7B-16Select4-up_proj

# Submitted 1/8 - Submitted batch job 1707364 - arc: /mnt/petrelfs/zhutong/smoe/outputs/random_16select4_moe
# Submitted 2/8 - Submitted batch job 1707365 - truthfulqa: /mnt/petrelfs/zhutong/smoe/outputs/random_16select4_moe
# Submitted 3/8 - Submitted batch job 1707366 - hellaswag: /mnt/petrelfs/zhutong/smoe/outputs/random_16select4_moe
# Submitted 4/8 - Submitted batch job 1707367 - mmlu: /mnt/petrelfs/zhutong/smoe/outputs/random_16select4_moe
# Submitted 5/8 - Submitted batch job 1707368 - arc: /mnt/petrelfs/zhutong/smoe/outputs/random_gate_16select4_moe
# Submitted 6/8 - Submitted batch job 1707369 - truthfulqa: /mnt/petrelfs/zhutong/smoe/outputs/random_gate_16select4_moe
# Submitted 7/8 - Submitted batch job 1707370 - hellaswag: /mnt/petrelfs/zhutong/smoe/outputs/random_gate_16select4_moe
# Submitted 8/8 - Submitted batch job 1707371 - mmlu: /mnt/petrelfs/zhutong/smoe/outputs/random_gate_16select4_moe

# Job 1/4 - Submitted batch job 1710206 - arc: /mnt/petrelfs/zhutong/smoe/outputs/cpt-moe-fpt-7b-random-64gpus-bs16_2-zero1default-1708772/checkpoint-2000
# Job 2/4 - Submitted batch job 1710207 - truthfulqa: /mnt/petrelfs/zhutong/smoe/outputs/cpt-moe-fpt-7b-random-64gpus-bs16_2-zero1default-1708772/checkpoint-2000
# Job 3/4 - Submitted batch job 1710208 - hellaswag: /mnt/petrelfs/zhutong/smoe/outputs/cpt-moe-fpt-7b-random-64gpus-bs16_2-zero1default-1708772/checkpoint-2000
# Job 4/4 - Submitted batch job 1710209 - mmlu: /mnt/petrelfs/zhutong/smoe/outputs/cpt-moe-fpt-7b-random-64gpus-bs16_2-zero1default-1708772/checkpoint-2000

# Job 1/5 - Submitted batch job 1710964 - arc: outputs/cpt-moe-fpt-7b-random-64gpus-bs16_2-zero1default-1708772/checkpoint-3000
# Job 2/5 - Submitted batch job 1710965 - truthfulqa: outputs/cpt-moe-fpt-7b-random-64gpus-bs16_2-zero1default-1708772/checkpoint-3000
# Job 3/5 - Submitted batch job 1710966 - hellaswag: outputs/cpt-moe-fpt-7b-random-64gpus-bs16_2-zero1default-1708772/checkpoint-3000
# Job 4/5 - Submitted batch job 1710967 - gsm8k: outputs/cpt-moe-fpt-7b-random-64gpus-bs16_2-zero1default-1708772/checkpoint-3000
# Job 5/5 - Submitted batch job 1710968 - mmlu: outputs/cpt-moe-fpt-7b-random-64gpus-bs16_2-zero1default-1708772/checkpoint-3000
