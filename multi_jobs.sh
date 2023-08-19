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
    "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Random-l2_norm/llama2_7B-8Select2-gate_proj"
    "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Random-l2_norm/llama2_7B-16Select4-gate_proj"
    "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-l2-l2_norm/llama2_7B-8Select2-gate_proj"
    "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-l2-l2_norm/llama2_7B-16Select4-gate_proj"
    "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-cos-l2_norm/llama2_7B-8Select2-gate_proj"
    "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-cos-l2_norm/llama2_7B-16Select4-gate_proj"
)

i=0
for model_dir in ${folders[@]}; do
    sbatch eval.sh "$model_dir"
    i=$((i + 1))
    echo "Submitted $i: $model_dir"
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

# Submitted batch job 1666420
# Submitted 1: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-l2-l2_norm/llama_7B-16Select4-up_proj
# Submitted batch job 1666421
# Submitted 2: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-l2-l2_norm/llama_7B-8Select2-up_proj
# Submitted batch job 1666422
# Submitted 3: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Random-l2_norm/llama_7B-8Select2-up_proj
# Submitted batch job 1666423
# Submitted 4: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Random-l2_norm/llama_7B-16Select4-up_proj
# Submitted batch job 1666424
# Submitted 5: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Graph-l2_norm-l2_norm/llama_7B-8Select2-up_proj
# Submitted batch job 1666425
# Submitted 6: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-cos-l2_norm/llama_7B-8Select2-up_proj
# Submitted batch job 1666426
# Submitted 7: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-cos-l2_norm/llama_7B-16Select4-up_proj
# Submitted batch job 1666427
# Submitted 8: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Graph-l1_norm-l1_norm/llama_7B-8Select2-up_proj
# Submitted batch job 1666428
# Submitted 9: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Graph-l1_norm-l1_norm/llama_7B-16Select4-up_proj

# Submitted batch job 1666817
# Submitted 1: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-l2-l2_norm/llama_7B-16Select4-up_proj
# Submitted batch job 1666818
# Submitted 2: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-l2-l2_norm/llama_7B-8Select2-up_proj
# Submitted batch job 1666819
# Submitted 3: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Random-l2_norm/llama_7B-8Select2-up_proj
# Submitted batch job 1666820
# Submitted 4: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Random-l2_norm/llama_7B-16Select4-up_proj
# Submitted batch job 1666821
# Submitted 5: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Graph-l2_norm-l2_norm/llama_7B-8Select2-up_proj
# Submitted batch job 1666822
# Submitted 6: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-cos-l2_norm/llama_7B-8Select2-up_proj
# Submitted batch job 1666823
# Submitted 7: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-cos-l2_norm/llama_7B-16Select4-up_proj
# Submitted batch job 1666824
# Submitted 8: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Graph-l1_norm-l1_norm/llama_7B-8Select2-up_proj
# Submitted batch job 1666825
# Submitted 9: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Graph-l1_norm-l1_norm/llama_7B-16Select4-up_proj

# Submitted batch job 1667517
# Submitted 1: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Random-l2_norm/llama2_7B-8Select2-gate_proj
# Submitted batch job 1667518
# Submitted 2: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Random-l2_norm/llama2_7B-16Select4-gate_proj
# Submitted batch job 1667519
# Submitted 3: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-l2-l2_norm/llama2_7B-8Select2-gate_proj
# Submitted batch job 1667520
# Submitted 4: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-l2-l2_norm/llama2_7B-16Select4-gate_proj
# Submitted batch job 1667521
# Submitted 5: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-cos-l2_norm/llama2_7B-8Select2-gate_proj
# Submitted batch job 1667522
# Submitted 6: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-cos-l2_norm/llama2_7B-16Select4-gate_proj

# Submitted batch job 1667527
# Submitted 1: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Random-l2_norm/llama2_7B-8Select2-gate_proj
# Submitted batch job 1667528
# Submitted 2: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Random-l2_norm/llama2_7B-16Select4-gate_proj
# Submitted batch job 1667529
# Submitted 3: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-l2-l2_norm/llama2_7B-8Select2-gate_proj
# Submitted batch job 1667530
# Submitted 4: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-l2-l2_norm/llama2_7B-16Select4-gate_proj
# Submitted batch job 1667531
# Submitted 5: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-cos-l2_norm/llama2_7B-8Select2-gate_proj
# Submitted batch job 1667532
# Submitted 6: /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-cos-l2_norm/llama2_7B-16Select4-gate_proj
