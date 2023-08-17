#!/bin/bash

# List of directories
directories=(
    "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-l2-l2_norm/llama2_7B-16Select4-up_proj"
    "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-l2-l2_norm/llama2_7B-8Select2-up_proj"
    "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-l2-l2_norm/llama_7B-16Select4-up_proj"
    "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-l2-l2_norm/llama_7B-8Select2-up_proj"
    "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Random-l2_norm/llama2_7B-16Select4-up_proj"
    "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Random-l2_norm/llama2_7B-8Select2-up_proj"
    "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Random-l2_norm/llama_7B-8Select2-up_proj"
    "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Random-l2_norm/llama_7B-16Select4-up_proj"
    "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Graph-l2_norm-l2_norm/llama2_7B-16Select4-gate_proj"
    "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Graph-l2_norm-l2_norm/llama2_7B-8Select2-gate_proj"
    "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Graph-l2_norm-l2_norm/llama_7B-8Select2-up_proj"
    "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Graph-l2_norm-l2_norm/llama2_7B-16Select4-up_proj"
    "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-cos-l2_norm/llama2_7B-8Select2-up_proj"
    "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-cos-l2_norm/llama_7B-8Select2-up_proj"
    "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-cos-l2_norm/llama_7B-16Select4-up_proj"
    "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-cos-l2_norm/llama2_7B-16Select4-up_proj"
    "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Graph-l1_norm-l1_norm/llama2_7B-8Select2-up_proj"
    "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Graph-l1_norm-l1_norm/llama2_7B-8Select2-gate_proj"
    "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Graph-l1_norm-l1_norm/llama_7B-8Select2-up_proj"
    "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Graph-l1_norm-l1_norm/llama2_7B-16Select4-gate_proj"
    "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Graph-l1_norm-l1_norm/llama_7B-16Select4-up_proj"
    "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Graph-l1_norm-l1_norm/llama2_7B-16Select4-up_proj"
)

tokenizer_files=(
    "tokenizer_config.json"
    "tokenizer.model"
    "special_tokens_map.json"
    "tokenizer.json"
)

tokenizer_dir="/mnt/petrelfs/share_data/quxiaoye/models/llama_7B"

# Loop through the directories and copy 'a.py' into them
for dir in "${directories[@]}"; do
    for token_file in "${tokenizer_files[@]}"; do
        cp "$tokenizer_dir/$token_file" "$dir/"
    done
done
