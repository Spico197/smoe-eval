#!/bin/bash

# List of directories
directories=(
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
    # "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Random-l2_norm/llama2_7B-8Select2-gate_proj"
    # "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Random-l2_norm/llama2_7B-16Select4-gate_proj"
    # "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-l2-l2_norm/llama2_7B-8Select2-gate_proj"
    # "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-l2-l2_norm/llama2_7B-16Select4-gate_proj"
    # "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-cos-l2_norm/llama2_7B-8Select2-gate_proj"
    # "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-cos-l2_norm/llama2_7B-16Select4-gate_proj"
    # "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-no-softmax/Clustering-l2-l2_norm/llama_13B-16Select4-gate_proj"
    # "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-test/Clustering-l2-l2_norm/llama2_7B-8Select2-gate_proj"
    # "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Graph-l2_norm-l2_norm/llama_7B-16Select4-up_proj"
    # "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Random-l2_norm/llama_7B-16Select4-up_proj"
    
    # "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Gradient-max-l1_norm-total/llama2_7B-16Select4-688Neurons"
    # "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Gradient-max-l1_norm-total/llama2_7B-16Select4-688Neurons-Share"
    # "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Gradient-max-l1_norm-total/llama2_7B-16Select4-1376Neurons-Share"
    # "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Gradient-min-l1_norm-total/llama2_7B-16Select4-688Neurons"
    # "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Gradient-min-l1_norm-total/llama2_7B-16Select4-688Neurons-Share"
    # "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Gradient-min-l1_norm-total/llama2_7B-16Select4-1376Neurons-Share"

    # "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-l2-l2_norm/llama2_7B-16Select4-up_proj"
    # "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-cos-l2_norm/llama2_7B-16Select4-up_proj"

    # "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune/Gradient-max-l1_norm-total-feature_grad/llama2_7B-0-0.98Percent-10787Neurons"
    # "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune/Gradient-max-l1_norm-total-feature_grad/llama2_7B-0-0.95Percent-10457Neurons"
    # "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune/Gradient-max-l1_norm-total-feature_grad/llama2_7B-0-0.90Percent-9907Neurons"
    # "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune/Gradient-max-l1_norm-total-feature_grad/llama2_7B-0-0.80Percent-8806Neurons"
    # "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune/Gradient-max-l1_norm-total-feature_grad/llama2_7B-0-0.60Percent-6604Neurons"
    # "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune/Gradient-max-l1_norm-total-feature_grad/llama2_7B-0-0.40Percent-4403Neurons"
    # "/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune/Gradient-max-l1_norm-total-feature_grad/llama2_7B-0-0.20Percent-2201Neurons"
    # /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Random-l2_norm/llama2_7B-16Select4-up_proj

    # /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune/Gradient-min-l1_norm-sample-feature_change/llama_7B-0-0.60Percent-6604Neurons
    # /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune/Gradient-min-l1_norm-sample-feature_change/llama_7B-0-0.90Percent-9907Neurons
    # /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune/Gradient-min-l1_norm-sample-feature_change/llama_7B-0-0.98Percent-10787Neurons
    # /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune/Gradient-min-l1_norm-sample-feature_change/llama_7B-0-0.80Percent-8806Neurons
    # /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune/Gradient-min-l1_norm-sample-feature_change/llama_7B-0-0.95Percent-10457Neurons
    # /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune/Gradient-min-l1_norm-sample-feature_change/llama_7B-0-0.99Percent-10897Neurons

    # /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune2/Gradient-min-l1_norm-sample-feature_change/llama_7B-0-0.60Percent-6604Neurons
    # /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune2/Gradient-min-l1_norm-sample-feature_change/llama_7B-0-0.80Percent-8806Neurons
    # /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune2/Gradient-min-l1_norm-sample-feature_change/llama_7B-0-0.90Percent-9907Neurons
    # /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune2/Gradient-min-l1_norm-sample-feature_change/llama_7B-0-0.95Percent-10457Neurons
    # /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune2/Gradient-min-l1_norm-sample-feature_change/llama_7B-0-0.98Percent-10787Neurons
    # /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune2/Gradient-min-l1_norm-sample-feature_change/llama_7B-0-0.99Percent-10897Neurons

    # /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune2/Gradient-max-l1_norm-sample-feature_change/llama_7B-0-0.20Percent-2201Neurons
    # /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune2/Gradient-max-l1_norm-sample-feature_change/llama_7B-0-0.25Percent-2752Neurons
    # /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune2/Gradient-max-l1_norm-sample-feature_change/llama_7B-0-0.40Percent-4403Neurons

    # /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune2/Gradient-min-l1_norm-sample-feature_change/llama_7B-0-0.20Percent-2201Neurons
    # /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune2/Gradient-min-l1_norm-sample-feature_change/llama_7B-0-0.25Percent-2752Neurons
    # /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune2/Gradient-min-l1_norm-sample-feature_change/llama_7B-0-0.40Percent-4403Neurons
    # /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune2/Gradient-min-l1_norm-sample-feature_change/llama_7B-0-0.60Percent-6604Neurons
    # /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune2/Gradient-min-l1_norm-sample-feature_change/llama_7B-0-0.80Percent-8806Neurons
    # /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune2/Gradient-min-l1_norm-sample-feature_change/llama_7B-0-0.90Percent-9907Neurons
    # /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune2/Gradient-min-l1_norm-sample-feature_change/llama_7B-0-0.95Percent-10457Neurons
    # /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune2/Gradient-min-l1_norm-sample-feature_change/llama_7B-0-0.98Percent-10787Neurons
    # /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune2/Gradient-min-l1_norm-sample-feature_change/llama_7B-0-0.99Percent-10897Neurons

    # /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune2/Random/llama_7B-0-0.25Percent-2752Neurons
    # /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune2/Random/llama_7B-0-0.40Percent-4403Neurons
    # /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune2/Random/llama_7B-0-0.60Percent-6604Neurons
    # /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune2/Random/llama_7B-0-0.80Percent-8806Neurons
    # /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune2/Random/llama_7B-0-0.90Percent-9907Neurons
    # /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune2/Random/llama_7B-0-0.95Percent-10457Neurons
    # /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune2/Random/llama_7B-0-0.98Percent-10787Neurons
    # /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune2/Random/llama_7B-0-0.99Percent-10897Neurons

    # /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Gradient-max-l1_norm-sample-feature_change/llama_7B-16Select4-688Neurons
    # /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Gradient-max-l1_norm-sample-feature_change/llama_7B-16Select4-688Neurons-Share
    # /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Gradient-max-l1_norm-sample-feature_change/llama_7B-16Select4-1376Neurons-Share
    # /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Gradient-max-l1_norm-sample-feature_change/llama_7B-16Select4-2752Neurons-Share
    # /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Gradient-max-l1_norm-sample-feature_change/llama_7B-16Select4-5504Neurons-Share

    # /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune2/Gradient-max-l1_norm-sample-feature_change/llama_7B-0-0.20Percent-2201Neurons
    # /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune2/Gradient-max-l1_norm-sample-feature_change/llama_7B-0-0.25Percent-2752Neurons
    # /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune2/Gradient-max-l1_norm-sample-feature_change/llama_7B-0-0.40Percent-4403Neurons
    # /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune2/Gradient-max-l1_norm-sample-feature_change/llama_7B-0-0.60Percent-6604Neurons
    # /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune2/Gradient-max-l1_norm-sample-feature_change/llama_7B-0-0.80Percent-8806Neurons
    # /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune2/Gradient-max-l1_norm-sample-feature_change/llama_7B-0-0.90Percent-9907Neurons
    # /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune2/Gradient-max-l1_norm-sample-feature_change/llama_7B-0-0.95Percent-10457Neurons
    # /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune2/Gradient-max-l1_norm-sample-feature_change/llama_7B-0-0.98Percent-10787Neurons
    # /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune2/Gradient-max-l1_norm-sample-feature_change/llama_7B-0-0.99Percent-10897Neurons

    /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Gradient-max-l1_norm-sample-feature_change/llama_7B-2Select1-5504Neurons
    /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Gradient-max-l1_norm-sample-feature_change/llama_7B-2Select1-5504Neurons-Share
    /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Gradient-max-l1_norm-sample-feature_change/llama_7B-4Select1-5504Neurons-Share
    /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Gradient-max-l1_norm-sample-feature_change/llama_7B-8Select1-5504Neurons-Share
    /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Gradient-max-l1_norm-sample-feature_change/llama_7B-16Select4-5504Neurons-Share
)

tokenizer_files=(
    "tokenizer_config.json"
    "tokenizer.model"
    "special_tokens_map.json"
    "tokenizer.json"
)

tokenizer_dir="/mnt/petrelfs/share_data/quxiaoye/models/llama_7B"
# tokenizer_dir="/mnt/petrelfs/share_data/quxiaoye/models/llama2_7B"

# Loop through the directories and copy 'a.py' into them
for dir in "${directories[@]}"; do
    for token_file in "${tokenizer_files[@]}"; do
        cp "$tokenizer_dir/$token_file" "$dir/"
    done
    echo "Done: $dir"
done
