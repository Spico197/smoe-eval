python listen_eval.py \
    llama_moe_8_2 \
    /mnt/petrelfs/share_data/quxiaoye/runs/llama2_random_split_112gpus_8_2/outputs/cpt-llama2_random_split_112gpus_8_2-2377343/ \
    -p MoE \
    --model_type llama-moe-causal \
    --tasks "sciq,piqa,winogrande,arc_e,logiqa,boolq,lambada,mmlu" \
    --evaluated "13260,sciq#13260,piqa#13260,winogrande#13260,arc_e#13260,logiqa#13260,boolq#13260,lambada#13260,mmlu" \
    --run_eval
