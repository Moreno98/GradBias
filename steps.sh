# Make sure to have coco_train.json under proposed_biases/coco/3 folder before running these scripts.
# To save you time, make sure to have the synonyms.json file under data/. Otherwise, this file will be automatically generated (it may take a while).

# Syntax tree baseline
CUDA_VISIBLE_DEVICES=0 python syntax_tree_baseline.py

# Answer ranking with LLM
# Available LLMs: 
# llama2-7B: requires 1 GPU and --nproc_per_node 1
# llama2-13B: requires 2 GPUs and --nproc_per_node 2
# llama3-8B: requires 1 GPU and --nproc_per_node 1
# llama3-70B: requires 8 GPUs and --nproc_per_node 8
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 answer_ranking_LLM.py --LLM llama3-8B --seed 0 --dataset coco