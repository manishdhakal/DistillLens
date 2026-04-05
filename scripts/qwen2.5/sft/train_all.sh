#! /bin/bash

seeds="10"
gpus=4
base_path=$(pwd)
port=3030

model="1.5B" # '1.5B', '3B'

# # TinyLlama Model
for seed in $seeds
do
    echo "Running for seed ${seed}"
    CMD="bash scripts/qwen2.5/sft/sft_${model}.sh ${base_path} ${port} ${gpus} lm ${seed}"
    echo Running command:{$CMD}
    $CMD
done