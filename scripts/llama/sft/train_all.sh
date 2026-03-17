#! /bin/bash

seeds="10"
gpus=4
base_path=$(pwd)
port=3030

# # TinyLlama Model
for seed in $seeds
do
    echo "Running for seed ${seed}"
    CMD="bash scripts/llama/sft/sft_1B.sh ${base_path} ${port} ${gpus} lm ${seed}"
    echo Running command:{$CMD}
    $CMD
done