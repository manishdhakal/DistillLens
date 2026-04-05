#! /bin/bash
types="rkl" #OR "fkl rkl fkl+rkl akl jsd"

gpus=4
base_path=$(pwd)
port=2014

# # TinyLlama Model

echo "Running for seed ${seed}"
for type in $types
do
    CMD="bash scripts/qwen2.5/distill_lens/train_1.5B_3B.sh ${base_path} 2020 ${gpus} ${type}"
    echo Running command:{$CMD}
    $CMD
done