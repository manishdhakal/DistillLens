#! /bin/bash

seeds="10"
types="akl jsd" #fkl rkl fkl+rkl akl jsd

gpus=4
base_path=$(pwd)
port=2014

# # TinyLlama Model
for seed in $seeds
do
    echo "Running for seed ${seed}"
    for type in $types
    do
        CMD="bash scripts/qwen2.5/kd/kd_1.5B_3.5B.sh ${base_path} 2020 ${gpus} ${type} ${seed}"
        echo Running command:{$CMD}
        $CMD
    done
done