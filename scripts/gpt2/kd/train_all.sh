#! /bin/bash

seeds="10" #"10 20 30 40 50"
types="fkl" #fkl rkl fkl+rkl akl jsd
gpus=4
base_path=$(pwd)
port=2014

model="medium"  # "base" or "medium"

# GPT-2 Model
for seed in $seeds
do
    echo "Running for seed ${seed}"
    for type in $types
    do
        CMD="bash scripts/gpt2/kd/train_${model}.sh ${base_path} ${port} ${gpus} ${type} ${seed}"
        echo Running command:{$CMD}
        $CMD
    done
done