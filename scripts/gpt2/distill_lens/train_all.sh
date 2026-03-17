#! /bin/bash

types="rkl" #OR "fkl rkl fkl+rkl akl jsd"
gpus=4
base_path=$(pwd)
port=2014

model="medium"  # "base" or "medium"

# GPT-2 Model
echo "Running for seed ${seed}"
for type in $types
do
    CMD="bash scripts/gpt2/distill_lens/train_${model}.sh ${base_path} 2020 ${gpus} ${type} ${seed}"
    echo Running command:{$CMD}
    $CMD
done