#! /bin/bash

seeds="10" #"10 20 30 40 50"
types="fkl" #"fkl rkl fkl+rkl akl"
gpus=4
base_path=$(pwd)
port=2014

model="medium"  # "base" or "medium"

for seed in $seeds
do
    echo "Running for seed ${seed}"
    for type in $types
    do
        CMD="bash scripts/gpt2/seqkd/seqkd_${model}.sh ${base_path} 2020 ${gpus} ${type} ${seed}"
        echo Running command: $CMD
        $CMD
    done
done