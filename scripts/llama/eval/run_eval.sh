gpus=${1-4}
base_path=$(pwd)

approaches=("distill_lens") #("kd" "distill_lens")
types=("rkl") #("fkl" "rkl" "fkl+rkl" "akl" "jsd" "lm")
datas=("dolly" "self_inst" "vicuna" "sinst" "uinst") #("dolly" "self_inst" "vicuna" "sinst" "uinst")
seeds=(10 20 30 40 50) #(10 20 30 40 50)

idx=(0 1 2 3)
port=4045

CKPT_NAME="TinyLlama-1.1B"

for seed in ${seeds[@]}
do
    for approach in ${approaches[@]}
    do
        for type in ${types[@]}
        do
            for data in ${datas[@]}
            do
                cmd="bash ${base_path}/scripts/llama/eval/eval_main_${data}.sh ${base_path} ${port} ${gpus} ${CKPT_NAME} ${approach} ${type} ${seed} --eval-batch-size 8"
                echo "$cmd"
                eval "$cmd"
            done
        done
    done
done