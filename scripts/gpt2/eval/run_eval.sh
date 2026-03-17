gpus=${1-4}
base_path=$(pwd)

approaches=("distill_lens") #("kd" "distill_lens")
types=("rkl") #("fkl" "rkl" "fkl+rkl" "akl" "jsd")
datas=("dolly" "self_inst" "vicuna" "sinst" "uinst") #("dolly" "self_inst" "vicuna" "sinst" "uinst")
seeds=(10 20 30 40 50) #(10 20 30 40 50)
port=4041

CKPT_NAME="gpt2-base"   # "gpt2-base" or "gpt2-medium"

for seed in ${seeds[@]}
do
    for approach in ${approaches[@]}
    do
        for type in ${types[@]}
        do
            for data in ${datas[@]}
            do
                cmd="bash ${base_path}/scripts/gpt2/eval/eval_main_${data}.sh ${base_path} ${port} ${gpus} ${CKPT_NAME} ${approach} ${type} ${seed} --eval-batch-size 32"
                echo "$cmd"
                eval "$cmd"
            done
        done
    done
done