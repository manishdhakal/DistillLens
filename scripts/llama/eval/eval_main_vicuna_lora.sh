#! /bin/bash

MASTER_ADDR=localhost
NNODES=1
NODE_RANK=0
BASE_PATH=${1-"/home/MiniLLM"}
MASTER_PORT=${2-2113}
GPUS_PER_NODE=${3-1}
CKPT_NAME=${4-"TinyLlama-1.1B"}
APPROACH=${5-"kd"} # 'kd', 'logit_lens'
TYPE=${6-"fkl"} # 'fkl', 'rkl', 'fkl+rkl', or 'akl'
SEED=${7-10}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
CKPT="${BASE_PATH}/checkpoints/${CKPT_NAME}/"
PEFT_CKPT="${BASE_PATH}/results/train/${CKPT_NAME}/${APPROACH}/${TYPE}/seed10/best/"
# data
DATA_NAMES="vicuna"
DATA_DIR="${BASE_PATH}/data/vicuna"
# hp
EVAL_BATCH_SIZE=16
# runtime
SAVE_PATH="${BASE_PATH}/results/eval_main/${CKPT_NAME}/${APPROACH}/${TYPE}/seed${SEED}/${DATA_NAMES}/"
TYPE="eval_main"


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --model-type llama"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --data-names ${DATA_NAMES}"
OPTS+=" --num-workers 0"
OPTS+=" --dev-num -1"
OPTS+=" --data-process-workers -1"
OPTS+=" --json-data"
# hp
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --max-length 512"
OPTS+=" --max-prompt-length 256"
# runtime
OPTS+=" --do-eval"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --seed ${SEED}"
# lora
OPTS+=" --peft lora"
OPTS+=" --peft-name lora"
OPTS+=" --peft-path ${PEFT_CKPT}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_zero1_fp16.json"
OPTS+=" --type ${TYPE}"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"


export NCCL_DEBUG=""
export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/evaluate.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
