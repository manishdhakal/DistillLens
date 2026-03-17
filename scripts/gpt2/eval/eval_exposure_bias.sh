#! /bin/bash

MASTER_ADDR=localhost
NNODES=1
NODE_RANK=0
BASE_PATH=${1-"/home/MiniLLM"}
MASTER_PORT=${2-2113}
GPUS_PER_NODE=${3-1}
CKPT_NAME=${4-"gpt2-base"}
APPROACH=${5-"kd"} # 'kd', 'logit_lens', mse
TYPE=${6-"fkl"} # 'fkl', 'rkl', 'fkl+rkl', or 'akl'
SEED=${7-10}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
BASE_PATH=${1-"/home/MiniLLM"}
CKPT_NAME=${4-"gpt2-base"}
CKPT="${BASE_PATH}/results/train/${CKPT_NAME}/${APPROACH}/${TYPE}/best/"
# CKPT="gpt2" # download automatically
TEACHER_MODEL_TYPE="gpt2"
TEACHER_CKPT_NAME="teacher-gpt2-1.5B"
TEACHER_CKPT="${BASE_PATH}/checkpoints/${TEACHER_CKPT_NAME}/"
# data
DATA_NAMES="dolly"
DATA_DIR="${BASE_PATH}/data/dolly"
# hp
EVAL_BATCH_SIZE=16
# runtime
SAVE_PATH="${BASE_PATH}/results/eval_exposure_bias/${CKPT_NAME}/${APPROACH}/${TYPE}/seed${SEED}/${DATA_NAMES}/"
TYPE="eval_exposure_bias"


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --model-type gpt2"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --data-names ${DATA_NAMES}"
OPTS+=" --num-workers 0"
OPTS+=" --dev-num 1000"
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
OPTS+=" --eb-sample-times 1"
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
