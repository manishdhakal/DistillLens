BASE_PATH=${1}

export TF_CPP_MIN_LOG_LEVEL=3

PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
    --data-dir ${BASE_PATH}/results/llama/gen/ \
    --processed-data-dir ${BASE_PATH}/processed_data/dolly/pseudo \
    --model-path ${BASE_PATH}/checkpoints/TinyLlama-1.1B \
    --data-process-workers 32 \
    --max-prompt-length 256 \
    --dev-num -1 \
    --model-type llama-7B-sft

cp ${BASE_PATH}/processed_data/dolly/full/llama/valid_0.bin ${BASE_PATH}/processed_data/dolly/pseudo/llama-7B-sft/
cp ${BASE_PATH}/processed_data/dolly/full/llama/valid_0.idx ${BASE_PATH}/processed_data/dolly/pseudo/llama-7B-sft/
cp ${BASE_PATH}/processed_data/dolly/full/llama/valid.jsonl ${BASE_PATH}/processed_data/dolly/pseudo/llama-7B-sft/