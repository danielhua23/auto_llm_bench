#!/usr/bin/bash
set -ex

export NCCL_IB_DISABLE=1

bash trtllm_latency_bench.sh --model_name "Llama-2-7b-hf"  --prompt_len 2048,4096 --new_tokens 128 --data_type float16  --tp 1 --static_bs 1,4,8,16,32,64,128,256,512
echo "7b fp16 done"
bash trtllm_latency_bench.sh --model_name "Llama-2-13b-hf"  --prompt_len 2048,4096 --new_tokens 128 --data_type float16  --tp 1 --static_bs 1,4,8,16,32,64,128,256,512
echo "13b fp16 done"
bash trtllm_latency_bench.sh --model_name "Llama-2-70b-chat-hf"  --prompt_len 2048,4096 --new_tokens 128 --data_type float16  --tp 8 --static_bs 1,4,8,16,32,64,128,256,512
echo "70b fp16 done"
