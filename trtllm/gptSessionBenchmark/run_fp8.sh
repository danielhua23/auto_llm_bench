#/usr/bin/bash
set -ex

export NCCL_IB_DISABLE=1

bash trtllm_latency_bench_fp8.sh --model_name "Llama-2-70b-chat-hf"  --prompt_len 2048 --new_tokens 2048 --data_type float16  --tp 2 --static_bs 1,4,8,16,32,64,128,256
bash trtllm_latency_bench_fp8.sh --model_name "Llama-2-70b-chat-hf"  --prompt_len 128,2048 --new_tokens 128 --data_type float16  --tp 2 --static_bs 1,4,8,16,32,64,128,256
echo "70b fp16 tp2 done"
bash trtllm_latency_bench_fp8.sh --model_name "Llama-2-70b-chat-hf"  --prompt_len 2048 --new_tokens 2048 --data_type float16  --tp 4 --static_bs 1,4,8,16,32,64,128,256
bash trtllm_latency_bench_fp8.sh --model_name "Llama-2-70b-chat-hf"  --prompt_len 128,2048 --new_tokens 128 --data_type float16  --tp 4 --static_bs 1,4,8,16,32,64,128,256
echo "70b fp16 tp4 done"
bash trtllm_latency_bench_fp8.sh --model_name "Llama-2-70b-chat-hf"  --prompt_len 128,2048 --new_tokens 128 --data_type float16  --tp 8 --static_bs 1,4,8,16,32,64,128,256
bash trtllm_latency_bench_fp8.sh --model_name "Llama-2-70b-chat-hf"  --prompt_len 2048 --new_tokens 2048 --data_type float16  --tp 8 --static_bs 1,4,8,16,32,64,128,256
echo "70b fp16 tp8 done"
