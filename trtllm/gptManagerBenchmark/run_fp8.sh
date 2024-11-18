bash run_tp1_tp2_tp4_tp8_fp8.sh --model_name "Llama-2-7b-chat-hf"  --prompt_len 128 --new_tokens 2048 --data_type float16  --tp 1 --static_bs 32,64,128,256
echo "7b tp1 fp8 gptm elumated done"
bash run_tp1_tp2_tp4_tp8_fp8.sh --model_name "Llama-2-70b-chat-hf"  --prompt_len 2048 --new_tokens 2048 --data_type float16  --tp 2 --static_bs 1,4,8,16,32,64,128,256
bash run_tp1_tp2_tp4_tp8_fp8.sh --model_name "Llama-2-70b-chat-hf"  --prompt_len 128,2048 --new_tokens 128 --data_type float16  --tp 2 --static_bs 1,4,8,16,32,64,128,256
echo "70b tp2 fp8 gptm elumated done"
bash run_tp1_tp2_tp4_tp8_fp8.sh --model_name "Llama-2-70b-chat-hf"  --prompt_len 2048 --new_tokens 2048 --data_type float16  --tp 4 --static_bs 1,4,8,16,32,64,128,256
bash run_tp1_tp2_tp4_tp8_fp8.sh --model_name "Llama-2-70b-chat-hf"  --prompt_len 128,2048 --new_tokens 128 --data_type float16  --tp 4 --static_bs 1,4,8,16,32,64,128,256
echo "70b tp4 fp16 gptm elumated done"
bash run_tp1_tp2_tp4_tp8_fp8.sh --model_name "Llama-2-70b-chat-hf"  --prompt_len 2048 --new_tokens 2048 --data_type float16  --tp 8 --static_bs 1,4,8,16,32,64,128,256
bash run_tp1_tp2_tp4_tp8_fp8.sh --model_name "Llama-2-70b-chat-hf"  --prompt_len 128,2048 --new_tokens 128 --data_type float16  --tp 8 --static_bs 1,4,8,16,32,64,128,256
echo "70b tp8 fp8 gptm elumated done
