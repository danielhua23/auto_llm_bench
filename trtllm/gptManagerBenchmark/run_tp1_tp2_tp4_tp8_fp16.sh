MODEL_ROOT="/home/models"

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_name) MODEL_NAME="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
    case $1 in
        --prompt_len) PROMPT_LEN="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
    case $1 in
        --new_tokens) NEW_TOKENS="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
    case $1 in
        --data_type) DATA_TYPE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
    case $1 in
        --tp) TP="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
    case $1 in
        --static_bs) STATIC_BS="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

PROMPT_LEN_SP=""
for i in $(echo $PROMPT_LEN | tr "," "\n")
do
  PROMPT_LEN_SP="$PROMPT_LEN_SP $i"
done

NEW_TOKENS_SP=""
for i in $(echo $NEW_TOKENS | tr "," "\n")
do
  NEW_TOKENS_SP="$NEW_TOKENS_SP $i"
done

DATA_TYPE_SP=""
for i in $(echo $DATA_TYPE | tr "," "\n")
do
  DATA_TYPE_SP="$DATA_TYPE_SP $i"
done

STATIC_BATCH_SIZE=""
for i in $(echo $STATIC_BS | tr "," "\n")
do
    STATIC_BATCH_SIZE="$STATIC_BATCH_SIZE $i"
done

if [ -z "$MODEL_NAME" ]; then
    echo "Error: Missing one or more required parameters."
    usage
fi

echo "=hyper params start="
echo $MODEL_NAME
echo $PROMPT_LEN_SP
echo $NEW_TOKENS_SP
echo $DATA_TYPE_SP
echo $TP
echo $STATIC_BATCH_SIZE

engine_rebuild=0
hf_model_path=$MODEL_ROOT/$MODEL_NAME
trtllm_ckpt_path=/home/models/trtllm_ckpt_fp16/$TP/$MODEL_NAME
engine_dir=/home/models/engines_fp16/GPTM_elumated/$TP/$MODEL_NAME
tp_size=$TP
data_type=$DATA_TYPE_SP
isl=$PROMPT_LEN_SP
osl=$NEW_TOKENS_SP
bs=$STATIC_BATCH_SIZE

# trtlllm engine build
for dt in $data_type; do
    if [[ "$MODEL_NAME" == "Llama-2-7b-hf"* ]]; then
        if ls "$trtllm_ckpt_path"/*.safetensors 1> /dev/null 2>&1; then
            echo "convert done previously"
        else
            # weights convert: https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama
            python3 /app/tensorrt_llm/examples/llama/convert_checkpoint.py  --model_dir $hf_model_path --dtype $data_type --output_dir $trtllm_ckpt_path --tp_size $tp_size
            if [ $? -ne 0 ]; then
                echo " '$MODEL_NAME' 7b ck_convert failed, EXIT"
                exit 1
            else
                echo " '$MODEL_NAME' 7b ck_convert done"
            fi
        fi
    elif [[ "$MODEL_NAME" == "Llama-2-13b-hf" ]]; then
        if ls "$trtllm_ckpt_path"/*.safetensors 1> /dev/null 2>&1; then
            echo "convert done previously"
        else
            python3 /app/tensorrt_llm/examples/llama/convert_checkpoint.py  --model_dir $hf_model_path --dtype $data_type --output_dir $trtllm_ckpt_path --tp_size $tp_size
            if [ $? -ne 0 ]; then
                echo " '$MODEL_NAME' 13b ck_convert failed, EXIT"
                exit 1
            else
                echo " '$MODEL_NAME' 13b ck_convert done"
            fi
        fi
    elif [[ "$MODEL_NAME" == "Llama-2-70b-chat-hf" ]]; then
        if ls "$trtllm_ckpt_path"/*.safetensors 1> /dev/null 2>&1; then
            echo "convert done previously"
        else
            python3 /app/tensorrt_llm/examples/llama/convert_checkpoint.py  --model_dir $hf_model_path --dtype $data_type --output_dir $trtllm_ckpt_path --tp_size $tp_size
            if [ $? -ne 0 ]; then
                echo " '$MODEL_NAME' 70b ck_convert failed, EXIT"
                exit 1
            else
                echo " '$MODEL_NAME' 70b ck_convert done"
            fi
        fi
    else
        echo "TODO"
    fi
done
one=1
two=2
for dt in $data_type; do
        for batch_size in $bs; do
                for in_num in $isl; do
                    for out_num in $osl; do
                        in_out_dims="${in_num},${out_num}"
                        echo "BS: $batch_size, ISL/OSL: $in_out_dims"
                        max_seq_len=$(($one*($in_num+$out_num)))
                        max_num_tokens=$((($two*$in_num)+$((in_num / two))))
                        token_num=$(($batch_size*($in_num+$out_num)))
                        echo "max_seq_len: ${max_seq_len}"
                        echo "max_num_tokens: ${max_num_tokens}"
                        trtllm-build --checkpoint_dir $trtllm_ckpt_path  --output_dir $engine_dir --max_batch_size $batch_size --max_input_len $in_num --max_seq_len $max_seq_len --max_num_tokens $max_num_tokens \
				--gemm_plugin bfloat16 \
				--gemm_plugin $data_type --use_fused_mlp  --gpt_attention_plugin $data_type --use_paged_context_fmha enable

                        echo "start benchmark ${batch_size},${in_num},${out_num}.."
                        python3 bench_engine_tp2_4_8.py \
			    --osl ${out_num} \
                            --trtllm_dir /app/tensorrt_llm/ \
			    --tpsize $tp_size \
                            --engine_dir $engine_dir \
                            --model_dir $hf_model_path \
                            --batch_size_list $batch_size \
                            --seq_len_list $in_num >> mgr_emulated_report_${MODEL_NAME}_b${batch_size}_isl${in_num}_osl${out_num}_fp16_tp${tp_size}.log
                        echo "benchmark done..."
                    done
                done
        done
done