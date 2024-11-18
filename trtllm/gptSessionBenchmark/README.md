# TensorRT-LLM gptSessionBenchmark
## benchmark env
Follwing envs have been tested:

* Docker image: tensorrt_llm/release:v0.13.0, tensorrt_llm/release:v0.12.0
* TensorRT-LLM: 0.13.0, 0.12.0
* TensorRT: 10.4, 10.3
## How to run
### 0.Check docker images on your machine
you can check the "tensorrt_llm/release:v0.13.0" image is availiable or not by docker images, if it is, you can skip the next step, or you must execute the below cmd first.

### 1.How to build trtllm:0.12.0 image (Nov-15)
notice: directly pip wheel install will miss cpp/benchmark/gptSessionBenchmark executable binaries, the following is required to compile cpp/benchmark in docker first.

```
git clone -b v0.12.0 --recursive  https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
# on H20/H100
make -C docker release_build CUDA_ARCH="90-real"
# until now, the benchmark binary has been built successfully.
```
### 2.launch container
note: need change the "/path/to/home" and "/path/to/models" to your work dir and model dir on your machine
```
docker run --net=host --pid=host --ipc=host --privileged -it --gpus all -v /path/to/home/:/home/ -v /path/to/models:/home/models --name container_name tensorrt_llm/release:v0.12.0
```
### 3.benchmark fp16 models
before benchmark fp16 models, ensure your model is under the dir "/home/models/", your compiled tensorrt_llm named "/app/tensorrt_llm" exists.
```
cd home
git clone https://github.com/danielhua23/auto_llm_bench.git
cd trtllm/gptSessionBenchmark
bash run_fp16.sh
```
explanation about run_fp16.sh
```
# we can add different models benchmark in the following lines, model_name refer to the huggingface model name, prompt_len refer to input seq len, new_tokens refer to output seq len, data_type refer to the model data type which is downloaded from huggingface, tp refer to the GPU numbers used to execute tensor parallel. static_bs refer to the batch size we want benchmark
bash trtllm_latency_bench_fp16.sh --model_name "Llama-2-7b-hf"  --prompt_len 2048,4096 --new_tokens 128 --data_type float16  --tp 1 --static_bs 1,4,8,16,32,64,128,256,512
echo "7b fp16 done"
bash trtllm_latency_bench_fp16.sh --model_name "Llama-2-13b-hf"  --prompt_len 2048,4096 --new_tokens 128 --data_type float16  --tp 1 --static_bs 1,4,8,16,32,64,128,256,512
echo "13b fp16 done"
bash trtllm_latency_bench_fp16.sh --model_name "Llama-2-70b-chat-hf"  --prompt_len 2048,4096 --new_tokens 128 --data_type float16  --tp 8 --static_bs 1,4,8,16,32,64,128,256,512
echo "70b fp16 done"
```
### 4.benchmark fp8 models
before benchmark fp8 models, ensure your model is under the dir "/home/models/", your compiled tensorrt_llm named "/app/tensorrt_llm" exists.
```
cd home
git clone https://github.com/danielhua23/auto_llm_bench.git
cd trtllm/gptSessionBenchmark
bash run_fp8.sh
```
explanation about run_fp8.sh
```
# we can add different models benchmark in the following lines, model_name refer to the huggingface model name, prompt_len refer to input seq len, new_tokens refer to output seq len, data_type refer to the model data type which is downloaded from huggingface, tp refer to the GPU numbers used to execute tensor parallel. static_bs refer to the batch size we want benchmark
bash trtllm_latency_bench_fp8.sh --model_name "Llama-2-7b-hf"  --prompt_len 2048,4096 --new_tokens 128 --data_type float16  --tp 1 --static_bs 1,4,8,16,32,64,128,256,512
echo "7b fp8 done"
bash trtllm_latency_bench_fp8.sh --model_name "Llama-2-13b-hf"  --prompt_len 2048,4096 --new_tokens 128 --data_type float16  --tp 1 --static_bs 1,4,8,16,32,64,128,256,512
echo "13b fp8 done"
bash trtllm_latency_bench_fp8.sh --model_name "Llama-2-70b-chat-hf"  --prompt_len 2048,4096 --new_tokens 128 --data_type float16  --tp 8 --static_bs 1,4,8,16,32,64,128,256,512
echo "70b fp8 done"
```


