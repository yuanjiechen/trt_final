cd ./examples/MiniGPT-4/RPTQ4LLM-master
python main.py --wbits 8 --abits 8 llama-7b
python build.py --model_dir ./weight/ --dtype float16 --use_gpt_attention_plugin float16 --use_gemm_plugin float16 --max_input_len 512 --max_output_len 1024 --visualize
/usr/local/TensorRT-9.0.0.2/bin/trtexec --onnx=./qformer.onnx --fp16 --saveEngine=./llama_outputs/qformer.engine
python summarize.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
python summarize.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0 --load_torch
python summarize.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0 --eval