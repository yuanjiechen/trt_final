python build.py --model_dir ./weight/ --dtype float16 --use_gpt_attention_plugin float16 --use_gemm_plugin float16 --max_input_len 512 --max_output_len 1024 --visualize
/usr/local/TensorRT-9.0.0.2/bin/trtexec --onnx=./qformer.onnx --fp16 --saveEngine=./llama_outputs/qformer.engine