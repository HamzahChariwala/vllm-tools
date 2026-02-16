#!/usr/bin/env python3
"""
Simple script to run Qwen2.5 72B with tensor parallelism over 4 GPUs and profiling enabled.
"""
from vllm import LLM, SamplingParams

# Initialize LLM with tensor parallelism and profiling
llm = LLM(
    model="Qwen/Qwen2.5-72B-Instruct",
    tensor_parallel_size=4,
    gpu_memory_utilization=0.8,  # Use 80% of GPU memory (leave headroom for profiler)
    enforce_eager=True,  # Disable CUDAGraph optimization
    profiler_config={
        "profiler": "torch",  # Use PyTorch profiler
        "torch_profiler_dir": "/home/azureuser/vllm-tools/tp2_test_traces",
        "torch_profiler_with_stack": False,
        "torch_profiler_with_flops": False,
        "torch_profiler_record_shapes": True,
        "torch_profiler_with_memory": True,
        "torch_profiler_trace_execution": True,  # Enable ExecutionTraceObserver
    },
)

# Simple prompt
prompts = ["Tell me about artificial intelligence in 100 words."]

# Sampling parameters
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)

# Start profiler (ET observer will start automatically in worker processes)
print("Starting profiler...")
llm.start_profile()

# Generate
outputs = llm.generate(prompts, sampling_params)

# Stop profiler (ET observer will stop automatically in worker processes)
print("Stopping profiler...")
llm.stop_profile()

# Print results
for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")
    print("-" * 80)

