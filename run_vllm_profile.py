#!/usr/bin/env python3
"""
Parameterized vLLM profiling script.

Usage:
    python run_vllm_profile.py \
        --model Qwen/Qwen2.5-72B-Instruct \
        --tensor-parallel-size 4 \
        --max-tokens 50 \
        --trace-dir /home/azureuser/vllm-tools/trace_storage/Qwen2.5-72B/tp4/tokens_50
"""
import argparse
import os

from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description="Profile a vLLM model with PyTorch profiler")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--max-tokens", type=int, default=10,
                        help="Maximum number of output tokens to generate")
    parser.add_argument("--trace-dir", required=True,
                        help="Directory to write profiler traces into")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8,
                        help="Fraction of GPU memory vLLM may use (default 0.8)")
    parser.add_argument("--prompt", default="Tell me about artificial intelligence in 100 words.",
                        help="Prompt to run during profiling")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.trace_dir, exist_ok=True)

    print(f"Model              : {args.model}")
    print(f"Tensor parallel    : {args.tensor_parallel_size}")
    print(f"Max output tokens  : {args.max_tokens}")
    print(f"Trace directory    : {args.trace_dir}")

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        profiler_config={
            "profiler": "torch",
            "torch_profiler_dir": args.trace_dir,
            "torch_profiler_with_stack": False,
            "torch_profiler_with_flops": False,
            "torch_profiler_record_shapes": True,
            "torch_profiler_with_memory": True,
            "torch_profiler_trace_execution": True,
        },
    )

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=args.max_tokens,
    )

    print("Starting profiler...")
    llm.start_profile()

    outputs = llm.generate([args.prompt], sampling_params)

    print("Stopping profiler...")
    llm.stop_profile()

    for output in outputs:
        print(f"Prompt   : {output.prompt}")
        print(f"Generated: {output.outputs[0].text}")
        print("-" * 80)

    print(f"Traces written to: {args.trace_dir}")


if __name__ == "__main__":
    main()
