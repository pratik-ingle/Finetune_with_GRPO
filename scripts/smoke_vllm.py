"""Phase 1 smoke test: vLLM generates one completion from Qwen2.5-Coder-0.5B-Instruct."""
import torch
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"


def main():
    print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, device {torch.cuda.get_device_name(0)}")
    llm = LLM(MODEL, gpu_memory_utilization=0.5, max_model_len=2048, seed=0)
    messages = [{"role": "user", "content": "Write a Python function `fib(n)` that returns the nth Fibonacci number."}]
    out = llm.chat(messages, SamplingParams(temperature=0.0, max_tokens=256))
    print(out[0].outputs[0].text)


if __name__ == "__main__":  # vLLM spawns worker processes, which re-import this module
    main()
