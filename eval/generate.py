"""Generate solutions with vLLM: writes <out>/<dataset>.samples.jsonl and <out>/config.json.

    python -m eval.generate --model Qwen/Qwen2.5-Coder-0.5B-Instruct --out runs/eval/coder-0.5b-greedy
    python -m eval.generate --model ... --n 10 --temperature 0.8 --out runs/eval/coder-0.5b-t0.8-n10
    python -m eval.generate --model ... --lora runs/grpo/x/checkpoint-100 --out runs/eval/x-step100

Each sample row has task_id, sample (index within the task), output (the raw completion, which starts
inside a ```python block; see src.prompts.build_prompt), code (completion_code's result) and
finish_reason ("length" means the output was cut off at --max-tokens).
"""
import argparse
import json
import os
from pathlib import Path

# Without batch-invariant kernels, about half of greedy outputs change between identical runs, because
# bf16 logits shift with whatever else is in the batch. Must be set before vLLM starts its engine.
os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")

import vllm  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.lora.request import LoRARequest  # noqa: E402

from src.data import DATASETS, load_problems  # noqa: E402
from src.prompts import build_prompt, completion_code  # noqa: E402

MAX_LORA_RANK = 64


def make_llm(model: str, seed: int, lora: bool = False) -> LLM:
    lora_args = {"enable_lora": True, "max_lora_rank": MAX_LORA_RANK, "max_loras": 1} if lora else {}
    # 0.85 ran out of memory with LoRA on large batches: vLLM's profiling misses the LoRA kernels' buffers.
    # With batch-invariant kernels the cache size doesn't change any output, only the speed.
    return LLM(model, seed=seed, max_model_len=4096, gpu_memory_utilization=0.7, **lora_args)


def generate(llm: LLM, problems: list[dict], params: SamplingParams, lora: LoRARequest | None = None) -> list[dict]:
    tokenizer = llm.get_tokenizer()
    outputs = llm.generate([build_prompt(p["problem"], tokenizer) for p in problems], params, lora_request=lora)
    return [{"task_id": problem["task_id"], "sample": i, "output": o.text,
             "code": completion_code(o.text), "finish_reason": o.finish_reason}
            for problem, out in zip(problems, outputs, strict=True) for i, o in enumerate(out.outputs)]


def write_samples(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="HF model id or local path")
    parser.add_argument("--lora", type=Path, help="LoRA adapter directory to apply on top of --model")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS,
                        default=["mbpp_validation", "mbpp_test", "mbppplus", "humanevalplus"])
    parser.add_argument("--n", type=int, default=1, help="samples per problem")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.n > 1 and args.temperature == 0:
        parser.error("--n > 1 needs --temperature > 0, or every sample is the same greedy output")

    llm = make_llm(args.model, args.seed, lora=args.lora is not None)
    lora = LoRARequest("adapter", 1, str(args.lora)) if args.lora else None
    problems = {name: load_problems(name) for name in args.datasets}

    args.out.mkdir(parents=True, exist_ok=True)
    example = build_prompt(problems[args.datasets[0]][0]["problem"], llm.get_tokenizer())
    config = {**vars(args), "out": str(args.out), "lora": args.lora and str(args.lora),
              "vllm": vllm.__version__, "example_prompt": example}
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    # A per-request seed makes sampled outputs independent of how vLLM batches the requests.
    params = SamplingParams(n=args.n, temperature=args.temperature, max_tokens=args.max_tokens, seed=args.seed)
    for name, probs in problems.items():
        path = args.out / f"{name}.samples.jsonl"
        write_samples(path, generate(llm, probs, params, lora))
        print(f"wrote {len(probs)} x {args.n} samples to {path}")


if __name__ == "__main__":
    main()
