"""Phase 4: build a GRPO train set from MBPP train, keeping only problems that give a gradient.

    python -m scripts.make_train_set --model Qwen/Qwen2.5-Coder-0.5B-Instruct --out data/train-coder-0.5b.jsonl

Samples 8 completions per MBPP train problem with the training settings (temperature 1.0, 512 tokens),
scores them with the training reward, and keeps problems where some but not all samples pass every test.
Each completion's advantage is (r - mean) / std within its group, so a problem where every sample gets
the same reward contributes nothing (see Phase 0). With partial credit, a group where nothing fully
passes can still differ, mostly by passing only the visible assert; we drop those too, so that a binary
reward gets a gradient on every kept problem as well, and to avoid rewarding a visible-test-only solution.

Each output row has task_id, text, tests, test_setup_code and entry_point (see src.data), plus the
filtering samples' rewards. The raw samples go next to it as <out>.samples.jsonl.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from vllm import SamplingParams

from eval.generate import generate, make_llm, write_samples
from src.data import load_problems
from src.rewards import unit_test_reward

FIELDS = ("task_id", "text", "tests", "test_setup_code", "entry_point")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n", type=int, default=8, help="samples per problem")
    parser.add_argument("--temperature", type=float, default=1.0, help="match the training temperature")
    parser.add_argument("--max-tokens", type=int, default=512, help="match max_completion_length")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    problems = load_problems("mbpp_train")
    llm = make_llm(args.model, args.seed)
    params = SamplingParams(n=args.n, temperature=args.temperature, max_tokens=args.max_tokens, seed=args.seed)
    samples = generate(llm, problems, params)
    write_samples(args.out.with_suffix(".samples.jsonl"), samples)

    by_task = {p["task_id"]: p for p in problems}
    rewards = unit_test_reward([s["output"] for s in samples],
                               tests=[by_task[s["task_id"]]["tests"] for s in samples],
                               test_setup_code=[by_task[s["task_id"]]["test_setup_code"] for s in samples])
    groups: dict[str, list[float]] = {}
    for s, r in zip(samples, rewards, strict=True):
        groups.setdefault(s["task_id"], []).append(r)

    kept = [{**{k: by_task[t][k] for k in FIELDS}, "filter_rewards": rs}
            for t, rs in groups.items() if 0 < sum(r == 1.0 for r in rs) < args.n]
    with args.out.open("w") as f:
        f.writelines(json.dumps(row) + "\n" for row in kept)

    passes = Counter(sum(r == 1.0 for r in rs) for rs in groups.values())
    partial_only = sum(1 for rs in groups.values() if max(rs) < 1.0 and len(set(rs)) > 1)
    print(f"kept {len(kept)}/{len(groups)} problems -> {args.out}")
    print(f"  dropped {passes[args.n]} where all {args.n} pass and {passes[0]} where none passes "
          f"({partial_only} of those had differing partial credit)")
    print("  problems by number of fully passing samples: " + ", ".join(f"{k}: {passes[k]}" for k in range(args.n + 1)))


if __name__ == "__main__":
    main()
