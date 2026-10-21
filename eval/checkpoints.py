"""Evaluate the base model and checkpoints of a GRPO run in one vLLM engine, so only the weights differ.

    python -m eval.checkpoints runs/grpo/coder-0.5b                    # Phase 5: validation curve
    python -m eval.checkpoints runs/grpo/coder-0.5b --steps 0 125 --name test \
        --datasets mbpp_test mbppplus humanevalplus                     # Phase 6: base vs chosen step

Step 0 is the base model; other steps apply checkpoint-<step>/ as a LoRA adapter. Generation is greedy.
Writes <run>/<name>/step-<N>/<dataset>.{samples,results}.jsonl and <run>/<name>.json, and reuses
samples that already exist, so it can be rerun as training adds checkpoints. Compare two steps with
`python -m eval.score --compare <run>/<name>/step-0 <run>/<name>/step-<N>`.
"""
import argparse
import json
from pathlib import Path

from vllm import SamplingParams
from vllm.lora.request import LoRARequest

from eval.generate import generate, make_llm, write_samples
from eval.score import MEMORY, TIMEOUT, score_run
from src.data import DATASETS, load_problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="output directory of train_grpo.py")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=["mbpp_validation"])
    parser.add_argument("--steps", nargs="+", type=int, help="default: 0 and every checkpoint")
    parser.add_argument("--name", default="validation")
    args = parser.parse_args()

    model = json.loads((args.run / "args.json").read_text())["model"]
    available = sorted(int(c.name.split("-")[1]) for c in args.run.glob("checkpoint-*"))
    steps = args.steps or [0] + available
    if missing := [s for s in steps if s and s not in available]:
        parser.error(f"no checkpoint for steps {missing}; available: {available}")
    problems = {name: load_problems(name) for name in args.datasets}
    params = SamplingParams(temperature=0.0, max_tokens=1024)
    llm = None

    results = []
    for step in steps:
        out = args.run / args.name / f"step-{step}"
        for name, probs in problems.items():
            samples = out / f"{name}.samples.jsonl"
            if not samples.exists():
                llm = llm or make_llm(model, seed=0, lora=True)
                lora = LoRARequest(f"step-{step}", step, str(args.run / f"checkpoint-{step}")) if step else None
                write_samples(samples, generate(llm, probs, params, lora))
        summaries = score_run(out, ks=[1], timeout=TIMEOUT, memory=MEMORY, workers=12)
        results.append({"step": step, **summaries})
        print(f"step {step:>4}: " + ", ".join(f"{name} {100 * s['base_pass@1']:.1f}/{100 * s['all_pass@1']:.1f}"
                                              for name, s in summaries.items()) + "  (pass@1 base/all tests)")
    (args.run / f"{args.name}.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
