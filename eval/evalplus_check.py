"""Cross-check eval/score.py against the EvalPlus CLI on the same samples (MBPP+ and HumanEval+).

    python -m eval.score RUN_DIR && python -m eval.evalplus_check RUN_DIR

Sanitizes each sample's code with EvalPlus's own sanitizer, as its codegen does before the numbers
Qwen publishes, runs `evalplus.evaluate`, and lists every sample where EvalPlus's verdict differs from
ours, for the base tests and for base + extra tests. Known sources of difference:
  - HumanEval/32: EvalPlus fails every solution, even its reference (its precision check can't be met
    in float64); we use the original HumanEval test for the base verdict, which a correct solution passes.
  - EvalPlus limits each input to 4x the reference's runtime, so it also fails slow-but-correct code;
    we only cap each run at 20 s.
  - The Hugging Face copy of MBPP+ that we test with differs from EvalPlus's release on a few problems.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from evalplus.sanitize import sanitize

from src.data import load_problems

EVALPLUS_NAMES = {"mbppplus": "mbpp", "humanevalplus": "humaneval"}


def check(run_dir: Path, name: str) -> None:
    entry_points = {p["task_id"]: p["entry_point"] for p in load_problems(name)}
    samples = [json.loads(line) for line in (run_dir / f"{name}.samples.jsonl").open()]
    ours = {(r["task_id"], r["sample"]): r for r in map(json.loads, (run_dir / f"{name}.results.jsonl").open())}
    evalplus_samples = run_dir / f"{name}.evalplus.jsonl"
    with evalplus_samples.open("w") as f:  # EvalPlus numbers samples per task in file order, like we do
        for s in samples:
            solution = sanitize(s["code"] or "", entry_points[s["task_id"]])
            f.write(json.dumps({"task_id": s["task_id"], "solution": solution}) + "\n")
    subprocess.run([sys.executable, "-m", "evalplus.evaluate", "--dataset", EVALPLUS_NAMES[name],
                    "--samples", str(evalplus_samples), "--i_just_wanna_run"], check=True)
    theirs = json.loads((run_dir / f"{name}.evalplus_eval_results.json").read_text())["eval"]

    disagree = []
    for task_id, results in theirs.items():
        for i, r in enumerate(results):
            base = r["base_status"] == "pass"
            plus = base and r["plus_status"] == "pass"
            mine = ours[(task_id, i)]
            if (mine["base"], mine["all"]) != (base, plus):
                disagree.append(f"{task_id} #{i}: ours base={mine['base']} all={mine['all']} ({mine['category']}), "
                                f"evalplus base={base} all={plus}")
    print(f"\n{name}: ours and EvalPlus agree on {len(ours) - len(disagree)}/{len(ours)} samples")
    for line in disagree:
        print("  " + line)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    for name in EVALPLUS_NAMES:
        if (args.run_dir / f"{name}.results.jsonl").exists():
            check(args.run_dir, name)


if __name__ == "__main__":
    main()
