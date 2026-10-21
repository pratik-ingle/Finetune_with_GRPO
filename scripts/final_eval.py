"""Phase 6: evaluate each run's chosen checkpoint against its base model on the test sets, once.

    python -m scripts.final_eval runs/grpo/coder-1.5b-s0 runs/grpo/coder-1.5b-s1 ...
    python -m scripts.final_eval --report runs/grpo/*-s[0-9]       # table from existing results only

Picks the step with the best validation pass@1 (the earliest among ties), evaluates step 0 and that step
greedily on MBPP test, MBPP+ and HumanEval+ in one engine (eval.checkpoints --name test), and writes
<run>/final_compare.json with the paired comparison. The report groups runs by name without the
"-s<seed>" suffix and shows each metric's change, mean and range over seeds.
"""
import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval.score import paired_rows, print_paired

DATASETS = ["mbpp_test", "mbppplus", "humanevalplus"]
REPORT = [("mbpp_test", "base", "MBPP test"), ("mbppplus_dedup", "base", "MBPP dedup"),
          ("mbppplus_dedup", "all", "MBPP+ dedup"), ("humanevalplus", "base", "HumanEval"),
          ("humanevalplus", "all", "HumanEval+")]


def best_step(run: Path) -> int:
    curve = json.loads((run / "validation.json").read_text())
    return max(curve, key=lambda c: (c["mbpp_validation"]["base_pass@1"], -c["step"]))["step"]


def evaluate(run: Path) -> None:
    step = best_step(run)
    subprocess.run([sys.executable, "-m", "eval.checkpoints", str(run), "--steps", "0", str(step),
                    "--name", "test", "--datasets", *DATASETS], check=True)
    rows = paired_rows(run / "test" / "step-0", run / "test" / f"step-{step}")
    (run / "final_compare.json").write_text(json.dumps({"step": step, "rows": rows}, indent=2) + "\n")
    print(f"\n{run}: base (A) vs step {step} (B), chosen on validation")
    print_paired(rows)


def report(runs: list[Path]) -> None:
    groups = defaultdict(list)
    for run in runs:
        path = run / "final_compare.json"
        if path.exists():
            groups[re.sub(r"-s\d+$", "", run.name)].append(json.loads(path.read_text()))
    print(f"\n{'config':<18}{'seeds':>6}{'steps':>14}" + "".join(f"{label:>22}" for *_, label in REPORT))
    print(f"{'':<38}" + "".join(f"{'base -> trained':>22}" for _ in REPORT))
    for config, results in groups.items():
        cells = []
        for dataset, tests, _ in REPORT:
            rows = [next(r for r in res["rows"] if r["dataset"] == dataset and r["tests"] == tests) for res in results]
            diffs = [100 * r["diff"] for r in rows]
            spread = f" [{min(diffs):+.1f},{max(diffs):+.1f}]" if len(diffs) > 1 else ""
            cells.append(f"{100 * rows[0]['a']:.1f} {np.mean(diffs):+.1f}{spread}")
        steps = ",".join(str(res["step"]) for res in results)
        print(f"{config:<18}{len(results):>6}{steps:>14}" + "".join(f"{c:>22}" for c in cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--report", action="store_true", help="only print the table from existing results")
    args = parser.parse_args()
    runs = [run for run in args.runs if (run / "validation.json").exists()]  # globs also catch the .log files
    if not args.report:
        for run in runs:
            if not (run / "final_compare.json").exists():
                evaluate(run)
    report(runs)


if __name__ == "__main__":
    main()
