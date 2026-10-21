"""Phase 5/6: one row per GRPO run, from TRL's log history and eval.checkpoints' validation curve.

    python -m scripts.summarize_runs runs/grpo/coder-0.5b-s0 runs/grpo/coder-0.5b-s1 ...

Train reward and the other training metrics are averaged over the first and last 10% of steps.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def summarize(run: Path) -> dict:
    history = [h for h in json.loads((run / "trainer_state.json").read_text())["log_history"] if "reward" in h]
    n = max(1, len(history) // 10)

    def early_late(key):
        values = [h[key] for h in history]
        return float(np.mean(values[:n])), float(np.mean(values[-n:]))

    row = {"run": run.name, "steps": len(history)}
    for key, short in (("rewards/unit_test_reward/mean", "test_reward"), ("frac_reward_zero_std", "zero_std"),
                       ("completions/mean_length", "length"), ("completions/clipped_ratio", "clipped")):
        alias = key if key in history[0] else "rewards/binary_unit_test_reward/mean"
        row[short] = early_late(alias)
    curve_path = run / "validation.json"
    if curve_path.exists():
        curve = [(c["step"], c["mbpp_validation"]["base_pass@1"]) for c in json.loads(curve_path.read_text())]
        best = max(curve, key=lambda c: (c[1], -c[0]))  # earliest step among ties
        row.update(val_base=curve[0][1], val_best=best[1], best_step=best[0], val_last=curve[-1][1])
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    print(f"{'run':<28}{'steps':>6}{'test reward':>16}{'zero-std':>14}{'length':>14}{'clipped':>14}"
          f"{'val step 0':>11}{'val best (step)':>17}{'val last':>9}")
    for run in args.runs:
        r = summarize(run)
        pair = lambda k, fmt: f"{fmt(r[k][0])} -> {fmt(r[k][1])}"  # noqa: E731
        line = (f"{r['run']:<28}{r['steps']:>6}{pair('test_reward', lambda v: f'{v:.2f}'):>16}"
                f"{pair('zero_std', lambda v: f'{v:.2f}'):>14}{pair('length', lambda v: f'{v:.0f}'):>14}"
                f"{pair('clipped', lambda v: f'{v:.2f}'):>14}")
        if "val_base" in r:
            line += (f"{100 * r['val_base']:>11.1f}{f'{100 * r['val_best']:.1f} ({r['best_step']})':>17}"
                     f"{100 * r['val_last']:>9.1f}")
        print(line)


if __name__ == "__main__":
    main()
