"""Score samples from eval/generate.py in the sandbox, and compare scored runs.

    python -m eval.score RUN_DIR [RUN_DIR ...]   # writes <dataset>.results.jsonl and summary.json per run
    python -m eval.score --compare RUN_A RUN_B   # paired comparison on the problems both runs share

Two pass rates per dataset: "base" uses only the standard tests (MBPP's asserts, HumanEval's original
check), which EvalPlus reports as MBPP and HumanEval; "all" adds MBPP's challenge tests or the EvalPlus
extra tests (MBPP+ and HumanEval+). mbppplus_dedup is MBPP+ without the problems in MBPP's train split.
Code is sanitized first (src.prompts.sanitize), as EvalPlus does before the numbers Qwen publishes.
"""
import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from math import comb
from pathlib import Path

import numpy as np

from src.data import MBPP_TRAIN_IDS, load_problems
from src.prompts import defined_names, sanitize
from src.sandbox import ExecResult, execute

TIMEOUT = 20.0  # Mbpp/599's reference needs ~10 s on its extra tests
MEMORY = 4 << 30  # EvalPlus's limit; Mbpp/255's extra tests need more than 2 GiB
METRICS = ("base", "all")
_SYNTAX_ERRORS = {"SyntaxError", "IndentationError", "TabError"}


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator from the Codex paper: n samples, c correct."""
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def categorize(code: str | None, result: ExecResult | None, entry_point: str) -> str:
    """"pass", or why the sample failed: the first problem found, from extraction through to the tests.

    `code` is the extracted code before sanitizing; `result` is the run that decided the outcome.
    """
    if code is None:
        return "no_code_block"
    if result.status == "ok" and all(result.passed):
        return "pass"
    try:
        names = defined_names(code)
    except Exception:
        return "syntax_error"
    if entry_point not in names:
        return "wrong_name"
    if result.status in ("timeout", "crash"):
        return result.status
    if result.status == "error":  # raised while loading, often from an assert the model added itself
        first = result.errors[0]
        if first in _SYNTAX_ERRORS:
            return "syntax_error"
    else:
        first = next(e for e, ok in zip(result.errors, result.passed) if not ok)
    return "assert_failure" if first == "AssertionError" else "runtime_error"


def score_sample(sample: dict, problem: dict, timeout: float, memory: int) -> dict:
    code, tests, setup, n = sample["code"], problem["tests"], problem["test_setup_code"], problem["n_base"]
    base_ok = all_ok = False
    result = None
    if code is not None:
        clean = sanitize(code, tests, setup)
        # Base and extra tests run separately, as in EvalPlus, so slow extra tests can't sink the base verdict.
        result = execute(clean, tests[:n], setup, timeout, memory)
        base_ok = all_ok = result.status == "ok" and all(result.passed)
        if base_ok and n < len(tests):
            result = execute(clean, tests[n:], setup, timeout, memory)
            all_ok = result.status == "ok" and all(result.passed)
    return {"task_id": sample["task_id"], "sample": sample["sample"],
            "category": categorize(code, result, problem["entry_point"]),
            "base": base_ok, "all": all_ok, "finish_reason": sample["finish_reason"]}


def score_run(run_dir: Path, ks: list[int], timeout: float, memory: int, workers: int) -> dict:
    summaries = {}
    for path in sorted(run_dir.glob("*.samples.jsonl")):
        name = path.name.removesuffix(".samples.jsonl")
        problems = {p["task_id"]: p for p in load_problems(name)}
        samples = [json.loads(line) for line in path.open()]
        # Run in a shuffled order: the samples of one task sit next to each other, and a few tasks'
        # extra tests need GBs of memory, so running them all at once could exhaust RAM.
        order = np.random.default_rng(0).permutation(len(samples))
        with ThreadPoolExecutor(workers) as pool:
            scored = pool.map(lambda i: score_sample(samples[i], problems[samples[i]["task_id"]], timeout, memory), order)
            rows = [row for _, row in sorted(zip(order, scored))]
        with (run_dir / f"{name}.results.jsonl").open("w") as f:
            f.writelines(json.dumps(row) + "\n" for row in rows)
        for view, view_rows in _views(name, rows):
            summaries[view] = summarize(view_rows, ks)
    (run_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    return summaries


def summarize(rows: list[dict], ks: list[int]) -> dict:
    counts = {m: _counts(rows, m) for m in METRICS}
    min_n = min(n for n, _ in counts["all"].values())
    summary = {"problems": len(counts["all"]), "samples_per_problem": min_n}
    for m in METRICS:
        rates = np.array([c / n for n, c in counts[m].values()])
        summary[f"{m}_pass@1"] = float(rates.mean())
        summary[f"{m}_pass@1_se"] = float(rates.std(ddof=1) / np.sqrt(len(rates)))
        for k in ks:
            if 1 < k <= min_n:
                summary[f"{m}_pass@{k}"] = float(np.mean([pass_at_k(n, c, k) for n, c in counts[m].values()]))
    summary["truncated"] = float(np.mean([r["finish_reason"] == "length" for r in rows]))
    summary["categories"] = {c: k / len(rows) for c, k in Counter(r["category"] for r in rows).most_common()}
    return summary


def print_summaries(run_dir: Path, summaries: dict, ks: list[int]) -> None:
    print(f"\n{run_dir}")
    print(f"{'dataset':<26}{'problems':>8}{'n':>4}  {'tests':<5}{'pass@1':>13}"
          + "".join(f"{f'pass@{k}':>9}" for k in ks if k > 1) + f"{'cut off':>9}")
    for name, s in summaries.items():
        for m in METRICS:
            head = f"{name:<26}{s['problems']:>8}{s['samples_per_problem']:>4}" if m == "base" else " " * 38
            pass1 = f"{100 * s[f'{m}_pass@1']:.1f} ± {100 * s[f'{m}_pass@1_se']:.1f}"
            at_k = "".join(f"{100 * s[f'{m}_pass@{k}']:>9.1f}" if f"{m}_pass@{k}" in s else f"{'':>9}"
                           for k in ks if k > 1)
            tail = f"{100 * s['truncated']:>8.1f}%" if m == "base" else ""
            print(f"{head}  {m:<5}{pass1:>13}{at_k}{tail}")
    print("\noutcomes, as a share of all samples:")
    for name, s in summaries.items():
        print(f"  {name:<24}" + ", ".join(f"{c} {100 * v:.1f}%" for c, v in s["categories"].items()))


def paired_rows(dir_a: Path, dir_b: Path) -> list[dict]:
    """Paired comparison per problem for every dataset both runs scored: pass rates, B - A with a
    bootstrap 95% CI, and McNemar's p when both runs are greedy (None otherwise)."""
    rows = []
    for path_a in sorted(dir_a.glob("*.results.jsonl")):
        path_b = dir_b / path_a.name
        if not path_b.exists():
            continue
        name = path_a.name.removesuffix(".results.jsonl")
        views_b = dict(_views(name, _load(path_b)))
        for view, rows_a in _views(name, _load(path_a)):
            for m in METRICS:
                counts_a, counts_b = _counts(rows_a, m), _counts(views_b[view], m)
                tasks = sorted(counts_a.keys() & counts_b.keys())
                a = np.array([counts_a[t][1] / counts_a[t][0] for t in tasks])
                b = np.array([counts_b[t][1] / counts_b[t][0] for t in tasks])
                greedy = all(counts_a[t][0] == counts_b[t][0] == 1 for t in tasks)
                rows.append({"dataset": view, "tests": m, "problems": len(tasks), "a": float(a.mean()),
                             "b": float(b.mean()), "diff": float(b.mean() - a.mean()), "ci": paired_bootstrap(a, b),
                             "mcnemar_p": mcnemar_p(a == 1, b == 1) if greedy else None})
    return rows


def print_paired(rows: list[dict]) -> None:
    print(f"{'dataset':<26}{'tests':<6}{'problems':>8}{'A':>8}{'B':>8}{'B - A':>8}{'95% CI':>18}{'McNemar p':>11}")
    for r in rows:
        ci = f"[{100 * r['ci'][0]:+.1f}, {100 * r['ci'][1]:+.1f}]"
        p = "-" if r["mcnemar_p"] is None else f"{r['mcnemar_p']:.3f}"
        print(f"{r['dataset']:<26}{r['tests']:<6}{r['problems']:>8}{100 * r['a']:>8.1f}{100 * r['b']:>8.1f}"
              f"{100 * r['diff']:>+8.1f}{ci:>18}{p:>11}")


def compare(dir_a: Path, dir_b: Path) -> None:
    """Paired comparison per problem: bootstrap CI of the difference, plus McNemar's test for greedy runs."""
    print(f"A = {dir_a}\nB = {dir_b}\n")
    print_paired(paired_rows(dir_a, dir_b))


def paired_bootstrap(a: np.ndarray, b: np.ndarray, iters: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """95% CI for mean(b - a), resampling problems (not samples) with replacement."""
    idx = np.random.default_rng(seed).integers(0, len(a), (iters, len(a)))
    diffs = (b - a)[idx].mean(axis=1)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def mcnemar_p(a: np.ndarray, b: np.ndarray) -> float:
    """Exact two-sided McNemar test on paired pass/fail outcomes: only the discordant problems count."""
    only_a, only_b = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n = only_a + only_b
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2**n)


def _counts(rows: list[dict], metric: str) -> dict[str, tuple[int, int]]:
    """task_id -> (samples, samples that passed)."""
    counts = defaultdict(lambda: [0, 0])
    for r in rows:
        counts[r["task_id"]][0] += 1
        counts[r["task_id"]][1] += r[metric]
    return {t: (n, c) for t, (n, c) in counts.items()}


def _views(name: str, rows: list[dict]):
    yield name, rows
    if name == "mbppplus":
        yield "mbppplus_dedup", [r for r in rows if int(r["task_id"].split("/")[1]) not in MBPP_TRAIN_IDS]


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="run directories written by eval.generate")
    parser.add_argument("--compare", action="store_true", help="compare two already-scored runs")
    parser.add_argument("--k", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--timeout", type=float, default=TIMEOUT)
    parser.add_argument("--memory-gb", type=float, default=MEMORY / 2**30)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.compare:
        if len(args.runs) != 2:
            parser.error("--compare takes exactly two runs")
        compare(*args.runs)
        return
    for run_dir in args.runs:
        summaries = score_run(run_dir, args.k, args.timeout, int(args.memory_gb * 2**30), args.workers)
        print_summaries(run_dir, summaries, args.k)


if __name__ == "__main__":
    main()
