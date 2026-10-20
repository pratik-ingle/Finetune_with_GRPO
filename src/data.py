"""Load MBPP, MBPP+ and HumanEval+ into one problem format, shared by training and eval.

Each problem is a dict with:
  task_id          EvalPlus-style id: "Mbpp/11" or "HumanEval/0"
  text             the bare MBPP description, or the HumanEval stub
  problem          the text the prompt shows; src.prompts.build_prompt turns it into a model's prompt
  tests            test statements, each scored independently by the sandbox
  test_setup_code  runs after the solution and before the tests
  n_base           the first n_base tests are the standard ones; the rest are MBPP challenge tests
                   or the MBPP+/HumanEval+ extra tests
  entry_point      the function the tests call
  reference        a reference solution, for checking the tests themselves
"""
import ast

from datasets import load_dataset

from src.prompts import defined_names, mbpp_problem

DATASETS = ("mbpp_train", "mbpp_validation", "mbpp_test", "mbppplus", "humanevalplus")
MBPP_TRAIN_IDS = range(601, 975)  # MBPP's train split is task_ids 601-974; MBPP+ reuses 108 of them


def load_problems(name: str) -> list[dict]:
    if name in ("mbpp_train", "mbpp_validation", "mbpp_test"):
        rows = load_dataset("google-research-datasets/mbpp", "full", split=name.removeprefix("mbpp_"))
        return [_mbpp(r["task_id"], r["text"], r["code"], r["test_list"], r["challenge_test_list"], r["test_setup_code"])
                for r in rows]
    if name == "mbppplus":
        rows = load_dataset("evalplus/mbppplus", split="test")
        return [_mbpp(r["task_id"], r["prompt"], r["code"], r["test_list"], [r["test"]], "\n".join(r["test_imports"]))
                for r in rows]
    if name == "humanevalplus":
        base_tests = {r["task_id"]: r["test"] for r in load_dataset("openai/openai_humaneval", split="test")}
        return [{
            "task_id": r["task_id"],
            "text": r["prompt"],
            "problem": r["prompt"],
            "tests": [f"{base_tests[r['task_id']]}\ncheck({r['entry_point']})",
                      f"{_fix_humaneval_test(r['task_id'], r['test'])}\ncheck({r['entry_point']})"],
            "test_setup_code": _prompt_helpers(r["prompt"], r["entry_point"]),
            "n_base": 1,
            "entry_point": r["entry_point"],
            "reference": r["prompt"] + r["canonical_solution"],
        } for r in load_dataset("evalplus/humanevalplus", split="test")]
    raise ValueError(f"unknown dataset {name!r}; expected one of {DATASETS}")


def _mbpp(task_id: int, text: str, code: str, tests: list[str], extra_tests: list[str], setup: str) -> dict:
    return {
        "task_id": f"Mbpp/{task_id}",
        "text": text,
        "problem": mbpp_problem(text, tests[:1]),
        "tests": tests + extra_tests,
        "test_setup_code": setup,
        "n_base": len(tests),
        "entry_point": _entry_point(code, tests),
        "reference": code,
    }


def _prompt_helpers(prompt: str, entry_point: str) -> str:
    """Imports and helper functions from a HumanEval prompt, e.g. encode_cyclic for HumanEval/38.

    The original tests call them, and a self-contained answer often leaves them out. Classic HumanEval
    ran prompt + completion, so they were always defined; running them as setup restores that.
    """
    lines = prompt.splitlines()
    return "\n".join("\n".join(lines[node.lineno - 1:node.end_lineno]) for node in ast.parse(prompt).body
                     if isinstance(node, (ast.Import, ast.ImportFrom))
                     or (isinstance(node, ast.FunctionDef) and node.name != entry_point))


def _fix_humaneval_test(task_id: str, test: str) -> str:
    # The HF copy of HumanEval/32's plus test unpacks the returned float and always raises TypeError.
    # EvalPlus's own harness special-cases find_zero with this check instead.
    if task_id != "HumanEval/32":
        return test
    bad = "assert _poly(*candidate(*inp), inp) <= 0.0001"
    assert test.count(bad) == 1, "HumanEval/32's plus test changed; recheck the fix"
    return test.replace(bad, "assert abs(_poly(*inp, candidate(*inp))) <= 0.0001")


def _entry_point(code: str, tests: list[str]) -> str:
    """The first function the tests call that the reference solution defines."""
    defined = defined_names(code)
    for test in tests:
        for node in ast.walk(ast.parse(test)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in defined:
                return node.func.id
    raise ValueError(f"no function called by the tests is defined by the reference: {tests[0]!r}")
