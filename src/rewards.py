"""TRL reward functions. TRL passes extra dataset columns (here `tests`, and `test_setup_code` if present) as kwargs.

Prompts come from src.prompts.build_prompt as plain strings, so each completion is a string that starts
inside a ```python block.
"""
import ast
from concurrent.futures import ThreadPoolExecutor

from src.prompts import completion_code, sanitize
from src.sandbox import run_tests

_pool = ThreadPoolExecutor(max_workers=12)


def unit_test_reward(completions, tests, test_setup_code=None, **kwargs):
    setups = test_setup_code or [""] * len(completions)
    codes = [sanitize(completion_code(c), t, s) for c, t, s in zip(completions, tests, setups)]  # same as eval
    return list(_pool.map(lambda job: run_tests(*job) if job[0].strip() else 0.0, zip(codes, tests, setups)))


def format_reward(completions, **kwargs):
    """1.0 if the code block is closed and its code parses, else 0.0; keep its weight small (~0.1)."""
    return [1.0 if "```" in c and _parses(completion_code(c)) else 0.0 for c in completions]


def _parses(code: str) -> bool:
    if not code:
        return False
    try:
        ast.parse(code)
    except Exception:  # SyntaxError, or ValueError/RecursionError on pathological input; never crash training
        return False
    return True
