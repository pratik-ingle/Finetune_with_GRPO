import numpy as np
import pytest

from eval.score import categorize, mcnemar_p, pass_at_k, summarize
from src.sandbox import ExecResult


def test_pass_at_k():
    assert pass_at_k(10, 0, 1) == 0.0
    assert pass_at_k(10, 10, 1) == 1.0
    assert pass_at_k(10, 3, 1) == pytest.approx(0.3)
    assert pass_at_k(10, 3, 5) == pytest.approx(1 - 21 / 252)  # 1 - C(7,5) / C(10,5)
    assert pass_at_k(10, 6, 5) == 1.0  # fewer than k failures: some draw of k always contains a pass


def test_mcnemar_p():
    same = np.array([True, False, True])
    assert mcnemar_p(same, same) == 1.0
    a, b = np.array([False] * 10), np.array([True] * 10)
    assert mcnemar_p(a, b) == pytest.approx(2 / 2**10)
    assert mcnemar_p(b, a) == mcnemar_p(a, b)


def ok(*marks):
    return ExecResult("ok", [m == "-" for m in marks], list(marks))


@pytest.mark.parametrize("code, result, expected", [
    (None, None, "no_code_block"),
    ("def f(): return 1", ok("-", "-"), "pass"),
    ("def f(:", ExecResult("error", errors=["SyntaxError"]), "syntax_error"),
    ("def f():\n  return 1\n return 2", ExecResult("error", errors=["IndentationError"]), "syntax_error"),
    ("def g(): return 1", ok("NameError", "NameError"), "wrong_name"),
    ("import nope\ndef f(): return 1", ExecResult("error", errors=["ModuleNotFoundError"]), "runtime_error"),
    ("def f(): return 1\nassert f() == 2", ExecResult("error", errors=["AssertionError"]), "assert_failure"),
    ("def f(): return 1", ok("-", "AssertionError"), "assert_failure"),
    ("def f(): return 1", ok("TypeError", "AssertionError"), "runtime_error"),
    ("def f():\n  while True: pass", ExecResult("timeout"), "timeout"),
    ("def f(): return 1", ExecResult("crash"), "crash"),
])
def test_categorize(code, result, expected):
    assert categorize(code, result, "f") == expected


def test_summarize():
    rows = [{"task_id": t, "base": b, "all": a, "category": "pass" if a else "assert_failure", "finish_reason": "stop"}
            for t, b, a in [("x", True, True), ("x", True, False), ("y", False, False), ("y", False, False)]]
    s = summarize(rows, ks=[1, 2])
    assert (s["problems"], s["samples_per_problem"]) == (2, 2)
    assert s["base_pass@1"] == pytest.approx(0.5)  # x: 2/2, y: 0/2
    assert s["all_pass@1"] == pytest.approx(0.25)  # x: 1/2, y: 0/2
    assert s["all_pass@2"] == pytest.approx(0.5)
    assert s["categories"] == {"assert_failure": 0.75, "pass": 0.25}
