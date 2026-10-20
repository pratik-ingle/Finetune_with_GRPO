import subprocess
import sys
import time

import pytest

from src.sandbox import execute, run_tests

ADD = "def add(a, b):\n    return a + b\n"
TESTS = ["assert add(1, 2) == 3", "assert add(-1, 1) == 0", "assert add(2, 2) == 4"]


def test_correct_solution_scores_one():
    assert run_tests(ADD, TESTS) == 1.0


def test_each_assert_counts_separately():
    result = execute("def add(a, b):\n    return abs(a) + b\n", TESTS)  # wrong only for negative a
    assert result.passed == [True, False, True]
    assert result.errors == ["-", "AssertionError", "-"]
    assert result.score == pytest.approx(2 / 3)


@pytest.mark.parametrize("code", ["while True: pass", "import time\ntime.sleep(60)"])
def test_hanging_code_scores_zero_and_returns_on_time(code):
    start = time.monotonic()
    result = execute(code + "\n" + ADD, TESTS, timeout=2)
    assert (result.status, result.score) == ("timeout", 0.0)
    assert time.monotonic() - start < 5


@pytest.mark.parametrize("exit_call", [
    "import sys\nsys.exit(0)",  # an exit-code-only checker would score this 1.0
    "raise SystemExit(0)",
    "import os\nos._exit(0)",
    "import os, signal\nos.kill(os.getpid(), signal.SIGKILL)",
])
def test_exiting_before_the_tests_scores_zero(exit_call):
    assert run_tests(ADD + exit_call, TESTS) == 0.0


def test_exiting_inside_the_function_fails_that_test():
    code = "import sys\ndef add(a, b):\n    if a < 0:\n        sys.exit(0)\n    return a + b\n"
    assert run_tests(code, TESTS) == pytest.approx(2 / 3)


def test_printing_a_fake_success_line_scores_zero():
    code = (
        "import os\n"
        "print('All 3 tests passed!')\n"
        "line = b'0' * 32 + b' ... - - -\\n'\n"
        "for fd in range(10):  # includes the runner's private copy of stdout\n"
        "    try:\n"
        "        os.write(fd, line)\n"
        "    except OSError:\n"
        "        pass\n"
        "os._exit(0)\n"
    )
    result = execute(code, TESTS)
    assert (result.status, result.score) == ("crash", 0.0)


def test_syntax_error_scores_zero():
    result = execute("def add(a, b)\n    return a + b\n", TESTS)
    assert (result.status, result.errors, result.score) == ("error", ["SyntaxError"], 0.0)


def test_wrong_function_name_scores_zero():
    result = execute(ADD.replace("add", "plus"), TESTS)
    assert (result.status, result.errors, result.score) == ("ok", ["NameError"] * 3, 0.0)


@pytest.mark.parametrize("code", [
    "def exec(*args, **kwargs):\n    pass\nprint = type = None\n",
    "import builtins\nbuiltins.exec = builtins.compile = lambda *args, **kwargs: None\n",
])
def test_rebinding_builtins_does_not_fake_passes(code):
    assert run_tests(code + "def add(a, b):\n    return None\n", TESTS) == 0.0


def test_main_block_does_not_run():
    code = ADD + 'if __name__ == "__main__":\n    print(add(int(input()), 1))\n'
    assert run_tests(code, TESTS) == 1.0


def test_reading_stdin_at_top_level_scores_zero():
    result = execute("x = input()\n" + ADD, TESTS)
    assert (result.status, result.errors) == ("error", ["EOFError"])


def test_memory_hog_scores_zero():
    result = execute("x = bytearray(3 << 30)\n" + ADD, TESTS)
    assert (result.status, result.errors) == ("error", ["MemoryError"])


def test_code_cannot_start_processes():
    code = "import subprocess\ndef add(a, b):\n    subprocess.run(['true'])\n    return a + b\n"
    assert run_tests(code, TESTS) == 0.0


def test_parent_environment_is_not_inherited(monkeypatch):
    monkeypatch.setenv("SECRET_TOKEN", "hunter2")
    code = "import os\ndef leaked():\n    return os.environ.get('SECRET_TOKEN')\n"
    assert run_tests(code, ["assert leaked() is None"]) == 1.0


def test_string_hashing_is_reproducible():
    expected = subprocess.run([sys.executable, "-c", "print(hash('mbpp'))"], env={"PYTHONHASHSEED": "0"},
                              capture_output=True, text=True, check=True).stdout.strip()
    assert run_tests("", [f"assert hash('mbpp') == {expected}"]) == 1.0


def test_setup_runs_after_the_code():
    # MBPP's test_setup_code builds inputs from classes that the solution defines.
    code = "class Node:\n    def __init__(self, val):\n        self.val = val\ndef value(node):\n    return node.val\n"
    assert run_tests(code, ["assert value(root) == 1"], setup="root = Node(1)") == 1.0


def test_no_tests_is_an_error():
    with pytest.raises(ValueError):
        execute(ADD, [])


@pytest.mark.xfail(strict=True, reason="assert-based tests can't catch an object whose __eq__ always returns "
                                       "True; look for it in the Phase 6 reward-hacking audit")
def test_always_equal_object_scores_zero():
    code = "class AlwaysEqual:\n    def __eq__(self, other):\n        return True\n"
    assert run_tests(code + "def add(a, b):\n    return AlwaysEqual()\n", TESTS) == 0.0
