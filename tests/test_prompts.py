import pytest
from evalplus.data import get_human_eval_plus, get_mbpp_plus
from evalplus.provider.utility import make_raw_chat_prompt
from transformers import AutoTokenizer

from src.data import load_problems
from src.prompts import (INSTRUCTION, RESPONSE_PREFIX, build_prompt, completion_code, defined_names, extract_code,
                         sanitize)


@pytest.mark.parametrize("name, evalplus_problems", [("mbppplus", get_mbpp_plus), ("humanevalplus", get_human_eval_plus)])
def test_prompts_match_evalplus(name, evalplus_problems):
    """Same prompt text as EvalPlus, so our baseline is comparable with Qwen's published numbers."""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")
    theirs = evalplus_problems()
    for p in load_problems(name):
        expected = make_raw_chat_prompt(theirs[p["task_id"]]["prompt"], INSTRUCTION, RESPONSE_PREFIX, tokenizer)
        assert build_prompt(p["problem"], tokenizer) == expected, p["task_id"]


def test_sanitize_keeps_what_the_tests_reach():
    code = (
        "import math\n"
        "SCALE = 2\n"
        "class Pair:\n    def __init__(self, a):\n        self.a = a\n"
        "def helper(x):\n    return x * SCALE\n"
        "@staticmethod\ndef unused():\n    pass\n"
        "def f(pairs):\n    return sum(helper(p.a) for p in pairs)\n"
        "result = f([Pair(1)])\n"
        "assert f([Pair(1)]) == 999  # the model's own wrong example\n"
        'if __name__ == "__main__":\n    print(f([]))\n'
    )
    kept = sanitize(code, ["assert f([Pair(1), Pair(2)]) == 6"])
    assert kept == ("import math\nSCALE = 2\nclass Pair:\n    def __init__(self, a):\n        self.a = a\n"
                    "def helper(x):\n    return x * SCALE\ndef f(pairs):\n    return sum(helper(p.a) for p in pairs)\n")


def test_sanitize_starts_from_names_the_tests_use_but_do_not_define():
    code = "def f(x):\n    return x\nresults = f(undefined)\n"
    test = "results = [1]\nfor r in results:\n    assert f(r) == r"
    assert sanitize(code, [test]) == "def f(x):\n    return x\n"


def test_sanitize_drops_a_cut_off_tail():
    code = "def f(x):\n    return x\nassert f(1) == 1\nassert f(2) == 2\nassert f(3) =="
    assert sanitize(code, ["assert f(3) == 3"]) == "def f(x):\n    return x\n"


def test_completion_code():
    assert completion_code("def f(): return 1\n```\nExplanation.") == "def f(): return 1\n"
    assert completion_code("def f(): return 1") == "def f(): return 1"  # cut off before the fence


def test_defined_names():
    code = "import math\nfrom os import path as p\nX, (Y, Z) = 1, (2, 3)\nclass C: pass\ndef f():\n    inner = 1\n"
    assert defined_names(code) == {"math", "p", "X", "Y", "Z", "C", "f"}
    with pytest.raises(SyntaxError):
        defined_names("def f(:")


def test_python_block():
    text = "Here you go:\n```python\ndef f():\n    return 1\n```\nThis returns 1."
    assert extract_code(text) == "def f():\n    return 1\n"


def test_first_python_block_wins():
    text = "```python\ndef f(): return 1\n```\nUsage:\n```python\nprint(f())\n```"
    assert extract_code(text) == "def f(): return 1\n"


def test_python_block_preferred_over_earlier_untagged_block():
    text = "Output:\n```\n1\n```\nCode:\n```py\ndef f(): return 1\n```"
    assert extract_code(text) == "def f(): return 1\n"


def test_untagged_block_is_the_fallback():
    assert extract_code("```\ndef f(): return 1\n```") == "def f(): return 1\n"


def test_unterminated_block_runs_to_the_end():
    assert extract_code("```python\ndef f():\n    return") == "def f():\n    return"


def test_no_code_block():
    assert extract_code("def f(): return 1") is None
    assert extract_code("```bash\npip install foo\n```") is None
