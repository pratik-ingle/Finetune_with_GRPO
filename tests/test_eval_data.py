"""The eval datasets load as expected, and their reference solutions pass under the eval limits."""
from concurrent.futures import ThreadPoolExecutor

import pytest

from eval.score import MEMORY, TIMEOUT
from src.data import load_problems
from src.prompts import sanitize
from src.sandbox import execute

SIZES = {"mbpp_train": 374, "mbpp_validation": 90, "mbpp_test": 500, "mbppplus": 378, "humanevalplus": 164}

# task_id -> why its reference fails. MBPP's own references are checked in test_mbpp_references.py.
KNOWN_BAD = {
    "HumanEval/32": "some extra-test polynomials have coefficients near 1e9, so float64 rounding alone puts "
                    "|poly(root)| far above the 1e-4 tolerance; EvalPlus fails its own reference here too",
}


@pytest.mark.parametrize("name", SIZES)
def test_dataset_loads(name):
    problems = load_problems(name)
    assert len(problems) == SIZES[name]
    for p in problems:
        prompt = p["problem"]
        assert p["entry_point"] in prompt
        assert 1 <= p["n_base"] <= len(p["tests"])
        if name != "humanevalplus":
            assert p["tests"][0] in prompt
            # A few MBPP problems repeat their first assert (e.g. Mbpp/37 lists it three times).
            hidden = [t for t in p["tests"][1:] if t != p["tests"][0]]
            assert not any(t in prompt for t in hidden), "only the first test may appear in the prompt"


@pytest.mark.parametrize("name", ["mbppplus", "humanevalplus"])
def test_reference_solutions_pass(name):
    """Through the same path eval/score.py takes: sanitize, then run under the eval limits."""
    def run(p):
        return execute(sanitize(p["reference"], p["tests"], p["test_setup_code"]), p["tests"], p["test_setup_code"],
                       TIMEOUT, MEMORY)

    problems = load_problems(name)
    with ThreadPoolExecutor(max_workers=12) as pool:
        results = pool.map(run, problems)
        failing = {p["task_id"]: (r.status, r.errors) for p, r in zip(problems, results) if r.score < 1.0}
    assert {t: f for t, f in failing.items() if t not in KNOWN_BAD} == {}, "references that newly fail"
    assert [t for t in KNOWN_BAD if t.split("/")[0].lower() in name and t not in failing] == [], \
        "listed as bad but now pass; update KNOWN_BAD"
