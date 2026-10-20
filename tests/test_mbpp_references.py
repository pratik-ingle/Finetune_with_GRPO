"""Every MBPP reference solution must score 1.0 on test_list + challenge_test_list, in every split."""
from concurrent.futures import ThreadPoolExecutor

import pytest
from datasets import concatenate_datasets, load_dataset

from src.prompts import sanitize
from src.sandbox import execute

# task_id -> why its reference fails. Keep failures listed here instead of skipping them, so a sandbox
# change that breaks (or fixes) a problem shows up as a test failure. All 974 passed on 2026-10-19,
# also under 20 different hash seeds.
KNOWN_BAD: dict[int, str] = {}


@pytest.fixture(scope="module")
def results():
    ds = load_dataset("google-research-datasets/mbpp", "full")
    rows = concatenate_datasets([ds[split] for split in ds])
    assert len(rows) == 974

    def run(row):
        # Raw, and sanitized as the reward function does: neither may lose anything the tests need.
        tests, setup = row["test_list"] + row["challenge_test_list"], row["test_setup_code"]
        raw = execute(row["code"], tests, setup)
        return row["task_id"], min(raw, execute(sanitize(row["code"], tests, setup), tests, setup),
                                   key=lambda r: r.score)

    with ThreadPoolExecutor(max_workers=12) as pool:
        return dict(pool.map(run, rows))


def test_reference_solutions_pass(results):
    failing = {task_id: (r.status, r.errors) for task_id, r in results.items() if r.score < 1.0}
    assert {t: f for t, f in failing.items() if t not in KNOWN_BAD} == {}, "references that newly fail"
    assert [t for t in KNOWN_BAD if t not in failing] == [], "listed as bad but now pass; update KNOWN_BAD"
