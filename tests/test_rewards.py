import pytest

from src.rewards import format_reward, unit_test_reward

TESTS = ["assert add(1, 2) == 3", "assert add(-1, 1) == 0", "assert add(2, 2) == 4"]
# Completions start inside the ```python block that the prompt opens.
COMPLETIONS = [
    "def add(a, b):\n    return a + b\n```\nThis adds the numbers.",
    "def add(a, b):\n    return abs(a) + b\n```",  # passes 2 of 3
    "def add(a, b):\n    return a + b\n",  # right code, but cut off before the closing fence
    "def add(a, b)\n    return a + b\n```",  # syntax error
    "```",  # empty block
    "def add(a, b):\n    return a + b\nassert add(1, 1) == 3  # its own wrong example\n```",  # sanitized away
]


def test_unit_test_reward():
    rewards = unit_test_reward(COMPLETIONS, tests=[TESTS] * len(COMPLETIONS))
    assert rewards == pytest.approx([1.0, 2 / 3, 1.0, 0.0, 0.0, 1.0])


def test_unit_test_reward_uses_setup_column():
    code = "class Node:\n    def __init__(self, val):\n        self.val = val\n```"
    rewards = unit_test_reward([code], tests=[["assert root.val == 1"]], test_setup_code=["root = Node(1)"])
    assert rewards == [1.0]


def test_format_reward():
    assert format_reward(COMPLETIONS) == [1.0, 1.0, 0.0, 0.0, 0.0, 1.0]
