"""Prompt building and code extraction, shared by training and eval so the two can't drift apart."""
import ast
import re

# EvalPlus's instruct prompt, which Qwen's published HumanEval/MBPP numbers use. It pre-fills the start
# of the assistant's reply up to an opened ```python block, so the model writes only code. Without the
# pre-fill, Qwen2.5-Coder-0.5B answers with "```\npython\n...", which breaks extraction.
INSTRUCTION = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
RESPONSE_PREFIX = "Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:"
_SPLIT = "<<<COMPLETION>>>"
_URL = re.compile(r"https?://\S+")  # some MBPP descriptions end with the URL they were copied from

# ```lang\n...```; an unterminated block (output cut off at the length limit) runs to the end.
_FENCE = re.compile(r"```([\w+-]*)[ \t]*\n(.*?)(?:```|\Z)", re.DOTALL)
_PYTHON_TAGS = {"python", "py", "python3"}


def build_prompt(problem: str, tokenizer) -> str:
    """The prompt for `problem` (an MBPP description or a HumanEval stub) as text in the model's chat template.

    It ends inside the assistant's ```python block. TRL and vLLM take the string as-is, so training and
    eval see exactly the same tokens. Use completion_code() on what the model writes after it.
    """
    messages = [
        {"role": "user", "content": f"{INSTRUCTION}\n```\n{problem.strip()}\n```\n"},
        {"role": "assistant", "content": f"{RESPONSE_PREFIX}\n```python\n{_SPLIT}\n```\n"},
    ]
    # Render a whole reply and cut it at the placeholder, as EvalPlus does: continue_final_message
    # would strip the newline after ```python.
    return tokenizer.apply_chat_template(messages, tokenize=False).split(_SPLIT)[0]


def mbpp_problem(text: str, visible_tests: list[str]) -> str:
    """An MBPP problem as EvalPlus writes it: the description, minus any source URL, plus the visible
    asserts. EvalPlus, and so eval, shows one, which gives away the function's name and signature."""
    tests = "\n".join(visible_tests)
    return f'"""\n{_URL.sub("", text)}\n{tests}\n"""\n'


def completion_code(completion: str) -> str:
    """The code in a completion of build_prompt(): everything up to the closing fence."""
    return extract_code("```python\n" + completion)


def extract_code(text: str) -> str | None:
    """Code from the first ```python block, else the first untagged block; None if there is neither."""
    blocks = _FENCE.findall(text)
    for tag, body in blocks:
        if tag.lower() in _PYTHON_TAGS:
            return body
    for tag, body in blocks:
        if not tag:
            return body
    return None


def sanitize(code: str, tests: list[str], setup: str = "") -> str:
    """Keep the imports, plus the top-level functions, classes and assignments the tests can reach.

    This drops the model's own example calls, asserts and prints, so a stray `assert f(x) == wrong` can't
    fail a correct solution. EvalPlus's sanitizer does the same starting from the entry point; we start
    from every name the tests and setup use, so helpers that only the tests touch (MBPP's Pair class)
    survive. Like EvalPlus, it drops everything from the first syntax error on, which rescues outputs
    that loop on asserts until the length limit cuts a line in half.
    """
    lines = code.splitlines()
    end = len(lines)
    while True:
        try:
            tree = ast.parse("\n".join(lines[:end]))
            break
        except SyntaxError as e:
            end = min(end - 1, (e.lineno or end) - 1)
        except ValueError:  # null bytes
            return code
        if end <= 0:
            return code
    defs: dict[str, list[ast.stmt]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defs.setdefault(node.name, []).append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for name in {n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)}:
                defs.setdefault(name, []).append(node)

    todo = set().union(*(_free_names(ast.parse(s)) for s in [*tests, setup]))
    keep, seen = set(), set()
    while todo:
        name = todo.pop()
        seen.add(name)
        for node in defs.get(name, []):
            keep.add(id(node))
            todo |= _free_names(node) - seen

    kept = []
    for node in tree.body:
        if id(node) in keep or isinstance(node, (ast.Import, ast.ImportFrom)):
            start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
            kept.append("\n".join(lines[start - 1:node.end_lineno]))
    return "\n".join(kept) + "\n"


def _free_names(node: ast.AST) -> set[str]:
    """Names that `node` reads but doesn't bind itself (ignoring scopes, which is close enough here).

    Starting from the tests' free names, not every name they mention, matters: MBPP+'s extra tests
    define `inputs` and `results`, and a model line like `results = f(x)` must not be pulled in.
    """
    read, bound = set(), set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            (read if isinstance(n.ctx, ast.Load) else bound).add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(n.name)
        elif isinstance(n, ast.arg):
            bound.add(n.arg)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            bound.update((a.asname or a.name).split(".")[0] for a in n.names)
    return read - bound


def defined_names(code: str) -> set[str]:
    """Names bound at the top level of `code`: functions, classes, assignments and imports.

    Raises SyntaxError (or ValueError for null bytes) if the code doesn't parse.
    """
    names = set()
    for node in ast.parse(code).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
    return names
