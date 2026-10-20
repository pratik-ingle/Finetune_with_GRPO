"""Run untrusted code against assert-style tests in a resource-limited subprocess."""
import json
import math
import os
import secrets
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

# Runs in the child. The job (including the nonce) arrives on stdin, so it never appears in
# /proc/<pid>/cmdline. The code under test gets its own namespace and its output goes to
# /dev/null; only this runner writes the result line, through a private copy of stdout.
_RUNNER = r"""
import json, os, resource, sys

def main():
    job = json.loads(sys.stdin.read())
    out = os.dup(1)
    null = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        os.dup2(null, fd)
    for res, value in ((resource.RLIMIT_AS, job["memory"]), (resource.RLIMIT_CPU, job["cpu"]),
                       (resource.RLIMIT_FSIZE, 16 << 20), (resource.RLIMIT_CORE, 0),
                       (resource.RLIMIT_NPROC, 0)):  # NPROC 0: no fork, so nothing outlives us
        resource.setrlimit(res, (value, value))

    # Private references, taken before the code can rebind builtins or patch modules.
    exec_, type_, write, base = exec, type, os.write, BaseException
    nonce, tests = job["nonce"], tuple(job["tests"])
    ns = {"__name__": "solution"}  # not "__main__": `if __name__ == "__main__":` blocks don't run
    marks, errors = "", []
    try:
        exec_(job["code"], ns)
        exec_(job["setup"], ns)  # after the code: MBPP setup builds inputs from classes it defines
    except base as e:
        marks, errors = "!", [type_(e).__name__]
    else:
        for t in tests:
            try:
                exec_(t, ns)
                marks += "."
                errors.append("-")
            except base as e:
                marks += "F"
                errors.append(type_(e).__name__)
    write(out, (nonce + " " + marks + " " + " ".join(errors) + "\n").encode())

main()
"""

# The code never sees the parent's environment (API keys, HF tokens). A fixed hash seed makes set
# iteration order, and so results, reproducible; it is why we pass -s -P instead of -I, which would
# also ignore PYTHONHASHSEED. One BLAS thread keeps numpy importable under RLIMIT_NPROC=0.
_ENV = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


@dataclass
class ExecResult:
    status: str  # "ok": tests ran; "error": the code or setup raised; "timeout"; "crash": died without reporting
    passed: list[bool] = field(default_factory=list)  # one per test when status == "ok"
    errors: list[str] = field(default_factory=list)  # exception type per test ("-" if it passed), or the code's own

    @property
    def score(self) -> float:
        return sum(self.passed) / len(self.passed) if self.passed else 0.0


def execute(code: str, tests: list[str], setup: str = "", timeout: float = 10.0, memory: int = 2 << 30) -> ExecResult:
    """Run `code`, then `setup`, then each test independently, in a fresh limited subprocess.

    `memory` caps the address space in bytes. 2 GiB keeps a dozen parallel reward workers within RAM
    during training; eval uses 4 GiB to match EvalPlus, whose extra tests need it (e.g. Mbpp/255).
    """
    if not tests:
        raise ValueError("no tests to run")
    nonce = secrets.token_hex(16)  # the code can't forge the result line without it
    # The CPU limit only backstops a child we never reap; with no threads, CPU time <= wall time, so
    # the extra second makes the wall-clock timeout fire first and report "timeout", not "crash".
    cpu = math.ceil(timeout) + 1
    job = json.dumps({"nonce": nonce, "code": code, "setup": setup, "tests": tests, "cpu": cpu, "memory": memory})
    with tempfile.TemporaryDirectory() as d:
        proc = subprocess.Popen([sys.executable, "-s", "-P", "-c", _RUNNER], cwd=d, env=_ENV,
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                start_new_session=True)
        try:
            out = proc.communicate(job.encode(), timeout=timeout)[0]
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            proc.communicate()
            return ExecResult("timeout")
        _kill_group(proc)

    for line in out.decode(errors="replace").splitlines():
        if line.startswith(nonce + " "):
            _, marks, rest = line.split(" ", 2)
            if marks == "!":
                return ExecResult("error", errors=[rest])
            if len(marks) == len(tests) and set(marks) <= {".", "F"}:
                return ExecResult("ok", [m == "." for m in marks], rest.split())
    return ExecResult("crash")  # exited early, was killed by a limit, or never reached the tests


def run_tests(code: str, tests: list[str], setup: str = "", timeout: float = 10.0) -> float:
    """Fraction of asserts passed; each assert runs independently for partial credit."""
    return execute(code, tests, setup, timeout).score


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
