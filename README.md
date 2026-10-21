# GRPO on MBPP: Learning Path

**Goal:** fine-tune a small code model with GRPO using unit-test rewards on MBPP, and measure the result with an evaluation harness you can trust.

**Hardware:** RTX 4070 SUPER (12 GB VRAM), NVIDIA driver 535 (CUDA ≤ 12.2), 30 GB RAM, 16 cores.

| Phase | Done when |
|---|---|
| [0. Concepts](#phase-0-learn-the-concepts) | You can explain why a group where every sample gets the same reward gives zero gradient |
| [1. Environment](#phase-1-environment) | vLLM generates a completion from Qwen2.5-Coder-0.5B-Instruct |
| [2. Sandbox + reward](#phase-2-sandbox-and-reward) | Pytest suite passes, including the reward-hacking cases |
| [3. Eval harness + baseline](#phase-3-eval-harness-and-baseline) | Baseline is close to published numbers and reproducible |
| [4. Training data](#phase-4-training-data) | Filtered train set with a `tests` column |
| [5. GRPO with TRL](#phase-5-grpo-with-trl) | Validation pass@1 goes up during training |
| [6. Evaluate + analyze](#phase-6-evaluate-and-analyze) | Paired comparison of base vs. best checkpoint, plus a reward-hacking audit |
| [7. Write GRPO yourself](#phase-7-optional-write-the-grpo-loop-yourself) | Your loop's reward curve matches TRL's on 0.5B |

---

## Phase 0: Learn the concepts

Read in this order:

- [ ] **MBPP paper**, [arXiv 2108.07732](https://arxiv.org/abs/2108.07732): dataset format, and why the tests appear in the prompt.
- [ ] **Codex paper**, [arXiv 2107.03374](https://arxiv.org/abs/2107.03374), §2.1: the unbiased pass@k estimator.
- [ ] **DeepSeekMath**, [arXiv 2402.03300](https://arxiv.org/abs/2402.03300), GRPO section: group-relative advantages with no value model.
- [ ] **Dr. GRPO**, [arXiv 2503.20783](https://arxiv.org/abs/2503.20783), and **DAPO**, [arXiv 2503.14476](https://arxiv.org/abs/2503.14476): known biases in plain GRPO, and why the KL penalty is often dropped.
- [ ] **EvalPlus**, [arXiv 2305.01210](https://arxiv.org/abs/2305.01210): why 3 asserts per problem is weak evidence, and what MBPP+ adds.
- [ ] **TRL** `GRPOTrainer` **docs.**

**The key idea:** for each prompt you sample G completions, and each one's advantage is `(r − mean) / std` within that group. **If all G completions get the same reward, that prompt gives zero gradient.** Most practical choices follow from this: how you filter data, partial-credit rewards, group size and temperature.

---

## Phase 1: Environment

- [ ] Create a separate env instead of using base Python 3.13 (`environment.yml` pins torch, vLLM and TRL to versions that work together):
  ```bash
  conda env create -f environment.yml
  conda activate grpo
  ```
- [ ] Check the driver. Driver 535 limits you to CUDA 12.2. The default PyPI torch 2.13 / vLLM 0.31 wheels are CUDA 13 builds that need driver 580+, so `environment.yml` uses their `+cu129` builds, which run on 535 through CUDA minor-version compatibility. If you see PTX or "unsupported toolchain" errors, upgrade the driver (580+) rather than pinning old versions.
- [ ] Smoke test: `python scripts/smoke_vllm.py` loads `Qwen/Qwen2.5-Coder-0.5B-Instruct` in vLLM and generates one completion.

---

## Phase 2: Sandbox and reward

This matters more than any other code in the project. A bug here quietly trains the model on the wrong signal.

```python
# src/sandbox.py
import secrets, subprocess, sys, tempfile

def run_tests(code: str, tests: list[str], setup: str = "", timeout: float = 10.0) -> float:
    """Fraction of asserts passed; each assert runs independently for partial credit."""
    nonce = secrets.token_hex(8)  # model can't forge the result line
    program = f"""import resource
resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
{code}
{setup}  # after the code: MBPP setup builds inputs from classes the code defines

_n = 0
for _t in {tests!r}:
    try:
        exec(_t, globals())
        _n += 1
    except BaseException:
        pass
print("{nonce}", _n)
"""
    with tempfile.TemporaryDirectory() as d:
        try:
            out = subprocess.run([sys.executable, "-I", "-c", program], cwd=d,
                                 stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                 timeout=timeout, start_new_session=True).stdout
        except subprocess.TimeoutExpired:
            return 0.0
    for line in reversed(out.splitlines()):
        if line.startswith(nonce):
            return int(line.split()[1]) / len(tests)
    return 0.0  # crashed, exited early, or never reached the tests
```

```python
# src/rewards.py: TRL passes extra dataset columns (here `tests`) as kwargs
from concurrent.futures import ThreadPoolExecutor
from src.prompts import completion_code, sanitize
from src.sandbox import run_tests

_pool = ThreadPoolExecutor(max_workers=12)

def unit_test_reward(completions, tests, **kwargs):
    # Prompts end inside a ```python block (see Phase 3), so each completion is a plain string of code.
    codes = [sanitize(completion_code(c), t) for c, t in zip(completions, tests)]
    return list(_pool.map(lambda ct: run_tests(*ct), zip(codes, tests)))
```

- [ ] Implement `run_tests` (above). `src/sandbox.py` hardens this sketch: the code runs in its own namespace so shadowing `exec` or `print` can't fake passes, the nonce arrives on stdin instead of the command line, forking is blocked, and the parent's environment variables are not passed on.
- [ ] Implement `extract_code` in `src/prompts.py`. Eval will import the same function.
- [ ] Implement `unit_test_reward`, plus a small `format_reward` (weight about 0.1) for a ````python` block that parses with `ast`.
- [ ] Write the pytest suite **before any training**:
  - [ ] Every MBPP reference solution scores 1.0. A few MBPP problems are flaky, so list those explicitly instead of skipping them silently.
  - [ ] `while True: pass` scores 0.0.
  - [ ] `import sys; sys.exit(0)` scores 0.0. A checker that only looks at the exit code would give this 1.0, which is a classic reward hack.
  - [ ] Printing a fake success line scores 0.0.
  - [ ] A syntax error or the wrong function name scores 0.0.

**Safety:** subprocess plus resource limits is fine for MBPP experiments on your own machine, but the model's code can still read and write your files. For stronger isolation, run the executor in Docker with `--network none`.

---

## Phase 3: Eval harness and baseline

Build this **before any training**.

| Split | Use |
|---|---|
| MBPP train (374) | GRPO training |
| MBPP validation (90) | Choosing checkpoints and hyperparameters |
| MBPP test (500) | Final report only |
| MBPP+ (`evalplus/mbppplus`, 378) | Final report; catches overfitting to the few visible tests |
| HumanEval+ | Regression check: did general coding get worse? |

**Contamination:** 108 of MBPP+'s 378 problems are MBPP train task IDs (601–974). `eval/score.py` also reports `mbppplus_dedup` (270 problems) without them; use that one after training.

- [ ] **One shared module** (`src/prompts.py`) for prompt building and code extraction, imported by both training and eval. A mismatch between training and eval prompts is the most common silent bug.
  - `build_prompt` renders EvalPlus's instruct prompt, which Qwen's published numbers use, as one string with the model's chat template. That includes the pre-filled start of the reply, which ends with an opened ````python` block. Without the pre-fill, 0.5B answers ````\npython\n…` and every sample fails. TRL's GRPO trainer can't continue a pre-filled assistant turn, so it gets the same string as a plain-text prompt.
  - `sanitize` keeps the imports and the definitions the tests can reach, and drops the model's own example asserts and prints. EvalPlus does the same before scoring, and it's worth +2 to +6 points. Unlike EvalPlus's sanitizer, ours starts from every name the tests use, so helper classes like MBPP's `Pair` survive.
- [ ] `eval/generate.py`**:** uses vLLM to write `samples.jsonl` (task_id, raw output, extracted code). Run greedy (n=1) and sampled (n=10–20, temperature 0.8).
- [ ] `eval/score.py`**:** reports greedy pass@1, unbiased pass@k, and a breakdown of failures: syntax error, wrong name, assert failure, timeout, missing code block.
  ```python
  import numpy as np

  def pass_at_k(n: int, c: int, k: int) -> float:
      """Unbiased estimator from the Codex paper: n samples, c correct."""
      if n - c < k:
          return 1.0
      return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
  ```
- [ ] **Statistics:** with 500 problems, pass@1 has a standard error of about 2 points. Compare models with a paired test across problems (McNemar's test or paired bootstrap), not by eye.
- [ ] **Cross-check with EvalPlus:** run the `evalplus` CLI on the same samples and confirm it agrees with your scorer.
- [ ] **Cross-check with published numbers:** compare your base-model score with Qwen's published numbers. If you're more than about 10 points off, your prompt or code extraction is broken, not the model.

**Done when:** reference solutions score about 100%, your baseline is close to the published numbers, and repeated greedy runs give identical results.

**Running it:** `scripts/baseline.sh` runs everything below for both models. For a single run:

```bash
python -m eval.generate --model Qwen/Qwen2.5-Coder-0.5B-Instruct --out runs/eval/coder-0.5b-greedy
python -m eval.score runs/eval/coder-0.5b-greedy        # writes <dataset>.results.jsonl + summary.json
python -m eval.evalplus_check runs/eval/coder-0.5b-greedy
python -m eval.score --compare RUN_A RUN_B              # paired bootstrap CI + McNemar per dataset
```

**Baseline (2026-10-20), greedy pass@1 in %.** MBPP and HumanEval use the base tests, and MBPP+ and HumanEval+ add the extra tests. MBPP and MBPP+ are the 378 MBPP+ problems, which is what Qwen reports:

| Model | MBPP val | MBPP test | MBPP | MBPP+ | MBPP+ dedup | HumanEval | HumanEval+ |
|---|---|---|---|---|---|---|---|
| 0.5B (ours) | 37.8 | 33.4 | 51.6 | 43.7 | 45.9 | 59.8 | 56.1 |
| 0.5B (Qwen) | | | 52.4 | 43.7 | | 61.6 | 57.3 |
| 1.5B (ours) | 50.0 | 51.2 | 69.3 | 59.3 | 60.0 | 68.9 | 64.0 |
| 1.5B (Qwen) | | | 69.2 | 59.4 | | 70.7 | 66.5 |

Sampled (T=0.8, n=10) on MBPP test: pass@1 / pass@10 = 31.9 / 57.8 (0.5B), 47.0 / 71.8 (1.5B). The EvalPlus CLI agrees with `eval/score.py` on 98.7–99.4% of samples. `eval/evalplus_check.py` explains the differences: the HF copy of MBPP+ differs from EvalPlus's release on a few problems, and EvalPlus fails correct but slow code.

**Gotchas found while building this:**

- **Determinism:** vLLM greedy decoding isn't deterministic by default. About half of all outputs changed between identical runs. `eval/generate.py` sets `VLLM_BATCH_INVARIANT=1`, which makes repeats byte-identical.
- **Memory:** some MBPP+ extra tests need more than the training sandbox's 2 GiB, so eval uses EvalPlus's 4 GiB limit and a 20 s timeout.
- **Bad tests in the data:** HumanEval/32's extra tests can't be passed in float64, even by the reference, in EvalPlus too. The HF copy of that test also had a bug (fixed in `src/data.py`).

---

## Phase 4: Training data

- [ ] **Prompt:** the problem text plus **only the first assert**, which shows the function name and signature. Reward on all asserts, including `challenge_test_list`. If you show all 3 tests and reward on the same 3, the model can learn to hardcode the answers.
- [ ] **Difficulty filter:** sample 8 completions per train problem from the base model. Drop problems where all 8 or none pass, since those give zero gradient (see Phase 0). Expect roughly half to survive.
- [ ] Store the tests as a `tests` column. TRL passes extra columns to reward functions as keyword arguments.

**Running it:** `python -m scripts.make_train_set --model Qwen/Qwen2.5-Coder-1.5B-Instruct --out data/train-coder-1.5b.jsonl`. The filter samples with the training settings (temperature 1.0, 512 tokens) and scores with the training reward. `train_grpo.py` turns each row into a `build_prompt` string.

**Result (2026-10-21):** the filter kept 183 of 374 problems for 0.5B-Instruct, 222 for 1.5B-Instruct and 186 for the 0.5B base model. A looser filter, "drop a problem only if all 8 rewards are equal", would keep 79 more for 0.5B. None of those 79 ever passes fully: their partial credit mostly comes from passing the one visible assert, which is the hardcoding incentive. A binary reward would also get no gradient from them.

---

## Phase 5: GRPO with TRL

- [ ] Start with **Qwen2.5-Coder-0.5B-Instruct** to debug the pipeline end to end.
- [ ] Move to **Qwen2.5-Coder-1.5B-Instruct with LoRA**. Full fine-tuning of 1.5B doesn't fit in 12 GB: Adam's optimizer state alone is about 18 GB.

```python
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

cfg = GRPOConfig(
    output_dir="runs/coder1.5b-grpo", learning_rate=2e-5,   # LoRA; full-FT would be ~1e-6
    num_generations=8, per_device_train_batch_size=8, gradient_accumulation_steps=4,
    max_completion_length=512, temperature=1.0, beta=0.0, num_train_epochs=3,
    reward_weights=[1.0, 0.1], bf16=True, gradient_checkpointing=True,
    use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=0.3,
    logging_steps=1, save_steps=50, report_to="wandb",
)
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
    reward_funcs=[unit_test_reward, format_reward],
    args=cfg,
    train_dataset=train_ds,
    peft_config=LoraConfig(r=32, lora_alpha=64, target_modules="all-linear", task_type="CAUSAL_LM"),
)
trainer.train()
```

TRL renames config fields between releases, so check these names against the version you install.

**What to watch:**

- [ ] Train reward.
- [ ] **Validation pass@1 every ~50 steps**, from a callback or by running the eval on saved checkpoints.
- [ ] `frac_reward_zero_std`: if it's high, filter harder or raise G or the temperature.
- [ ] Completion length: a collapse or blow-up is a warning sign.

**If 1.5B runs out of memory:** lower `vllm_gpu_memory_utilization` or `max_completion_length`. As a last resort, set `use_vllm=False`, which is slower.

**Running it:** `scripts/phase56.sh` runs every training run in this phase and in Phase 6, each followed by its validation curve. A single run:

```bash
python train_grpo.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --train-data data/train-coder-1.5b.jsonl \
    --out runs/grpo/coder-1.5b-s0 --vllm-memory 0.45 --vllm-sleep
python -m eval.checkpoints runs/grpo/coder-1.5b-s0      # validation pass@1 for step 0 and every checkpoint
python -m scripts.summarize_runs runs/grpo/coder-1.5b-s*
```

**What** `train_grpo.py` **changes from the sketch above:**

- **Prompts** are `build_prompt` strings, not chat messages, because TRL can't continue the pre-filled reply in chat format.
- **dtype:** TRL 1.16 loads the model in float32 unless told otherwise. The frozen base is loaded in bf16 instead.
- **vLLM memory:** at `vllm_gpu_memory_utilization=0.3`, 1.5B leaves vLLM no room for its KV cache. `--vllm-memory 0.45 --vllm-sleep` fixes this: vLLM offloads itself during each training step, and the run peaks at 9.4 GB. Each 1.5B run then prints one harmless "CUDA Error … cumem_allocator" line at shutdown.
- **Batch shape:** each optimizer step uses 4 prompts × 8 samples, which keeps the number of steps fixed across group sizes. Three epochs are 135 steps for 0.5B (about 11 minutes at 4 s/step) and 165 for 1.5B (about 33 minutes at 10.5 s/step). learn.md's learning rate of 2e-5 worked without tuning.
- **Checkpoints** are saved every 25 steps, as LoRA adapters only. Every training completion and its reward also goes to `<run>/completions.jsonl` for the audit.

**Result (2026-10-21):** validation pass@1 rose in all 6 runs. Validation has 90 problems, so its standard error is about 5 points:

| Model | Step 0 | Best (step), seeds 0 / 1 / 2 | Last, seeds 0 / 1 / 2 |
|---|---|---|---|
| 1.5B | 52.2 | 61.1 (150) / 63.3 (165) / 58.9 (75) | 60.0 / 63.3 / 55.6 |
| 0.5B | 37.8 | 47.8 (100) / 44.4 (75) / 40.0 (75) | 46.7 / 43.3 / 37.8 |

Over training:

- **Train reward** went from about 0.6 to 0.7–0.8.
- `frac_reward_zero_std` rose from about 0.1 to 0.25–0.4, as more problems became fully solved and stopped giving a gradient.
- **Completion length** stayed flat, and nothing was cut off.

---

## Phase 6: Evaluate and analyze

- [ ] Pick the checkpoint on **validation**.
- [ ] Evaluate the base model and that checkpoint **once each** on MBPP test, MBPP+ and HumanEval+, using paired tests.
- [ ] **Check for reward hacking:** search completions for hardcoded test values, `sys.exit`, and try/except wrappers that swallow errors. A growing gap between MBPP and MBPP+ scores points to hardcoding.
- [ ] **Ablations** (each one teaches something):
  - [ ] Binary vs. partial-credit reward.
  - [ ] 1 vs. 3 visible tests in the prompt.
  - [ ] Group size G = 4, 8 or 16.
  - [ ] `beta` = 0 vs. 0.04.
  - [ ] Starting from the Instruct model vs. the base model. Expect larger gains from base, because RL also teaches the output format.
- [ ] RL is noisy, so use 2–3 random seeds before claiming a result.

**Expectations:** Qwen-Coder-Instruct is already heavily tuned, so MBPP gains may be only a few points. Without paired tests, those can be indistinguishable from noise.

**Running it:**

- `python -m scripts.final_eval runs/grpo/<run> ...` picks each run's best validation step. It evaluates that step and step 0 once each, on the test sets, in the same vLLM engine. It then stores the paired comparison, and `--report` prints the summary table.
- `python -m eval.audit --train <run>` audits training completions, and `--samples A B` compares two sets of eval samples.

**Result (2026-10-21):** the main runs, in greedy pass@1. Each cell is the base model's score, then the mean change over seeds [min, max]. "MBPP+ dedup" is the 270 MBPP+ problems outside MBPP's train split, with all tests.

| Model | Seeds | MBPP test | MBPP+ dedup | HumanEval+ |
|---|---|---|---|---|
| 1.5B-Instruct | 3 | 51.0, +3.2 [+1.8, +4.2] | 61.9, +3.2 [+0.4, +4.8] | 64.0, +5.5 [+3.7, +6.7] |
| 0.5B-Instruct | 3 | 33.4, +5.0 [+4.2, +6.0] | 46.3, +2.3 [+0.4, +4.8] | 54.9, −0.6 [−1.8, +1.2] |

For 1.5B, seeds 0 and 1 beat the base model on every test set, mostly with McNemar p < 0.05; seed 1, for example, gains 4.2 points on MBPP test (p = 0.005). Seed 2's validation curve peaked early, at step 75. On the test sets it gains between −0.7 and +4.3 points, and none of those changes is significant. So one seed in three would have shown nothing, which is why the seeds matter.

**Reward-hacking audit:**

- **No hacks found.** Across all 21 runs, late training completions and the test-set samples contain no exits, no catch-all `except` that hides errors, no always-equal objects and no introspection of the runner.
- **"hardcoded" flags are rare.** Between 0 and 10 per ~900 completions are flagged. On inspection they are domain constants, such as month names and the RGB value 255, or attempts that failed.
- **The MBPP–MBPP+ gap doesn't grow.** It is 7.4 points for the 0.5B base model and 5.2–9.6 after training. For 1.5B it is 9.3 before and 7.8–8.9 after.

**Ablations** on 0.5B, with seeds 0 and 1 for every row, so the main row here is not the 3-seed one above. Each cell is the change in points from the starting model, mean [min, max]. The "late" columns are training averages over the last 10% of steps.

| 0.5B config | MBPP test | MBPP+ dedup | HumanEval+ | Zero-std groups, late | Length, late |
|---|---|---|---|---|---|
| G=8, partial credit, 1 visible test, beta=0 (main) | +5.4 [+4.8, +6.0] | +3.3 [+1.9, +4.8] | −0.3 [−1.8, +1.2] | 0.28 | 274 |
| Binary reward | +4.4 [+4.0, +4.8] | +3.5 [+3.3, +3.7] | −2.1 [−2.4, −1.8] | 0.29 | 209 |
| 3 visible tests | +4.9 [+4.4, +5.4] | +4.4 [+3.0, +5.9] | +1.2 [+1.2, +1.2] | 0.21 | 270 |
| G=4 | +2.5 [+1.8, +3.2] | +1.5 [+0.4, +2.6] | −2.4 [−4.3, −0.6] | 0.29 | 170 |
| G=16 | +6.8 [+6.4, +7.2] | +7.0 [+5.2, +8.9] | −0.9 [−1.2, −0.6] | 0.16 | 235 |
| beta=0.04 | +3.1 [+2.6, +3.6] | +2.6 [+1.5, +3.7] | −0.3 [−1.8, +1.2] | 0.19 | 211 |
| Base model, not Instruct (starts at 34.8 / 43.0 / 34.1) | +4.1 [+3.2, +5.0] | +6.5 [+5.9, +7.0] | −9.1 [−9.1, −9.1] | 0.22 | 351 |

**What the ablations show.** With 2 seeds, differences of 1–2 points are within the seed spread.

- **Group size** matters most. G=16 gives the largest gains and G=4 the smallest. At G=4, more groups end up with all-equal rewards, so they contribute no gradient (Phase 0). G=16 also costs twice the samples per step.
- **Binary vs partial credit:** no clear difference. The Phase 4 filter already guarantees reward variance under both.
- **3 visible tests:** no hardcoding appeared, and the scores were no worse. 135 LoRA steps on 0.5B weren't enough for the model to find that hack. The risk is real in principle, but it didn't show up at this budget.
- **beta=0.04:** somewhat smaller MBPP gains, as expected when a KL penalty holds the policy near the base model.
- **Starting from the base model:** the largest MBPP+ gain, as predicted, but a 9–11 point drop on HumanEval.
  - **Missing import:** after training, the model stops writing `from typing import List`, which MBPP solutions never need. HumanEval stubs with `List[...]` hints then fail with `NameError` when the function is defined, in 21 of 164 problems.
  - **No end of turn:** it also never learns to stop. 40–70% of its training completions run to the 512-token limit.
  - So RL taught it MBPP's format, too narrowly.
- **0.5B doesn't transfer to HumanEval,** while 1.5B does (+5.5 on HumanEval+).

**Caveats:**

- The ablations ran on 0.5B only, with 2 seeds each.
- Validation has only 90 problems, so choosing the best of 8 checkpoints is noisy. The test sets were used only for these final, one-time comparisons.

---

## Phase 7 (optional): Write the GRPO loop yourself

This is the best phase for learning. Once the TRL run works, write a loop of about 200 lines:

- [ ] Sample G completions per prompt with HF `generate`.
- [ ] Score them with `unit_test_reward`.
- [ ] Normalize rewards within each group to get advantages.
- [ ] Compute a token-level clipped policy loss and take an optimizer step.
- [ ] Compare its reward curve with TRL's on the 0.5B model.

---

## Suggested layout

```
src/prompts.py    # build prompt + extract code (shared by train & eval)
src/sandbox.py    # run code + tests with limits and nonce
src/rewards.py    # TRL reward functions
src/data.py       # load MBPP, splits, dedupe vs MBPP+, difficulty filter
eval/generate.py  # vLLM sampling -> samples.jsonl
eval/score.py     # pass@k, error breakdown, paired stats
eval/evalplus_check.py  # same samples through the EvalPlus CLI
eval/checkpoints.py     # base + every checkpoint in one vLLM engine (validation curve, final test eval)
eval/audit.py           # reward-hacking audit
scripts/make_train_set.py   # Phase 4 difficulty filter
scripts/phase56.sh          # every training run in priority order
scripts/final_eval.py       # Phase 6: best checkpoint vs base on the test sets, seeds table
scripts/summarize_runs.py   # training metrics + validation curve, one row per run
train_grpo.py
tests/            # pytest for sandbox and rewards
```
