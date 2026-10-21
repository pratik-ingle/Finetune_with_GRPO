# GRPO on MBPP

Fine-tuning Qwen2.5-Coder-0.5B/1.5B-Instruct with GRPO, using unit-test pass rates as the reward, on one 12 GB GPU. The evaluation harness reproduces Qwen's published scores to within about 2 points. Every claim is checked with paired statistics over several seeds and audited for reward hacking.

The project was built phase by phase from [learn.md](learn.md). That file is the learning path; it records every design decision, gotcha and intermediate result, so read it for the details. This README is the summary.

## Results

GRPO with LoRA on the MBPP train split, with the checkpoint chosen on MBPP validation. Each cell is the base model's greedy pass@1, followed by the mean change over seeds [worst seed, best seed]:

| Model | Seeds | MBPP test (500) | MBPP+ dedup (270) | HumanEval+ (164) |
|---|---|---|---|---|
| Qwen2.5-Coder-1.5B-Instruct | 3 | 51.0, **+3.2** [+1.8, +4.2] | 61.9, **+3.2** [+0.4, +4.8] | 64.0, **+5.5** [+3.7, +6.7] |
| Qwen2.5-Coder-0.5B-Instruct | 3 | 33.4, **+5.0** [+4.2, +6.0] | 46.3, **+2.3** [+0.4, +4.8] | 54.9, −0.6 [−1.8, +1.2] |

- **MBPP+ dedup** is MBPP+ minus the 108 problems that also appear in MBPP's train split, scored with EvalPlus's extra tests.
- **1.5B:** two of three seeds beat the base model significantly (McNemar p < 0.05 on most test sets). The third seed's gain is within noise.
- **Reward hacking:** none found across all 21 runs.
  - The audit found no early exits, no errors swallowed by catch-all `except` blocks, no always-equal objects and no introspection of the test runner.
  - The few "hardcoded" flags turned out, on inspection, to be domain constants such as month names.
  - The MBPP–MBPP+ gap, which grows when a model hardcodes visible tests, stayed flat.

Ablations on 0.5B, 2 seeds each, as change in pass@1 from the starting model:

| Config | MBPP test | MBPP+ dedup | HumanEval+ |
|---|---|---|---|
| Main: G=8, partial-credit reward, 1 visible test, beta=0 | +5.4 | +3.3 | −0.3 |
| Group size G=4 | +2.5 | +1.5 | −2.4 |
| Group size G=16 | **+6.8** | **+7.0** | −0.9 |
| Binary reward | +4.4 | +3.5 | −2.1 |
| 3 visible tests | +4.9 | +4.4 | +1.2 |
| KL penalty beta=0.04 | +3.1 | +2.6 | −0.3 |
| Base model instead of Instruct | +4.1 | +6.5 | **−9.1** |

- **Group size matters most.** With G=4, more groups get identical rewards for every sample and so contribute no gradient.
- **The base model loses ground on HumanEval.** Starting from the base model gives the largest MBPP+ gain. But the trained model also stops writing `from typing import List`, which MBPP solutions never need, so it fails HumanEval's type-hinted function stubs.
- **3 visible tests didn't lead to hardcoding.** At this training budget, showing all of a problem's tests in the prompt caused no hardcoding.

The baseline numbers, compared with Qwen's, are in [learn.md, Phase 3](learn.md#phase-3-eval-harness-and-baseline), and the full ablation analysis is in [Phase 6](learn.md#phase-6-evaluate-and-analyze).

## How it works

1. **Sandbox** ([src/sandbox.py](src/sandbox.py)): runs each candidate solution in a fresh subprocess.
   - **Limits:** caps on memory, CPU time and file size; no new processes; none of your environment variables.
   - **Partial credit:** each assert is scored independently.
   - **Forgery-resistant result:** the code under test runs in its own namespace, and the result is reported through a line it can't fake.
   - **Tests:** the pytest suite covers `sys.exit`, fake success output, rebinding builtins, memory hogs and more.
2. **Prompts and extraction** ([src/prompts.py](src/prompts.py)): one module shared by training and eval.
   - **Prompt:** EvalPlus's instruct prompt, rendered with the chat template into a single string that ends inside an opened ```` ```python ```` block.
   - **Sanitizer:** like EvalPlus's, it drops the model's own example asserts and prints before testing.
3. **Eval harness** ([eval/](eval/)):
   - generates samples with vLLM, in batch-invariant mode so greedy runs are byte-identical;
   - scores pass@k with a breakdown of failure types;
   - compares runs with a paired bootstrap and McNemar's test;
   - cross-checks against the EvalPlus CLI (98.7–99.4% agreement).
4. **Training data** ([scripts/make_train_set.py](scripts/make_train_set.py)): samples 8 completions per MBPP train problem and keeps only the problems where some samples pass and some fail. Groups where every sample gets the same reward give zero gradient.
5. **GRPO** ([train_grpo.py](train_grpo.py)):
   - **Setup:** TRL `GRPOTrainer` with LoRA (r=32, all linear layers) on a frozen bf16 base, with vLLM colocated for generation.
   - **Settings:** G=8, 4 prompts per step, 3 epochs, learning rate 2e-5, TRL's default DAPO loss and no KL penalty.
   - **Reward:** fraction of tests passed, plus a small reward (weight 0.1) for a well-formed code block.
6. **Analysis:**
   - [eval/checkpoints.py](eval/checkpoints.py) evaluates the base model and every checkpoint in one vLLM engine.
   - [scripts/final_eval.py](scripts/final_eval.py) compares the checkpoint chosen on validation with the base model, once, on the test sets.
   - [eval/audit.py](eval/audit.py) scans completions for reward hacking.

## Setup

Tested on an RTX 4070 SUPER (12 GB), NVIDIA driver 535, 30 GB RAM, Ubuntu 22.04.

```bash
conda env create -f environment.yml
conda activate grpo
python scripts/smoke_vllm.py    # vLLM generates one completion
pytest                          # about 1 minute; downloads the datasets on first run
```

The environment pins torch 2.13, vLLM 0.31 and TRL 1.16, using their CUDA 12.9 builds, because the default PyPI builds need driver 580 or newer. Use `conda activate`: it also sets `VLLM_USE_FLASHINFER_SAMPLER=0`, which this setup needs.

## Reproducing the results

Run everything from the repo root with the `grpo` environment active. Times are for the GPU above.

```bash
# Phase 3: baseline for both models, plus the EvalPlus cross-check (about 40 min)
bash scripts/baseline.sh

# Phase 4: difficulty-filtered train sets (the results are already in data/)
python -m scripts.make_train_set --model Qwen/Qwen2.5-Coder-1.5B-Instruct --out data/train-coder-1.5b.jsonl

# Phases 5-6: every training run, each followed by its validation curve (about 6 h for all 21 runs)
bash scripts/phase56.sh                # or a subset, e.g. bash scripts/phase56.sh 'coder-1.5b-*'
python -m scripts.final_eval runs/grpo/*-s[0-9]     # best checkpoint vs base on the test sets
python -m eval.audit --train runs/grpo/coder-1.5b-s0
```

A single training run and its evaluation:

```bash
python train_grpo.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --train-data data/train-coder-1.5b.jsonl \
    --out runs/grpo/coder-1.5b-s0 --vllm-memory 0.45 --vllm-sleep   # about 33 min, 9.4 GB peak
python -m eval.checkpoints runs/grpo/coder-1.5b-s0                  # validation pass@1 per checkpoint
python -m eval.score --compare RUN_A RUN_B                          # paired comparison of two scored runs
```

**What's in the repo:**
- **Included:** the Phase 3 baseline samples and scores, in `runs/eval/`, and the train sets, in `data/`.
- **Not included:** the trained LoRA adapters (about 12 GB) and the W&B logs. W&B runs in offline mode, and `wandb sync` uploads a run.

## Layout

```
src/
  sandbox.py           run code + tests in a limited subprocess
  prompts.py           prompt building, code extraction, sanitizer (shared by train and eval)
  rewards.py           TRL reward functions
  data.py              MBPP, MBPP+, HumanEval+ in one problem format
eval/
  generate.py          vLLM sampling -> <dataset>.samples.jsonl
  score.py             pass@k, failure breakdown, paired comparison
  evalplus_check.py    the same samples through the EvalPlus CLI
  checkpoints.py       base + checkpoints in one vLLM engine (LoRA)
  audit.py             reward-hacking audit
scripts/
  smoke_vllm.py        Phase 1 smoke test
  baseline.sh          Phase 3 baseline for both models
  make_train_set.py    Phase 4 difficulty filter
  phase56.sh           every training run, in priority order
  final_eval.py        Phase 6 test-set comparison and seeds table
  summarize_runs.py    training metrics and validation curve, one row per run
train_grpo.py          Phase 5 GRPO training
tests/                 sandbox, rewards, prompts, scoring, eval data, audit
learn.md               the learning path, with every decision and result
```

## Limitations

- **The sandbox isn't a security boundary.** It's a subprocess with resource limits: generated code can still read and write your files. For stronger isolation, run the executor in Docker with `--network none`.
- **The ablations are small.** They ran on 0.5B only, with 2 seeds each. Differences of 1–2 points are within the seed spread.
- **Checkpoint choice is noisy.** MBPP validation has only 90 problems. The test sets were used only once per run, for the final comparison.
- **Phase 7 isn't done.** Writing the GRPO loop by hand is optional and not started.

## References

**Data and models:**
- [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp)
- [MBPP+](https://huggingface.co/datasets/evalplus/mbppplus) and [HumanEval+](https://huggingface.co/datasets/evalplus/humanevalplus) from [EvalPlus](https://github.com/evalplus/evalplus)
- [HumanEval](https://huggingface.co/datasets/openai/openai_humaneval)
- Qwen2.5-Coder [0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct), [1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) and [0.5B base](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B)

**Methods:**
- GRPO: [DeepSeekMath](https://arxiv.org/abs/2402.03300)
- [Dr. GRPO](https://arxiv.org/abs/2503.20783)
- [DAPO](https://arxiv.org/abs/2503.14476)
- [TRL](https://github.com/huggingface/trl)
- [vLLM](https://github.com/vllm-project/vllm)
