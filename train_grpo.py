"""Phase 5: GRPO with TRL on a Phase 4 train set, LoRA on a frozen bf16 base, vLLM colocated for generation.

    python train_grpo.py --model Qwen/Qwen2.5-Coder-0.5B-Instruct --train-data data/train-coder-0.5b.jsonl \
        --out runs/grpo/coder-0.5b

Writes LoRA checkpoints to <out>/checkpoint-*/ (score them with eval.checkpoints), TRL's log history to
<out>/trainer_state.json, and every training completion with its reward to <out>/completions.jsonl for
the Phase 6 reward-hacking audit. W&B runs in offline mode; `wandb sync` uploads a run later.
The flags with defaults other than learn.md's are the Phase 6 ablations.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "offline")

from datasets import Dataset  # noqa: E402
from peft import LoraConfig  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

from src.prompts import build_prompt, mbpp_problem  # noqa: E402
from src.rewards import binary_unit_test_reward, format_reward, unit_test_reward  # noqa: E402

MICRO_BATCH = 8  # completions per forward/backward pass


def logged(reward_fn, path: Path):
    """Wrap a reward function so every completion it scores is appended to `path`."""
    def wrapper(completions, **kwargs):
        rewards = reward_fn(completions, **kwargs)
        step = kwargs["trainer_state"].global_step
        with path.open("a") as f:
            for task_id, completion, reward in zip(kwargs["task_id"], completions, rewards, strict=True):
                f.write(json.dumps({"step": step, "task_id": task_id, "completion": completion, "reward": reward}) + "\n")
        return rewards
    wrapper.__name__ = reward_fn.__name__  # TRL logs each reward under its function's name
    return wrapper


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-data", required=True, type=Path, help="jsonl from scripts.make_train_set")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--max-steps", type=int, default=-1, help="stop early, for debugging")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--num-generations", type=int, default=8, help="group size G")
    parser.add_argument("--prompts-per-step", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.0, help="KL penalty")
    parser.add_argument("--reward", choices=["partial", "binary"], default="partial")
    parser.add_argument("--visible-tests", type=int, default=1, help="asserts shown in the prompt")
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--vllm-memory", type=float, default=0.3, help="vllm_gpu_memory_utilization")
    parser.add_argument("--vllm-sleep", action="store_true",
                        help="offload vLLM's weights and cache during training steps; 1.5B needs it on 12 GB")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows = [json.loads(line) for line in args.train_data.open()]
    train = Dataset.from_list([{
        "prompt": build_prompt(mbpp_problem(r["text"], r["tests"][:args.visible_tests]), tokenizer),
        "tests": r["tests"],  # rewards always use every test, however many the prompt shows
        "test_setup_code": r["test_setup_code"],
        "task_id": r["task_id"],
    } for r in rows])

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "args.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n")
    completions_log = args.out / "completions.jsonl"
    completions_log.unlink(missing_ok=True)
    test_reward = unit_test_reward if args.reward == "partial" else binary_unit_test_reward

    completions_per_step = args.prompts_per_step * args.num_generations
    if completions_per_step % MICRO_BATCH:
        parser.error(f"--prompts-per-step x --num-generations must be a multiple of {MICRO_BATCH}")
    config = GRPOConfig(
        output_dir=str(args.out), run_name=args.out.name, seed=args.seed,
        learning_rate=args.lr, num_train_epochs=args.epochs, max_steps=args.max_steps,
        num_generations=args.num_generations, per_device_train_batch_size=MICRO_BATCH,
        gradient_accumulation_steps=completions_per_step // MICRO_BATCH,
        max_completion_length=args.max_completion_length, temperature=1.0, beta=args.beta,
        reward_weights=[1.0, 0.1], bf16=True, gradient_checkpointing=True,
        model_init_kwargs={"dtype": "bfloat16"},  # TRL's default is float32; the base is frozen under LoRA
        use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=args.vllm_memory,
        vllm_enable_sleep_mode=args.vllm_sleep, vllm_max_model_length=2048,
        logging_steps=1, save_steps=args.save_steps, save_only_model=True, report_to="wandb",
    )
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[logged(test_reward, completions_log), format_reward],
        args=config,
        train_dataset=train,
        processing_class=tokenizer,
        peft_config=LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, target_modules="all-linear",
                               task_type="CAUSAL_LM"),
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(str(args.out / "final"))


if __name__ == "__main__":
    main()
