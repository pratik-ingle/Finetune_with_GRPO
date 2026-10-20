#!/usr/bin/env bash
# Phase 3 baseline for both models: greedy twice (the repeat must match byte for byte), then
# 10 samples per problem at temperature 0.8 for pass@k. Run from the repo root in the grpo env.
set -euo pipefail

for size in 0.5b 1.5b; do
    model="Qwen/Qwen2.5-Coder-${size^^}-Instruct"
    out="runs/eval/coder-${size}"
    python -m eval.generate --model "$model" --out "$out-greedy"
    python -m eval.generate --model "$model" --out "$out-greedy-repeat"
    python -m eval.generate --model "$model" --n 10 --temperature 0.8 --out "$out-t0.8-n10"
    python -m eval.score "$out-greedy" "$out-greedy-repeat" "$out-t0.8-n10"
    for f in "$out-greedy"/*.samples.jsonl; do
        if cmp -s "$f" "$out-greedy-repeat/$(basename "$f")"; then
            echo "repeat identical: $f"
        else
            echo "REPEAT DIFFERS: $f"
        fi
    done
    python -m eval.evalplus_check "$out-greedy"
done
