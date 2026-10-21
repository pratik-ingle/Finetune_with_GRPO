#!/usr/bin/env bash
# Phases 5-6: every GRPO run in priority order, each followed by its validation curve (eval.checkpoints).
# Finished runs (with a validation.json) are skipped, so the queue can be rerun or resumed. An optional
# glob limits it to matching run names: scripts/phase56.sh 'abl-*'. Run from the repo root in the grpo env.
set -uo pipefail
pattern=${1:-*}
# Everything is cached after Phase 3/4; without this, a DNS hiccup kills a run when vLLM lists Hub files.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
mkdir -p runs/grpo

run() {  # run <name> <train_grpo.py args...>
    local name=$1 out=runs/grpo/$1 log=runs/grpo/$1.log
    shift
    [[ $name == $pattern ]] || return 0
    if [[ -f $out/validation.json ]]; then echo "skip $name (done)"; return 0; fi
    rm -rf "$out"
    echo "$(date +%T) start $name"
    if python train_grpo.py --out "$out" "$@" > "$log" 2>&1 && python -m eval.checkpoints "$out" >> "$log" 2>&1; then
        echo "$(date +%T) done  $name: validation pass@1 by step: $(grep '^step' "$log" | awk '{printf "%s=%s ", $2, $4}')"
    else
        echo "$(date +%T) FAILED $name, see $log"
    fi
}

M05="--model Qwen/Qwen2.5-Coder-0.5B-Instruct --train-data data/train-coder-0.5b.jsonl"
M15="--model Qwen/Qwen2.5-Coder-1.5B-Instruct --train-data data/train-coder-1.5b.jsonl --vllm-memory 0.45 --vllm-sleep"
BASE="--model Qwen/Qwen2.5-Coder-0.5B --train-data data/train-coder-0.5b-base.jsonl"
if [[ ! -f data/train-coder-0.5b-base.jsonl && abl-base-s0 == $pattern ]]; then
    python -m scripts.make_train_set --model Qwen/Qwen2.5-Coder-0.5B --out data/train-coder-0.5b-base.jsonl \
        > runs/grpo/make-train-base.log 2>&1
fi

run coder-1.5b-s0 $M15 --seed 0
for s in 0 1 2; do run coder-0.5b-s$s $M05 --seed $s; done
for s in 0 1; do  # ablations, each against coder-0.5b-s$s
    run abl-binary-s$s $M05 --seed $s --reward binary
    run abl-vis3-s$s $M05 --seed $s --visible-tests 3
    run abl-g4-s$s $M05 --seed $s --num-generations 4
    run abl-g16-s$s $M05 --seed $s --num-generations 16
    run abl-beta0.04-s$s $M05 --seed $s --beta 0.04
    run abl-base-s$s $BASE --seed $s
done
for s in 1 2; do run coder-1.5b-s$s $M15 --seed $s; done
