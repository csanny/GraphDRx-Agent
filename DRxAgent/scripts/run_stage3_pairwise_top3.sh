#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

INPUT=${INPUT:-data/candidate_pairs.grounded.normalized.jsonl}
RUN_ROOT=${RUN_ROOT:-output/configuration_selection}
EVAL_ROOT=${EVAL_ROOT:-analysis/configuration_selection}
JUDGE_PROVIDER=${JUDGE_PROVIDER:-openai}
JUDGE_MODEL=${JUDGE_MODEL:-gpt-5.6-terra}
JUDGE_REASONING_EFFORT=${JUDGE_REASONING_EFFORT:-medium}
PAIRWISE_MAX_OUTPUT_TOKENS=${PAIRWISE_MAX_OUTPUT_TOKENS:-6000}
CONFIRM_RUN_GPT_JUDGE=${CONFIRM_RUN_GPT_JUDGE:-0}

if [[ "$CONFIRM_RUN_GPT_JUDGE" != "1" ]]; then
  echo "Refusing to start paid/external pairwise evaluation."
  echo "Run with CONFIRM_RUN_GPT_JUDGE=1 after inspecting the pointwise results."
  exit 2
fi

python scripts/05_pairwise_top3.py \
  --top3 "$EVAL_ROOT/summary/top3_configurations.txt" \
  --input "$INPUT" \
  --run-root "$RUN_ROOT" \
  --output "$EVAL_ROOT/top3_pairwise_swapped.jsonl" \
  --provider "$JUDGE_PROVIDER" \
  --judge-model "$JUDGE_MODEL" \
  --reasoning-effort "$JUDGE_REASONING_EFFORT" \
  --max-output-tokens "$PAIRWISE_MAX_OUTPUT_TOKENS"

python scripts/07_analyze_pairwise.py \
  --input "$EVAL_ROOT/top3_pairwise_swapped.jsonl" \
  --output-dir "$EVAL_ROOT/pairwise_summary"

echo "Pairwise confirmation complete: $EVAL_ROOT/top3_pairwise_swapped.jsonl"
echo "Pairwise summaries: $EVAL_ROOT/pairwise_summary"
