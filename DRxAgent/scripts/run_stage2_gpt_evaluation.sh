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
JUDGE_MAX_OUTPUT_TOKENS=${JUDGE_MAX_OUTPUT_TOKENS:-12000}
JUDGE_MAX_CHARS_PER_SECTION=${JUDGE_MAX_CHARS_PER_SECTION:-6000}
CONFIRM_RUN_GPT_JUDGE=${CONFIRM_RUN_GPT_JUDGE:-0}

mkdir -p "$EVAL_ROOT"

python scripts/03_run_judge.py \
  --input "$INPUT" \
  --run-root "$RUN_ROOT" \
  --output "$EVAL_ROOT/judge_scores.jsonl" \
  --provider "$JUDGE_PROVIDER" \
  --judge-model "$JUDGE_MODEL" \
  --qc "$EVAL_ROOT/automatic_qc.csv" \
  --mode disease \
  --reasoning-effort "$JUDGE_REASONING_EFFORT" \
  --max-output-tokens "$JUDGE_MAX_OUTPUT_TOKENS" \
  --max-chars-per-section "$JUDGE_MAX_CHARS_PER_SECTION" \
  --dry-run

if [[ "$CONFIRM_RUN_GPT_JUDGE" != "1" ]]; then
  echo "Dry-run manifest created. Refusing to start paid/external GPT evaluation."
  echo "Inspect: $EVAL_ROOT/judge_scores.run_manifest.json"
  echo "Then run with CONFIRM_RUN_GPT_JUDGE=1."
  exit 2
fi

python scripts/03_run_judge.py \
  --input "$INPUT" \
  --run-root "$RUN_ROOT" \
  --output "$EVAL_ROOT/judge_scores.jsonl" \
  --provider "$JUDGE_PROVIDER" \
  --judge-model "$JUDGE_MODEL" \
  --qc "$EVAL_ROOT/automatic_qc.csv" \
  --mode disease \
  --reasoning-effort "$JUDGE_REASONING_EFFORT" \
  --max-output-tokens "$JUDGE_MAX_OUTPUT_TOKENS" \
  --max-chars-per-section "$JUDGE_MAX_CHARS_PER_SECTION"

python scripts/04_analyze_configurations.py \
  --judge-output "$EVAL_ROOT/judge_scores.jsonl" \
  --qc "$EVAL_ROOT/automatic_qc.csv" \
  --input "$INPUT" \
  --output-dir "$EVAL_ROOT/summary" \
  --bootstrap 20000 \
  --seed 42

echo
echo "Stage 2 complete. Main outputs:"
echo "  $EVAL_ROOT/summary/configuration_summary.csv"
echo "  $EVAL_ROOT/summary/configuration_primary_score_bootstrap.csv"
echo "  $EVAL_ROOT/summary/top_vs_runner_up_paired_bootstrap.csv"
echo "  $EVAL_ROOT/summary/role_assignment_primary_score_marginals.csv"
echo "  $EVAL_ROOT/summary/role_diagnostic_marginals.csv"
echo "  $EVAL_ROOT/summary/critical_flag_summary.csv"
echo "  $EVAL_ROOT/summary/score_distribution_by_direct_relation.csv"
echo "  $EVAL_ROOT/summary/configuration_scores_by_direct_relation.csv"
echo "  $EVAL_ROOT/summary/configuration_rank_stability_no_direct.csv"
echo "  $EVAL_ROOT/summary/direct_relation_score_association_by_configuration.csv"
echo "  $EVAL_ROOT/summary/qualitative_notes.csv"
echo "  $EVAL_ROOT/summary/top3_configurations.txt"
echo
echo "Then run order-swapped pairwise confirmation:"
echo "  CONFIRM_RUN_GPT_JUDGE=1 JUDGE_MODEL='$JUDGE_MODEL' bash scripts/run_stage3_pairwise_top3.sh"
