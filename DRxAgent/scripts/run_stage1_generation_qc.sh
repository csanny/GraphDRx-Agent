#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

INPUT=${INPUT:-data/candidate_pairs.grounded.normalized.jsonl}
RUN_ROOT=${RUN_ROOT:-output/configuration_selection}
EVAL_ROOT=${EVAL_ROOT:-analysis/configuration_selection}
CONFIGS=${CONFIGS:-}
WITH_DISEASE_SUMMARY=${WITH_DISEASE_SUMMARY:-0}

if [[ ! -f "$INPUT" ]]; then
  echo "Grounded evidence input not found: $INPUT" >&2
  echo "Run first: bash scripts/build_grounded_input.sh" >&2
  exit 2
fi

mkdir -p "$EVAL_ROOT"

echo "[1/3] Validating grounded 15-disease x top-5 evidence packets."
python scripts/00_validate_input.py \
  --input "$INPUT" \
  --normalized-output "$INPUT" \
  --expected-diseases 15 \
  --top-k 5

SUMMARY_ARGS=()
if [[ "$WITH_DISEASE_SUMMARY" == "1" ]]; then
  SUMMARY_ARGS+=(--with-disease-summary)
fi

echo "[2/3] Generating Sponsor-Expert-Chair reports. GraphDRx is NOT run."
if [[ -n "$CONFIGS" ]]; then
  read -r -a CONFIG_ARRAY <<< "$CONFIGS"
  python scripts/01_run_reports.py \
    --input "$INPUT" \
    --output-root "$RUN_ROOT" \
    --configs "${CONFIG_ARRAY[@]}" \
    "${SUMMARY_ARGS[@]}"
else
  python scripts/01_run_reports.py \
    --input "$INPUT" \
    --output-root "$RUN_ROOT" \
    "${SUMMARY_ARGS[@]}"
fi

echo "[3/3] Running deterministic quantitative QC."
python scripts/02_auto_qc.py \
  --input "$INPUT" \
  --run-root "$RUN_ROOT" \
  --output "$EVAL_ROOT/automatic_qc.csv" \
  --summary-output "$EVAL_ROOT/automatic_qc_summary.csv"

echo
echo "Stage 1 complete. Inspect before GPT evaluation:"
echo "  $EVAL_ROOT/automatic_qc_summary.csv"
echo "  $EVAL_ROOT/automatic_qc.csv"
