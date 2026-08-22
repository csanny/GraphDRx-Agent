#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# One matched output tree for the final main and all final ablations.
export OUTPUT_ROOT="${OUTPUT_ROOT:-output_graphdrx_final_study}"
export LOG_DIR="${LOG_DIR:-logs_graphdrx_final_study}"
export ANALYSIS_DIR="${ANALYSIS_DIR:-analysis_graphdrx_final_study}"
export RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

# Final GraphDRx Main and matched leave-one-component-out ablations.
RUNS="main,ablation_no_T,ablation_no_B,ablation_no_C,ablation_no_D" \
bash ./run_exp1_full.sh

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  python analyze_exp1_graphdrx_results.py \
    --output-root "$OUTPUT_ROOT" \
    --analysis-dir "$ANALYSIS_DIR" \
    --main-run main
fi

echo "[DONE] outputs : $OUTPUT_ROOT"
echo "[DONE] logs    : $LOG_DIR"
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  echo "[DONE] analysis: $ANALYSIS_DIR"
fi
