#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

NEO4J_PASSWORD="${NEO4J_PASSWORD:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output}"
TEST_CASES_CSV="${TEST_CASES_CSV:-data/graphdrx_disease_benchmark.csv}"
DRUG_RAG_CSV="${DRUG_RAG_CSV:-data/rag_corpus/Drug_data_All_RAG.csv}"
DISEASE_FEATURES_CSV="${DISEASE_FEATURES_CSV:-data/rag_corpus/disease_features.csv}"
CONFIG="${CONFIG:-code/configs/graphdrx_method_config.json}"
RUNS="${RUNS:-main}"
AREA_FILTER="${AREA_FILTER:-}"
DISEASES="${DISEASES:-}"
DISEASE_LIST_FILE="${DISEASE_LIST_FILE:-}"
SAMPLE_PER_AREA="${SAMPLE_PER_AREA:-0}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
MAX_CASES="${MAX_CASES:-0}"
DRY_RUN="${DRY_RUN:-0}"
RESUME="${RESUME:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
LOG_DIR="${LOG_DIR:-logs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"

has_run() {
  local needle="$1"
  IFS=',' read -r -a parts <<< "$RUNS"
  for item in "${parts[@]}"; do
    item="${item// /}"
    [[ "$item" == "$needle" || "$item" == "all" ]] && return 0
  done
  return 1
}

common_args=(
  --config "$CONFIG"
  --neo4j-password "$NEO4J_PASSWORD"
  --test-cases-csv "$TEST_CASES_CSV"
  --drug-rag-csv "$DRUG_RAG_CSV"
  --disease-features-csv "$DISEASE_FEATURES_CSV"
  --output-root "$OUTPUT_ROOT"
  --disable-mechanism-lookup
  --mask-eval-diseases-from-prior
  --enable-sparse-area-rescue
  --sample-seed "$SAMPLE_SEED"
)

[[ -n "$AREA_FILTER" ]] && common_args+=(--area-filter "$AREA_FILTER")
[[ "$SAMPLE_PER_AREA" != "0" ]] && common_args+=(--sample-per-area "$SAMPLE_PER_AREA")
[[ "$MAX_CASES" != "0" ]] && common_args+=(--max-cases "$MAX_CASES")
[[ -n "$DISEASE_LIST_FILE" ]] && common_args+=(--disease-list-file "$DISEASE_LIST_FILE")
[[ "$DRY_RUN" == "1" ]] && common_args+=(--dry-run)
[[ "$RESUME" == "1" ]] && common_args+=(--resume)

if [[ -n "$DISEASES" ]]; then
  IFS='|' read -r -a disease_parts <<< "$DISEASES"
  for disease in "${disease_parts[@]}"; do
    [[ -n "$disease" ]] && common_args+=(--disease "$disease")
  done
fi

extra_args=()
if [[ -n "$EXTRA_ARGS" ]]; then
  read -r -a extra_args <<< "$EXTRA_ARGS"
fi

run_condition() {
  local name="$1"
  shift
  local log_file="$LOG_DIR/${RUN_ID}_${name}.log"

  echo "[GraphDRx] run=$name output=$OUTPUT_ROOT log=$log_file"

  python code/graphdrx_retrieval_pipeline.py \
    "${common_args[@]}" "${extra_args[@]}" \
    --run-name "$name" "$@" 2>&1 | tee "$log_file"
}

# Final main method with stable single-indirect-only reserve ordering.
has_run main && run_condition main


# Clean leave-one-branch-out ablations for the final T/B/C/D method.
## T = target-disease biological-context retrieval
## B = biological-context retrieval from embedding-similar diseases
## C = semantic therapeutic-prior support with target-disease KG verification
## D = direct target-disease gene-action retrieval
has_run ablation_no_T && run_condition ablation_no_T --disable-target-context-graph

has_run ablation_no_B && run_condition ablation_no_B --disable-vector-anchor-graph

has_run ablation_no_C && run_condition ablation_no_C --disable-semantic-prior-branch

has_run ablation_no_D && run_condition ablation_no_D --disable-common-direct-graph

exit 0
