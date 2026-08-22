#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

TOP5_CSV=${TOP5_CSV:-data/selected_top5.csv}
RAG_DIR=${RAG_DIR:-data/rag_corpus}
OUTPUT=${OUTPUT:-data/candidate_pairs.grounded.jsonl}
COVERAGE_CSV=${COVERAGE_CSV:-data/evidence_coverage.csv}
COVERAGE_JSON=${COVERAGE_JSON:-data/evidence_coverage_summary.json}
PREVIEW_DIR=${PREVIEW_DIR:-data/evidence_preview}

python scripts/00_build_evidence_packets.py \
  --top5-csv "$TOP5_CSV" \
  --rag-dir "$RAG_DIR" \
  --output "$OUTPUT" \
  --coverage-csv "$COVERAGE_CSV" \
  --coverage-json "$COVERAGE_JSON" \
  --preview-dir "$PREVIEW_DIR" \
  --expected-diseases 15 \
  --top-k 5

python scripts/00_validate_input.py \
  --input "$OUTPUT" \
  --normalized-output data/candidate_pairs.grounded.normalized.jsonl \
  --expected-diseases 15 \
  --top-k 5
