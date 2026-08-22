#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath
_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
from pathlib import Path

import pandas as pd

from graphdrx_agent_eval.common import atomic_write_json
from graphdrx_agent_eval.evidence_builder import (
    CandidateEvidenceBuilder,
    EvidenceBuildError,
    Neo4jRepository,
    RagRepository,
)


def validate_top5(df: pd.DataFrame, expected_diseases: int, top_k: int) -> pd.DataFrame:
    required = {
        "disease", "source_disease", "therapeutic_area", "rank", "drug",
        "retrieval_score", "best_path_pattern", "best_path_nodes", "best_path_rels",
    }
    missing = required - set(df.columns)
    if missing:
        raise EvidenceBuildError(f"Fixed top-5 CSV missing required columns: {sorted(missing)}")
    df = df.copy()
    df["rank"] = pd.to_numeric(df["rank"], errors="raise").astype(int)
    df["retrieval_score"] = pd.to_numeric(df["retrieval_score"], errors="raise").astype(float)
    df["disease"] = df["disease"].astype(str).str.strip()
    df["drug"] = df["drug"].astype(str).str.strip()
    if df["disease"].nunique() != expected_diseases:
        raise EvidenceBuildError(f"Expected {expected_diseases} diseases, found {df['disease'].nunique()}")
    if df.duplicated(subset=["disease", "rank", "drug"], keep=False).any():
        raise EvidenceBuildError("Duplicate disease-rank-drug rows found in fixed top-5 CSV")
    for disease, sub in df.groupby("disease", sort=False):
        ranks = sorted(sub["rank"].tolist())
        expected = list(range(1, top_k + 1))
        if ranks != expected:
            raise EvidenceBuildError(f"{disease}: expected ranks {expected}, found {ranks}")
    sort_cols = ["selection_order", "rank"] if "selection_order" in df.columns else ["disease", "rank"]
    return df.sort_values(sort_cols)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build grounded Agent packets from the fixed top-5 CSV, four RAG CSVs, and live Neo4j. GraphDRx is not rerun."
    )
    ap.add_argument("--top5-csv", default="data/selected_top5.csv")
    ap.add_argument("--rag-dir", default="data/rag_corpus")
    ap.add_argument("--output", default="data/candidate_pairs.grounded.jsonl")
    ap.add_argument("--coverage-csv", default="data/evidence_coverage.csv")
    ap.add_argument("--coverage-json", default="data/evidence_coverage_summary.json")
    ap.add_argument("--preview-dir", default="data/evidence_preview")
    ap.add_argument("--expected-diseases", type=int, default=15)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--neo4j-uri", default=None)
    ap.add_argument("--neo4j-user", default=None)
    ap.add_argument("--neo4j-password", default=None)
    ap.add_argument("--neo4j-database", default=None)
    ap.add_argument("--allow-partial", action="store_true", help="write packets even when core grounding evidence is incomplete")
    args = ap.parse_args()

    try:
        df = validate_top5(pd.read_csv(args.top5_csv), args.expected_diseases, args.top_k)
        rag = RagRepository(args.rag_dir)
        neo = Neo4jRepository(args.neo4j_uri, args.neo4j_user, args.neo4j_password, args.neo4j_database)
        neo.verify()
        builder = CandidateEvidenceBuilder(rag, neo)
        packets = []
        failures = []
        try:
            for row_no, (_, row) in enumerate(df.iterrows(), 2):
                try:
                    packet = builder.build(row, row_no)
                    packets.append(packet)
                    cov = packet["evidence_coverage"]
                    print(
                        f"[{len(packets):02d}/75] {packet['disease']} | r{packet['rank']} {packet['drug']} | "
                        f"core_ready={cov['core_ready']} drug_rag={cov['drug_rag_matched']} "
                        f"disease_rag={cov['disease_rag_matched']}"
                    )
                except Exception as exc:
                    failures.append({
                        "disease": str(row.get("disease", "")),
                        "rank": int(row.get("rank", 0)),
                        "drug": str(row.get("drug", "")),
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    print("FAILED:", failures[-1])
        finally:
            neo.close()

        expected_n = args.expected_diseases * args.top_k
        if failures:
            atomic_write_json(Path(args.coverage_json).with_name("evidence_build_failures.json"), failures)
            raise EvidenceBuildError(f"Evidence construction failed for {len(failures)} pairs; see evidence_build_failures.json")
        if len(packets) != expected_n:
            raise EvidenceBuildError(f"Expected {expected_n} packets, built {len(packets)}")

        coverage_rows = []
        for packet in packets:
            coverage_rows.append({
                "case_id": packet["case_id"],
                "disease": packet["disease"],
                "rank": packet["rank"],
                "drug": packet["drug"],
                **packet["evidence_coverage"],
                "disease_rag_match_method": packet["entity_resolution"]["disease"]["rag_match_method"],
                "drug_rag_match_method": packet["entity_resolution"]["drug"]["rag_match_method"],
            })
        coverage_df = pd.DataFrame(coverage_rows)
        incomplete = coverage_df.loc[~coverage_df["core_ready"]]
        if not incomplete.empty and not args.allow_partial:
            Path(args.coverage_csv).parent.mkdir(parents=True, exist_ok=True)
            coverage_df.to_csv(args.coverage_csv, index=False)
            raise EvidenceBuildError(
                f"Core grounding evidence incomplete for {len(incomplete)} pairs. "
                f"Inspect {args.coverage_csv}; use --allow-partial only after manual review."
            )

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            for packet in packets:
                f.write(json.dumps(packet, ensure_ascii=False) + "\n")

        Path(args.coverage_csv).parent.mkdir(parents=True, exist_ok=True)
        coverage_df.to_csv(args.coverage_csv, index=False)
        bool_cols = [c for c in coverage_df.columns if coverage_df[c].dtype == bool]
        summary = {
            "n_packets": len(packets),
            "n_diseases": int(coverage_df["disease"].nunique()),
            "core_ready_n": int(coverage_df["core_ready"].sum()),
            "coverage_rates": {col: float(coverage_df[col].mean()) for col in bool_cols},
            "numeric_totals": {
                col: int(coverage_df[col].sum())
                for col in (
                    "drug_gene_mechanism_rows", "drug_disease_development_rows", "counter_evidence_rows",
                    "path_segments_expected", "path_segment_evidence_rows",
                )
            },
            "graphdrx_rerun": False,
        }
        atomic_write_json(args.coverage_json, summary)

        preview_dir = Path(args.preview_dir)
        preview_dir.mkdir(parents=True, exist_ok=True)
        for packet in packets:
            if packet["disease"] == "Cushing syndrome" and int(packet["rank"]) == 1:
                atomic_write_json(preview_dir / "cushing_syndrome__r01__betamethasone.json", packet)
                break
        atomic_write_json(preview_dir / "first_packet.json", packets[0])

        print("\nPASS: grounded evidence packets built")
        print(f"Packets: {output}")
        print(f"Coverage: {args.coverage_csv}")
        print(f"Coverage summary: {args.coverage_json}")
        print(f"Human review preview: {preview_dir / 'cushing_syndrome__r01__betamethasone.json'}")
        print("GraphDRx rerun: NO")
    except EvidenceBuildError as exc:
        raise SystemExit(f"EVIDENCE BUILD FAILED: {exc}") from exc


if __name__ == "__main__":
    main()
