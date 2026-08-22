#!/usr/bin/env python3
"""Public GraphDRx experiment entry point.

Area adaptation is restricted to aggregate KG-topology policies stored in
``configs/graphdrx_method_config.json``. Runtime code does not contain disease-name
rules, candidate allowlists, or performance-derived area weights.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

ROOT = Path(__file__).resolve().parent
ENGINE_PATH = ROOT / "graphdrx_retrieval_engine.py"
DEFAULT_CONFIG = ROOT / "configs" / "graphdrx_method_config.json"

AREA_ALIASES = {
    "neuro": "neuro_mental",
    "neurology": "neuro_mental",
    "neurologic": "neuro_mental",
    "neurological": "neuro_mental",
    "mental": "neuro_mental",
    "mental_health": "neuro_mental",
    "neuromental": "neuro_mental",
    "neuro_mentalic": "neuro_mental",
    "neuro/mental": "neuro_mental",
    "metabolic/endocrine": "metabolic_endocrine",
    "autoimmune/derm/inflammatory": "autoimmune_derm_inflammatory",
    "hematologic/genetic": "hematologic_genetic",
}


def normalize_area(value: Any) -> str:
    key = str(value or "").strip().lower().replace("/", "_").replace("-", "_").replace(" ", "_")
    key = re.sub(r"_+", "_", key).strip("_")
    return AREA_ALIASES.get(key, key)


def normalize_name(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_case_key(seed: int, area: str, disease: str) -> str:
    raw = f"{seed}|{normalize_area(area)}|{normalize_name(disease)}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def parse_name_values(values: Optional[Sequence[str]]) -> set[str]:
    out: set[str] = set()
    for value in values or []:
        for part in re.split(r"[,|]", str(value or "")):
            part = normalize_name(part)
            if part:
                out.add(part)
    return out


def load_name_file(path: Optional[str]) -> set[str]:
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Disease list file not found: {p}")
    out: set[str] = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        first = re.split(r"[,\t]", line, maxsplit=1)[0].strip()
        if normalize_name(first) not in {"disease", "disease_name"}:
            out.add(normalize_name(first))
    return out


def select_case_rows(
    csv_path: str,
    area_filter: Optional[str] = None,
    disease_names: Optional[Iterable[str]] = None,
    sample_per_area: Optional[int] = None,
    sample_seed: int = 42,
    max_cases: Optional[int] = None,
) -> pd.DataFrame:
    """Select cases without using labels, drug lists, or prior model performance."""
    df = pd.read_csv(csv_path)
    if "disease" not in df.columns or "area" not in df.columns:
        raise ValueError("Test-case CSV must contain disease and area columns.")
    df = df.copy()
    df["area"] = df["area"].map(normalize_area)
    df["_disease_norm"] = df["disease"].map(normalize_name)

    if area_filter:
        allowed = {normalize_area(x) for x in re.split(r"[,|]", area_filter) if str(x).strip()}
        df = df[df["area"].isin(allowed)]

    requested = {normalize_name(x) for x in (disease_names or []) if normalize_name(x)}
    if requested:
        found = set(df["_disease_norm"])
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"Disease names not found after area filtering: {missing}")
        df = df[df["_disease_norm"].isin(requested)]

    if sample_per_area is not None and int(sample_per_area) > 0:
        n = int(sample_per_area)
        sampled = []
        for area, group in df.groupby("area", sort=True):
            g = group.copy()
            g["_sample_key"] = [stable_case_key(sample_seed, area, x) for x in g["disease"]]
            sampled.append(g.sort_values(["_sample_key", "_disease_norm"]).head(n))
        df = pd.concat(sampled, ignore_index=True) if sampled else df.head(0)

    df = df.sort_values(["area", "_disease_norm"]).reset_index(drop=True)
    if max_cases is not None and int(max_cases) > 0:
        df = df.head(int(max_cases)).copy()
    return df.drop(columns=[c for c in ["_sample_key"] if c in df.columns])


def build_retrieval_cases(selected_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Create retrieval-safe case objects from a benchmark selection.

    Only identifiers needed by retrieval are retained. Gold drugs, example drugs,
    counts, and selection annotations remain outside the engine input.
    """
    return [
        {
            "disease": str(row["disease"]),
            "area": normalize_area(row["area"]),
            "case_id": normalize_name(row["disease"]).replace(" ", "_"),
            "use_kg_gold": True,
        }
        for _, row in selected_df.iterrows()
    ]


def import_engine() -> Any:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(Path.cwd()))
    spec = importlib.util.spec_from_file_location("graphdrx_retrieval_engine", ENGINE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import engine: {ENGINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def set_seed(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)


def apply_config_to_engine(engine: Any, cfg: Dict[str, Any], output_root: Path) -> None:
    retrieval = cfg.get("retrieval", {}) or {}
    drug_rerank = cfg.get("drug_rerank", {}) or {}
    engine.OUT_BASE_DIR = output_root
    engine.CANDIDATE_K = int(retrieval.get("candidate_k", engine.CANDIDATE_K))
    engine.PATH_LIMIT_PER_PATTERN = int(retrieval.get("path_limit_per_pattern", engine.PATH_LIMIT_PER_PATTERN))
    engine.VERIFY_TIMEOUT_SEC = int(retrieval.get("prior_verify_timeout_sec", engine.VERIFY_TIMEOUT_SEC))
    engine.DEFAULT_PRIOR_VERIFY_PATTERNS = list(retrieval.get("prior_verify_enabled_patterns", engine.DEFAULT_PRIOR_VERIFY_PATTERNS))
    engine.DRUG_EMBEDDING_WEIGHT = float(drug_rerank.get("drug_embedding_weight", engine.DRUG_EMBEDDING_WEIGHT))
    engine.OFFLINE_RERANK_DRUG_SIM_WEIGHT = float(drug_rerank.get("offline_rerank_drug_weight", engine.OFFLINE_RERANK_DRUG_SIM_WEIGHT))

    anchor_policy_cfg = cfg.get("anchor_pattern_policy", {}) or {}
    engine.GLOBAL_DISABLED_ANCHOR_PATTERNS = {
        str(name) for name in (anchor_policy_cfg.get("globally_disabled") or [])
    }
    anchor_policy = anchor_policy_cfg.get("areas", {}) or {}
    engine.AREA_ANCHOR_PATTERN_POLICY = {
        normalize_area(area): {
            "disabled": set(spec.get("disabled", [])),
            "limits": dict(spec.get("limits", {})),
            "timeouts": dict(spec.get("timeouts", {})),
        }
        for area, spec in anchor_policy.items()
    }

    structural = (cfg.get("structural_area_policy", {}) or {}).get("areas", {}) or {}
    engine.STRUCTURAL_AREA_POLICY = {normalize_area(k): dict(v) for k, v in structural.items()}
    engine.SPARSE_RESCUE_DEFAULT_AREAS = {
        normalize_area(area)
        for area, spec in structural.items()
        if bool((spec.get("sparse_rescue") or {}).get("enabled", False))
    }
    engine.TRANSLATIONAL_CALIBRATION_CONFIG = dict(cfg.get("translational_calibration", {}) or {})

    scoring_cfg = dict(cfg.get("scoring", {}) or {})
    if scoring_cfg.get("drug_gene_action_type_weights"):
        engine.DRUG_GENE_ACTION_TYPE_WEIGHTS = {
            str(k): float(v) for k, v in scoring_cfg["drug_gene_action_type_weights"].items()
        }
    if scoring_cfg.get("drug_gene_relation_weights"):
        engine.DRUG_GENE_ACTION_RELATION_FALLBACK_WEIGHTS = {
            str(k): float(v) for k, v in scoring_cfg["drug_gene_relation_weights"].items()
        }
    if scoring_cfg.get("sparse_drug_action_type_weights"):
        engine.SPARSE_DRUG_ACTION_TYPE_WEIGHTS = {
            str(k): float(v) for k, v in scoring_cfg["sparse_drug_action_type_weights"].items()
        }
    if scoring_cfg.get("sparse_drug_relation_weights"):
        engine.SPARSE_DRUG_TARGET_ACTION_REL_WEIGHTS = {
            str(k): float(v) for k, v in scoring_cfg["sparse_drug_relation_weights"].items()
        }
    if scoring_cfg.get("sparse_disease_drug_relation_weights"):
        declared_disease_drug_weights = {
            str(k): float(v) for k, v in scoring_cfg["sparse_disease_drug_relation_weights"].items()
        }
        engine.SPARSE_DISEASE_DRUG_PRIOR_REL_WEIGHTS = dict(declared_disease_drug_weights)
        engine.DISEASE_DRUG_VERIFICATION_REL_WEIGHTS = dict(declared_disease_drug_weights)
    engine.GENERAL_RERANK_CONFIG = dict(scoring_cfg.get("final_integration", {}) or engine.GENERAL_RERANK_CONFIG)

    family_specificity = cfg.get("family_hierarchy_specificity", {}) or {}
    engine.FAMILY_HIERARCHY_MIN_SPECIFICITY = float(
        family_specificity.get("minimum_specificity", engine.FAMILY_HIERARCHY_MIN_SPECIFICITY)
    )
    engine.FAMILY_HIERARCHY_SUPPORT_BASE = float(
        family_specificity.get("support_base", engine.FAMILY_HIERARCHY_SUPPORT_BASE)
    )
    engine.FAMILY_HIERARCHY_MAX_REQUIRED_SUPPORTS = int(
        family_specificity.get("max_required_supports", engine.FAMILY_HIERARCHY_MAX_REQUIRED_SUPPORTS)
    )

    floor_cfg = cfg.get("candidate_floor_rescue", {}) or {}
    engine.GLOBAL_CANDIDATE_FLOOR = int(
        floor_cfg.get("minimum_candidates", engine.GLOBAL_CANDIDATE_FLOOR)
    )
    if floor_cfg:
        engine.GLOBAL_CANDIDATE_FLOOR_POLICY = engine._merge_sparse_policy(
            engine.GLOBAL_CANDIDATE_FLOOR_POLICY,
            {k: v for k, v in floor_cfg.items() if k != "minimum_candidates"},
        )

    retry_cfg = cfg.get("pattern_timeout_retry", {}) or {}
    if retry_cfg:
        engine.PATTERN_TIMEOUT_RETRY_CONFIG = dict(retry_cfg)

    # Relation semantics carry labels/polarity only. All runtime numeric weights
    # are loaded from the single `scoring` section above, preventing duplicate
    # or hidden max(config, engine) overrides.
    semantics = cfg.get("relation_semantics", {}) or {}
    for rel, spec in semantics.items():
        label = spec.get("semantic_label")
        if label:
            engine.SPARSE_RELATION_SEMANTIC_LABELS[str(rel)] = str(label)


def apply_ablation_flags(engine: Any, args: argparse.Namespace) -> None:
    engine.ENABLE_STRUCTURAL_SCORING = not args.disable_structural_scoring
    engine.ENABLE_VERIFIER_EVIDENCE_HIERARCHY = not args.disable_verifier_evidence_hierarchy
    engine.ENABLE_TOPOLOGY_SPECIFICITY = bool(args.enable_topology_specificity)
    engine.ENABLE_TRANSLATIONAL_CALIBRATION = not args.disable_translational_calibration
    if args.disable_structural_scoring:
        engine.AREA_ANCHOR_PATTERN_POLICY = {}
        engine.STRUCTURAL_AREA_POLICY = {
            area: {**spec, "prior_verify_patterns": list(engine.DEFAULT_PRIOR_VERIFY_PATTERNS), "sparse_rescue": {"enabled": False}}
            for area, spec in engine.STRUCTURAL_AREA_POLICY.items()
        }


def build_suffix(args: argparse.Namespace) -> str:
    parts = [args.run_name]

    if args.area_filter:
        parts.append(
            "area-" +
            re.sub(r"[^a-z0-9]+", "-", normalize_area(args.area_filter)).strip("-")
        )

    if args.sample_per_area:
        parts.append(f"sample-{int(args.sample_per_area)}-per-area")
    if args.disease or args.disease_filter or args.disease_list_file:
        parts.append("disease-subset")
    if args.disable_target_context_graph:
        parts.append("no-anchor-T")
    if args.disable_vector_anchor_graph:
        parts.append("no-vector-B")
    if args.disable_semantic_prior_branch:
        parts.append("no-semantic-C")
    if args.disable_common_direct_graph:
        parts.append("no-direct-D")

    parts.append("single-indirect-only-reserve")

    return "_" + "_".join(parts)


def manifest_payload(cfg: Dict[str, Any], args: argparse.Namespace, selected_df: pd.DataFrame) -> Dict[str, Any]:
    files = [ENGINE_PATH, Path(__file__).resolve(), Path(args.config).resolve()]
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": cfg.get("method_name", "GraphDRx retrieval pipeline"),
        "config_id": cfg.get("config_id", "graphdrx_method"),
        "run_name": args.run_name,
        "selection": {
            "test_cases_csv": args.test_cases_csv,
            "area_filter": args.area_filter,
            "diseases": list(args.disease or []),
            "disease_filter": args.disease_filter,
            "disease_list_file": args.disease_list_file,
            "sample_per_area": args.sample_per_area,
            "sample_seed": args.sample_seed,
            "max_cases": args.max_cases,
            "selected_count": int(len(selected_df)),
            "selected_cases": selected_df[["disease", "area"]].to_dict(orient="records"),
        },
        "flags": {
            "mechanism_lookup_enabled": not args.disable_mechanism_lookup,
            "disease_embedding_prior_enabled": not args.disable_disease_embedding_prior,
            "target_disease_context_enabled": not args.disable_target_context_graph,
            "vector_neighbor_context_enabled": not args.disable_disease_embedding_prior,
            "vector_neighbor_anchor_graph_enabled": bool(not args.disable_disease_embedding_prior and not args.disable_vector_anchor_graph),
            "semantic_prior_branch_enabled": bool(not args.disable_disease_embedding_prior and not args.disable_semantic_prior_branch),
            "prior_drug_target_graph_enabled": bool(
                not args.disable_disease_embedding_prior
                and not args.disable_semantic_prior_branch
                and not args.disable_prior_drug_graph
            ),
            "common_direct_disease_gene_graph_enabled": not args.disable_common_direct_graph,
            "drug_embedding_enabled": not args.disable_drug_embedding,
            "general_rerank_enabled": not args.disable_general_rerank,
            "structural_evidence_ranking_enabled": True,
            "structural_evidence_ranking_mode": "single_indirect_only_reserve",
            "sparse_area_rescue_enabled": bool(args.enable_sparse_area_rescue and not args.disable_sparse_area_rescue),
            "structural_scoring_enabled": not args.disable_structural_scoring,
            "verifier_evidence_hierarchy_enabled": not args.disable_verifier_evidence_hierarchy,
            "topology_specificity_enabled": bool(args.enable_topology_specificity),
            "translational_calibration_enabled": not args.disable_translational_calibration,
            "mask_eval_diseases_from_prior": args.mask_eval_diseases_from_prior,
            "eval1_rank_view": "retrieval_final",
            "default_downstream_rank_view": "translational_priority",
        },
        "structural_evidence_ranking": {
            "enabled": True,
            "mode": "single_indirect_only_reserve",
            "prior_threshold": float(
                (cfg.get("structural_evidence_ranking", {}) or {}).get(
                    "prior_threshold", 1e-12
                )
            ),
            "changes_numeric_scores": False,
            "deletes_candidates": False,
            "uses_gold_labels_at_runtime": False,
        },
        "design_constraints": cfg.get("design_constraints", {}),
        "sha256": {str(x.relative_to(ROOT.parent) if ROOT.parent in x.parents else x): sha256_file(x) for x in files if x.exists()},
        "config": cfg,
    }


def print_selected_cases(df: pd.DataFrame) -> None:
    print(f"[SELECTED CASES] n={len(df)}")
    for area, group in df.groupby("area", sort=True):
        print(f"  {area}: {len(group)}")
        for disease in group["disease"].tolist():
            print(f"    - {disease}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run GraphDRx with an independently frozen nonbenchmark topology policy.")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", ""))
    p.add_argument("--test-cases-csv", default="data/graphdrx_disease_benchmark.csv")
    p.add_argument("--drug-rag-csv", default=os.getenv("GRAPHDRX_DRUG_RAG_CSV", "data/rag_corpus/Drug_data_All_RAG.csv"))
    p.add_argument("--disease-features-csv", default=os.getenv("GRAPHDRX_DISEASE_RAG_CSV", "data/rag_corpus/disease_features.csv"))
    p.add_argument("--output-root", default="output")
    p.add_argument("--run-name", default="main")

    p.add_argument("--area-filter", default=None, help="Comma-separated area names.")
    p.add_argument("--disease", action="append", default=[], help="Run one disease; repeat for multiple diseases.")
    p.add_argument("--disease-filter", default=None, help="Compatibility alias: comma- or pipe-separated disease names.")
    p.add_argument("--disease-list-file", default=None)
    p.add_argument("--sample-per-area", type=int, default=None, help="Deterministically select N diseases per area.")
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--dry-run", action="store_true", help="Print selected diseases and exit before importing Neo4j/Ollama code.")
    p.add_argument("--resume", action="store_true", help="Continue from the existing incremental all_results file in the selected output directory.")

    p.add_argument("--disable-mechanism-lookup", action="store_true")
    p.add_argument("--disable-disease-embedding-prior", action="store_true")
    p.add_argument("--disable-target-context-graph", action="store_true", help="Disable target-disease biological-context retrieval (T).")
    p.add_argument("--disable-vector-anchor-graph", action="store_true", help="Disable branch 1-2: vector-neighbor KG anchor candidate generation.")
    p.add_argument("--disable-prior-drug-graph", action="store_true", help="Disable only C_v prior-drug target-graph verification while retaining C_p support on graph-existing candidates.")
    p.add_argument(
        "--disable-semantic-prior-branch",
        action="store_true",
        help="Clean C-branch ablation: retain disease-vector neighbors for B, but remove C_p scores and all C_v prior-only candidate admission.",
    )
    p.add_argument(
        "--disable-common-direct-graph",
        action="store_true",
        help="Clean D-branch ablation: disable the axis-independent direct DRUG-GENE-DISEASE candidate query.",
    )
    p.add_argument("--disable-drug-embedding", action="store_true")
    p.add_argument("--disable-general-rerank", action="store_true")
    p.add_argument("--prior-mask-disease-list-csv", default=None)
    p.add_argument("--mask-eval-diseases-from-prior", action="store_true")
    p.add_argument("--max-similarity-for-prior", type=float, default=None)
    p.add_argument("--keep-unverified-prior-candidates", action="store_true")
    p.add_argument("--enable-sparse-area-rescue", action="store_true")
    p.add_argument("--disable-sparse-area-rescue", action="store_true")
    p.add_argument("--sparse-rescue-areas", default="infectious,hematologic_genetic")
    p.add_argument("--sparse-rescue-protect-top-n", type=int, default=None)
    p.add_argument("--disable-structural-scoring", "--disable-kg-structure-weighting", dest="disable_structural_scoring", action="store_true")
    p.add_argument(
        "--disable-verifier-evidence-hierarchy",
        action="store_true",
        help="Ablation: restore the original prior-verifier treatment of gene-gene and BP/PATH bridge evidence.",
    )
    p.add_argument(
        "--enable-topology-specificity",
        action="store_true",
        help="Optional experiment: additionally activate drug/disease-gene/bridge/phenotype degree penalties.",
    )
    p.add_argument("--disable-translational-calibration", "--disable-area-property-calibration", dest="disable_translational_calibration", action="store_true")
    p.add_argument("--disable-area-policy-final", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_json(Path(args.config))
    set_seed(int(cfg.get("random_seed", 42)))

    requested = parse_name_values(args.disease) | parse_name_values([args.disease_filter]) | load_name_file(args.disease_list_file)
    selected_df = select_case_rows(
        args.test_cases_csv,
        area_filter=args.area_filter,
        disease_names=requested,
        sample_per_area=args.sample_per_area,
        sample_seed=args.sample_seed,
        max_cases=args.max_cases,
    )
    if selected_df.empty:
        raise ValueError("No cases selected.")
    print_selected_cases(selected_df)
    if args.dry_run:
        print("[DRY RUN COMPLETE] No Neo4j or model call was made.")
        return

    engine = import_engine()
    apply_config_to_engine(engine, cfg, Path(args.output_root))
    apply_ablation_flags(engine, args)

    # Pass only retrieval-safe identifiers into the engine. Benchmark columns
    # containing drug examples, gold counts, or selection metadata are never
    # included in the retrieval case object. Evaluation labels are queried
    # separately after ranking has been finalized.
    cases = build_retrieval_cases(selected_df)

    disease_prior_cfg = cfg.get("disease_prior", {}) or {}
    drug_cfg = cfg.get("drug_rerank", {}) or {}
    structural_cfg = cfg.get("structural_evidence_ranking", {}) or {}
    structural_prior_threshold = float(
        structural_cfg.get("prior_threshold", 1e-12)
    )
    eval_diseases: List[str] = []
    if args.mask_eval_diseases_from_prior:
        mask_csv = args.prior_mask_disease_list_csv or args.test_cases_csv
        mask_df = pd.read_csv(mask_csv)
        if "disease" in mask_df.columns:
            eval_diseases = [str(x).strip() for x in mask_df["disease"].dropna().tolist() if str(x).strip()]

    out_dir = Path(args.output_root) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_payload(cfg, args, selected_df)
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "graphdrx_method_config_used.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    previous = os.environ.get("GRAPHDRX_FORCE_OUTPUT_DIR")
    os.environ["GRAPHDRX_FORCE_OUTPUT_DIR"] = str(out_dir)
    try:
        engine.run_pilot(
            ablation_mode="conditional",
            neo4j_password=args.neo4j_password,
            test_cases=cases,
            drug_rag_csv=args.drug_rag_csv,
            disease_features_csv=args.disease_features_csv,
            use_drug_embedding_rerank=not args.disable_drug_embedding,
            drug_embedding_weight=float(drug_cfg.get("drug_embedding_weight", 0.6)),
            use_general_offline_rerank=not args.disable_general_rerank,
            offline_rerank_drug_weight=float(drug_cfg.get("offline_rerank_drug_weight", 0.4)),
            output_tag_suffix=build_suffix(args),
            use_mechanism_lookup=not args.disable_mechanism_lookup,
            use_disease_embedding_prior=not args.disable_disease_embedding_prior,
            use_target_context_graph=not args.disable_target_context_graph,
            use_vector_anchor_graph=not args.disable_vector_anchor_graph,
            use_prior_drug_graph=not args.disable_prior_drug_graph,
            use_semantic_prior_branch=not args.disable_semantic_prior_branch,
            use_common_direct_graph=not args.disable_common_direct_graph,
            structural_reserve_prior_threshold=structural_prior_threshold,
            eval_diseases_for_prior_mask=eval_diseases,
            mask_eval_diseases_from_prior=args.mask_eval_diseases_from_prior,
            max_similarity_for_prior=args.max_similarity_for_prior,
            disease_prior_mode=str(disease_prior_cfg.get("mode", "vector_anchor_verify")),
            prior_verify_top_n=int(disease_prior_cfg.get("prior_verify_top_n", 120)),
            prior_verify_limit_per_pattern=int(disease_prior_cfg.get("prior_verify_limit_per_pattern", 5)),
            keep_unverified_prior_candidates=args.keep_unverified_prior_candidates,
            vector_anchor_top_diseases=int(disease_prior_cfg.get("vector_anchor_top_diseases", 30)),
            vector_anchor_max_genes=int(disease_prior_cfg.get("vector_anchor_max_genes", 100)),
            vector_anchor_max_terms=int(disease_prior_cfg.get("vector_anchor_max_terms", 140)),
            enable_sparse_area_rescue=(args.enable_sparse_area_rescue and not args.disable_sparse_area_rescue and not args.disable_structural_scoring),
            sparse_rescue_areas=[x.strip() for x in str(args.sparse_rescue_areas).split(",") if x.strip()],
            sparse_rescue_protect_top_n=args.sparse_rescue_protect_top_n,
            resume_existing=args.resume,
        )
    finally:
        if previous is None:
            os.environ.pop("GRAPHDRX_FORCE_OUTPUT_DIR", None)
        else:
            os.environ["GRAPHDRX_FORCE_OUTPUT_DIR"] = previous

    (out_dir / "run_manifest.json").write_text(json.dumps(manifest_payload(cfg, args, selected_df), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[GRAPHDRX COMPLETE] {out_dir}")


if __name__ == "__main__":
    main()
