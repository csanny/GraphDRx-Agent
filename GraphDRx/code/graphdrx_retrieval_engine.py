# ============================================================
# GraphDRx: KG-grounded drug-repurposing retrieval engine
#
# Area adaptation is frozen from an independent nonbenchmark KG-topology audit:
# motif availability, graph sparsity, candidate expansion, and node degree.
# No disease-name rules, drug/gene allowlists, gold labels, or outcome-derived
# weights are used by retrieval or scoring.
# ============================================================

import os
import re
import csv
import json
import time
import math
import copy
import argparse
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable
from collections import defaultdict, Counter
import ast
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import signal
from contextlib import contextmanager
from neo4j import Query
from neo4j import GraphDatabase
from graphdrx_disease_prior import (
    prepare_embedding_prior_for_case,
    merge_graph_candidates_with_embedding_priors,
)
from graphdrx_drug_prior import (
    load_drug_rag_df,
    apply_drug_embedding_rerank_for_case,
)
# ============================================================
# 0. Settings
# ============================================================

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "") 

ABLATION_MODE = "conditional"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

CANDIDATE_K = 200
PATH_LIMIT_PER_PATTERN = 30
VERIFY_TIMEOUT_SEC = int(os.getenv("GRAPHDRX_PRIOR_VERIFY_TIMEOUT_SEC", "25"))
# Prior-drug seed verification uses a fixed, paper-facing pattern policy.
# Unknown areas fall back to the two fast core verifiers; known areas can add
# phenotype or family-label transfer according to area-level policy below.
DEFAULT_PRIOR_VERIFY_PATTERNS = ["gene_gene", "bio_bridge"]
CURRENT_CASE_AREA = ""
STRUCTURAL_AREA_POLICY: Dict[str, Dict[str, Any]] = {}
TRANSLATIONAL_CALIBRATION_CONFIG: Dict[str, Any] = {}
GENERAL_RERANK_CONFIG: Dict[str, Any] = {
    "disease_prior_weight": 1.0,
    "drug_embedding_weight": 0.4,
    "side_effect_risk_weight": 0.1,
}
ENABLE_TRANSLATIONAL_CALIBRATION = True
ENABLE_STRUCTURAL_SCORING = True
# Prior-verifier evidence hierarchy and topology-degree correction are separate.
# The default experiment keeps the evidence hierarchy enabled while leaving the
# newly activated drug/bridge degree penalties disabled. This isolates the effect
# of ranking direct gene-gene support above unmatched broad BP/PATH bridges.
ENABLE_VERIFIER_EVIDENCE_HIERARCHY = True
ENABLE_TOPOLOGY_SPECIFICITY = False
OUTPUT_COMPACT_TOP_K = int(os.getenv("GRAPHDRX_OUTPUT_COMPACT_TOP_K", "200"))
WRITE_FULL_ALL_RESULTS_JSON = os.getenv("GRAPHDRX_WRITE_FULL_ALL_RESULTS_JSON", "0").strip().lower() not in {"0", "false", "no"}


AREA_ALIASES = {
    "neuro": "neuro_mental",
    "neuro_mental": "neuro_mental",
    "neurology": "neuro_mental",
    "neurologic": "neuro_mental",
    "neurological": "neuro_mental",
    "mental": "neuro_mental",
    "mental_health": "neuro_mental",
    "neuromental": "neuro_mental",
    "neuro_mentalic": "neuro_mental",
    "neuro__mental": "neuro_mental",
    "neuro/mental": "neuro_mental",
    "neuro/mentalic": "neuro_mental",
    "metabolic-endocrine": "metabolic_endocrine",
    "metabolic/endocrine": "metabolic_endocrine",
    "autoimmune-derm-inflammatory": "autoimmune_derm_inflammatory",
    "autoimmune/derm/inflammatory": "autoimmune_derm_inflammatory",
    "hematologic-genetic": "hematologic_genetic",
    "hematologic/genetic": "hematologic_genetic",
}


def normalize_area_name(area: Any) -> str:
    """Canonicalize benchmark and config area labels.

    Runtime code uses neuro_mental as the canonical key while reports can
    display it as neuro/mental. Slashes, spaces, and dashes are normalized so
    old benchmark labels such as neuro, neuro/mentalic, and neuro mental map to
    the same area.
    """
    key = str(area or "").strip().lower()
    key = key.replace("/", "_").replace("-", "_").replace(" ", "_")
    key = re.sub(r"_+", "_", key).strip("_")
    return AREA_ALIASES.get(key, key)


def display_area_name(area: Any) -> str:
    return "neuro/mental" if normalize_area_name(area) == "neuro_mental" else normalize_area_name(area)


def area_prior_verify_patterns(area: Any, disease: Any = None) -> List[str]:
    """Return verifier motifs from the topology-only area policy.

    `disease` is accepted for API compatibility but is deliberately ignored.
    """
    a = normalize_area_name(area)
    cfg = STRUCTURAL_AREA_POLICY.get(a, {}) or {}
    patterns = cfg.get("prior_verify_patterns")
    if patterns:
        return [str(x) for x in patterns]
    return list(DEFAULT_PRIOR_VERIFY_PATTERNS)



def structural_area_option(area: Any, key: str, default: Any = None) -> Any:
    """Read a topology-derived area option without consulting disease names."""
    cfg = STRUCTURAL_AREA_POLICY.get(normalize_area_name(area), {}) or {}
    return cfg.get(key, default)


def enabled_prior_verify_patterns_for_current_case(disease: Any = None) -> List[str]:
    return area_prior_verify_patterns(CURRENT_CASE_AREA, disease=disease)

USE_MECHANISM_LOOKUP = True
USE_DISEASE_EMBEDDING_PRIOR = True

OUT_BASE_DIR = Path("output")
OUT_DIR = None  # Output directories are created per run by the pipeline.

# Drug MoA embedding rerank.
# Safe drug embedding uses MoA/pharmacodynamics/pathway/drug-gene actions, not indication text.
USE_DRUG_EMBEDDING_RERANK = True
DRUG_EMBEDDING_WEIGHT = 0.6
DRUG_EMBEDDING_MAX_CANDIDATES = 300
DRUG_RAG_CSV = os.getenv("GRAPHDRX_DRUG_RAG_CSV", "data/rag_corpus/Drug_data_All_RAG.csv")
DISEASE_RAG_CSV = os.getenv("GRAPHDRX_DISEASE_RAG_CSV", "data/rag_corpus/disease_features.csv")

# Optional external evaluation set CSV. Expected column: disease.


# ------------------------------------------------------------
# Disease-agnostic final rerank
# ------------------------------------------------------------
# Integrates graph evidence, masked disease-neighbor prior, drug-representation
# similarity, and disease-specific side-effect counter-evidence exactly once.
# Translational properties are deferred to the final global calibration.
USE_GENERAL_OFFLINE_RERANK = True
OFFLINE_RERANK_DRUG_SIM_WEIGHT = 0.4

HIT_KS = [1, 5, 10, 20, 50, 100, 200]

PATTERN_LIMITS = {
    "drug_to_anchor_gene": 20,
    "drug_to_disease_direct_gene_action": 20,
    "drug_to_anchor_bioprocess": 15,
    "drug_to_disease_anchor_gene": 30,
    "drug_to_disease_via_anchor_bio": 30,
    "drug_to_disease_via_anchor_gene_path": 35,
    "drug_target_to_anchor_disease_gene_via_bio": 30,
    "drug_target_ppi_to_anchor_gene": 30,            # narrow PPI replacement
    "drug_to_disease_via_vector_phenotype_gene": 10,
}

PATTERN_TIMEOUTS = {
    "drug_to_anchor_gene": 20,
    "drug_to_disease_direct_gene_action": 20,
    "drug_to_anchor_bioprocess": 20,
    "drug_to_disease_anchor_gene": 30,
    "drug_to_disease_via_anchor_bio": 35,
    "drug_to_disease_via_anchor_gene_path": 45,
    "drug_target_to_anchor_disease_gene_via_bio": 45,
    "drug_target_ppi_to_anchor_gene": 30,
    "drug_to_disease_via_vector_phenotype_gene": 18,
}

# All defined anchor-guided patterns below are potentially executable; obsolete
# disabled query definitions were removed from this minimal paper configuration.
DISABLED_ANCHOR_PATTERNS = set()


# Area-level motif budgets are injected from graphdrx_method_config.json.
# The engine has no embedded disease, drug, gene, or outcome-derived policy.
AREA_ANCHOR_PATTERN_POLICY: Dict[str, Dict[str, Any]] = {}
# Independent nonbenchmark structural audits may identify patterns that are
# nonselective across every area. These global exclusions are loaded from the
# method config and therefore do not depend on the query disease or benchmark.
GLOBAL_DISABLED_ANCHOR_PATTERNS: set = set()

def area_anchor_pattern_policy(area: Any) -> Dict[str, Any]:
    return AREA_ANCHOR_PATTERN_POLICY.get(normalize_area_name(area), {})


def area_pattern_is_disabled(area: Any, pattern_name: str) -> bool:
    name = str(pattern_name)
    if name in GLOBAL_DISABLED_ANCHOR_PATTERNS:
        return True
    cfg = area_anchor_pattern_policy(area)
    return name in set(cfg.get("disabled") or set())


def area_pattern_limit(area: Any, pattern_name: str, default_limit: int) -> int:
    cfg = area_anchor_pattern_policy(area)
    limits = cfg.get("limits") or {}
    return int(limits.get(pattern_name, default_limit))


def area_pattern_timeout(area: Any, pattern_name: str, default_timeout: int) -> int:
    cfg = area_anchor_pattern_policy(area)
    timeouts = cfg.get("timeouts") or {}
    return int(timeouts.get(pattern_name, default_timeout))

DIRECT_GOLD_RELS = [
    "indication",
]

OFFLABEL_GOLD_RELS = [
    "indication",
    "off_label_use",
]

SUPPORTIVE_POSITIVE_RELS_DISCOVERY = [
    "indication",
    "off_label_use",
    "tested_indication",
    "studied_for_treatment_of",
]

DEFAULT_POSITIVE_RELS_DISCOVERY = DIRECT_GOLD_RELS


NEGATIVE_DD_RELS = [
    "contraindication",
]

PROHIBITED_PATH_RELS = [
    "indication",
    "off_label_use",
    "tested_indication",
    "studied_for_treatment_of",
    "studied_for_marker_mechanism_of",
    "contraindication",
]

DISEASE_CONTEXT_FIELDS_CORE = [
    "mondo_name",
    "group_name_bert",

    "mondo_definition",
    "umls_description",
    "orphanet_definition",
    "orphanet_clinical_description",

    "mayo_symptoms",
    "mayo_causes",
    "mayo_risk_factors",
    "mayo_complications",
]

DISEASE_CONTEXT_FIELDS_OPTIONAL = [
    "orphanet_prevalence",
    "orphanet_epidemiology",
]

DISEASE_CONTEXT_FIELDS_EXCLUDE = [
    "orphanet_management_and_treatment",
    "mayo_prevention",
    "mayo_see_doc",
]

DISEASE_KG_CONTEXT = {
    "disease_genes": True,          # DISEASE - associated_with - GENE
    "phenotypes": True,             # DISEASE - phenotype_present/absent - PHENO
    "disease_family_names": True,   # DISEASE - parent_child - DISEASE
    "exposures": False,             # EXPO는 제외
    "direct_drug_disease_edges": False,
}


DRUG_PHENOTYPE_RISK_RELS = [
    "side_effect",
]

DRUG_GENE_RELS = [
    "target",
    "inhibition",
    "activation",
    "binding",
    "modulation",
    "enzyme",
    "transporter",
    "carrier",
]

GENERIC_BRIDGE_PATTERNS = [
    "positive regulation of",
    "negative regulation of",
    "regulation of transcription",
    "transcription by rna polymerase ii",
    "gene expression",
    "protein phosphorylation",
    "kinase activity",
    "cell population proliferation",
    "cell-cell signaling",
    "intracellular signal transduction",
    "g protein-coupled receptor signaling pathway",
    "protein transport",
    "metal ion binding",
    "protein kinase binding",
    "regulation of apoptotic process",
    "cellular response to chemical stimulus",
    "response to drug",
]

GENERIC_BIO_TERMS = {
    "protein binding",
    "identical protein binding",
    "plasma membrane",
    "integral component of plasma membrane",
    "integral component of membrane",
    "cytoplasm",
    "cytosol",
    "extracellular exosome",
    "extracellular space",
    "cell surface",
    "membrane",
    "protein transport",
    "ion transport",
    "protein-containing complex",
    "response to toxic substance",
    "transmembrane transport",
    "xenobiotic metabolic process",
    "signal transduction",
    "immune response",
    "inflammatory response",
    "immune system process",
    "signaling receptor activity",
    "signaling receptor binding",
    "enzyme binding",
    "receptor binding",
    
    # CC / MF generic
    "nucleus",
    "nucleoplasm",
    "metal ion binding",
    "protein kinase binding",
    "kinase binding",
    "enzyme binding",
    "protein serine/threonine kinase activity",

    # broad signaling / transcription
    "intracellular signal transduction",
    "cell-cell signaling",
    "g protein-coupled receptor signaling pathway",
    "regulation of transcription by rna polymerase ii",
    "positive regulation of transcription by rna polymerase ii",
    "negative regulation of transcription by rna polymerase ii",
    "positive regulation of transcription, dna-templated",
    "negative regulation of transcription, dna-templated",
    "positive regulation of gene expression",
    "negative regulation of gene expression",

    # broad regulation
    "positive regulation of protein phosphorylation",
    "negative regulation of protein phosphorylation",
    "positive regulation of kinase activity",
    "negative regulation of kinase activity",
    "positive regulation of erk1 and erk2 cascade",
    "negative regulation of erk1 and erk2 cascade",
    "positive regulation of cell population proliferation",
    "negative regulation of cell population proliferation",
    "regulation of cell cycle",

    # too broad immune / transport
    "adaptive immune response",
    "innate immune response",
    "cytokine-mediated signaling pathway",
    "intracellular protein transport",

    "signaling",
    "regulation of signaling",
    "cell communication",
    "response to stimulus",
    "cell proliferation",
    "cell migration",
    "apoptotic process",
    "response to drug",
}


# Runtime answer maps are intentionally empty. Specificity is computed from KG degree.
FAMILY_LABEL_BROAD_CONTEXT_EXCLUDES: set = set()
BROAD_HUB_GENES: set = set()
GENE_ALIAS_MAP: Dict[str, str] = {}

DISEASE_CONNECTED_PATTERNS = {
    "drug_to_disease_direct_gene_action",
    "drug_to_disease_via_anchor_bio",
    "drug_to_disease_anchor_gene",
    "drug_to_disease_via_anchor_gene_path",
    "drug_target_to_anchor_disease_gene_via_bio",
    "drug_target_ppi_to_anchor_gene",
    "drug_to_disease_via_vector_phenotype_gene",
}
ANCHOR_ONLY_PATTERNS = {"drug_to_anchor_gene", "drug_to_anchor_bioprocess"}
BROAD_ANCHOR_TERMS: set = set()
VERY_BROAD_BUT_DISEASE_CONNECTED_ANCHORS: set = set()
EXACT_TOO_BROAD_ANCHORS: set = set()
NOISY_BUT_DISEASE_CONNECTED_ANCHORS: set = set()
FALLBACK_LOW_PRIORITY_GENES: set = set()
TARGET_CONTEXT_OFF_NOISY_BRIDGE_TERMS: set = set()
FALLBACK_CONTEXT_MECHANISM_RULES: List[Dict[str, Any]] = []
MECHANISM_CLASS_TO_TARGET_GENES: Dict[str, List[str]] = {}












# ============================================================
# 2. Basic helpers
# ============================================================


class WallClockTimeout(Exception):
    pass

@contextmanager
def wall_clock_timeout(seconds: int):
    def _handler(signum, frame):
        raise WallClockTimeout(f"wall-clock timeout after {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        
def normalize_name(x: Any) -> str:
    return str(x or "").strip().lower()


def normalize_gene(x: Any) -> str:
    s = str(x or "").strip().upper()
    s = s.replace("β", "BETA").replace("γ", "GAMMA")
    return GENE_ALIAS_MAP.get(s, s)


def normalize_mechanism_gene(g):
    x = normalize_gene(g)
    return GENE_ALIAS_MAP.get(x, x)


def unique_keep_order(items: List[Any]) -> List[str]:
    out = []
    seen = set()
    for x in items or []:
        s = str(x or "").strip()
        if not s:
            continue
        ns = normalize_name(s)
        if ns in seen:
            continue
        seen.add(ns)
        out.append(s)
    return out


def make_driver(neo4j_password: str | None = None):
    password = neo4j_password or NEO4J_PASSWORD

    if not password:
        raise RuntimeError(
            "Neo4j password is empty. Set it with:\n"
            "export NEO4J_PASSWORD='your_password'\n"
            "or pass --neo4j-password your_password"
        )

    return GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, password),
    )


def run_query_auto(
    driver,
    query: str,
    params: Optional[Dict[str, Any]] = None,
    timeout_sec: int = 180,
) -> List[Dict[str, Any]]:
    """
    Neo4j query with BOTH:
    1. Neo4j transaction timeout
    2. Python wall-clock timeout

    LIMIT only limits returned rows, not search time.
    """
    try:
        with wall_clock_timeout(timeout_sec):
            with driver.session() as session:
                result = session.run(
                    Query(query, timeout=timeout_sec),
                    params or {},
                )
                return [record.data() for record in result]

    except WallClockTimeout as e:
        raise TimeoutError(str(e))


def flatten_values(x: Any) -> List[Any]:
    """Flatten Neo4j scalar/list/array relationship properties."""
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        out: List[Any] = []
        for v in x:
            out.extend(flatten_values(v))
        return out
    return [x]


def max_numeric_value(x: Any, default: float = 0.0) -> float:
    vals: List[float] = []
    for v in flatten_values(x):
        try:
            vals.append(float(v))
        except Exception:
            continue
    return max(vals) if vals else default


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (list, tuple, set)):
            return max_numeric_value(x, default=default)
        return float(x)
    except Exception:
        return default


def rank_of_name(names: List[str], target: str) -> Optional[int]:
    nt = normalize_name(target)
    for i, name in enumerate(names, start=1):
        if normalize_name(name) == nt:
            return i
    return None


def hit_at_k(ranks: List[Optional[int]], k: int) -> bool:
    return any(r is not None and r <= k for r in ranks)


def reciprocal_rank(ranks: List[Optional[int]]) -> float:
    valid = [r for r in ranks if r is not None]
    if not valid:
        return 0.0
    return 1.0 / min(valid)


def mask_terms_in_text(text: str, terms: List[str]) -> str:
    masked = str(text or "")
    for term in sorted(terms or [], key=len, reverse=True):
        term = str(term or "").strip()
        if not term:
            continue
        masked = re.sub(
            re.escape(term),
            "[MASKED_DRUG]",
            masked,
            flags=re.IGNORECASE,
        )
    return masked

def generic_bridge_penalty(name: Any) -> float:
    lname = normalize_name(name)

    if not lname:
        return 0.0

    penalty = 0.0

    if lname in GENERIC_BIO_TERMS:
        penalty += 1.5

    if any(p in lname for p in GENERIC_BRIDGE_PATTERNS):
        penalty += 1.5

    return penalty


# ============================================================
# 2.5 Generic ontology-neighbor utilities
# ============================================================

def _simple_name_tokens(x: Any) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9]+", normalize_name(x)) if len(t) >= 3]


def build_disease_name_idf_from_df(disease_df: pd.DataFrame) -> Dict[str, float]:
    names = []
    if isinstance(disease_df, pd.DataFrame):
        for col in ("disease", "mondo_name", "group_name_bert"):
            if col in disease_df.columns:
                names.extend(str(x) for x in disease_df[col].dropna().tolist())
    docs = [set(_simple_name_tokens(x)) for x in names if str(x).strip()]
    n = max(len(docs), 1)
    counts = Counter(t for doc in docs for t in doc)
    return {t: math.log((n + 1.0) / (df + 1.0)) + 1.0 for t, df in counts.items()}


def _generic_context_reliability(query_disease: Any, context_name: Any, disease_name_idf: Optional[Dict[str, float]] = None) -> float:
    q = set(_simple_name_tokens(query_disease))
    c = set(_simple_name_tokens(context_name))
    if not c:
        return 0.0
    if normalize_name(query_disease) == normalize_name(context_name):
        return 1.0
    weights = disease_name_idf or {}
    overlap = q & c
    numerator = sum(float(weights.get(t, 1.0)) for t in overlap)
    denominator = sum(float(weights.get(t, 1.0)) for t in (q | c)) or 1.0
    return round(numerator / denominator, 6)


# Disease-neighbor reliability uses only generic token-IDF overlap.
# Disease/area-specific alias vocabularies and keyword subtype rules are absent.

# ============================================================
# 3. Ollama helper# ============================================================
# 3. Ollama helper
# ============================================================





# ============================================================
# 4. Disease-only KG context and gold construction
# ============================================================

def norm_text(x: Any) -> str:
    return str(x or "").strip().lower()


def get_disease_rag_row(
    disease: str,
    disease_df: pd.DataFrame,
) -> Optional[Dict[str, Any]]:
    """Return a disease-text row only when the name matches exactly.

    The target disease is never resolved by substring or fuzzy matching. Related
    disease information may enter later only through explicit KG relations or
    embedding-neighbor retrieval.
    """
    q = norm_text(disease)
    df = disease_df.copy()

    for col in ["mondo_name", "group_name_bert"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    if "mondo_name" in df.columns:
        hit = df[df["mondo_name"].map(norm_text) == q]
        if len(hit) > 0:
            return hit.iloc[0].to_dict()

    if "group_name_bert" in df.columns:
        hit = df[df["group_name_bert"].map(norm_text) == q]
        if len(hit) > 0:
            return hit.iloc[0].to_dict()

    return None

DISEASE_TEXT_FIELDS_SAFE = [
    "mondo_definition",
    "umls_description",
    "orphanet_definition",
    "orphanet_clinical_description",
    "mayo_symptoms",
    "mayo_causes",
    "mayo_risk_factors",
    "mayo_complications",
]

def clean_nan_text(x: Any) -> str:
    s = str(x or "").strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def build_safe_disease_rag_text(
    disease_row: Optional[Dict[str, Any]],
    gold_drugs: Optional[List[str]] = None,
    max_chars_per_field: int = 1200,
) -> Dict[str, str]:
    if not disease_row:
        return {}

    out = {}

    for field in DISEASE_TEXT_FIELDS_SAFE:
        text = clean_nan_text(disease_row.get(field))
        if not text:
            continue

        # 기존 코드에 있는 함수 사용
        text = mask_terms_in_text(text, gold_drugs or [])

        out[field] = text[:max_chars_per_field]

    return out

def resolve_exact_disease_node(driver, disease: str) -> Dict[str, Any]:
    """Resolve an exact disease concept as a virtual bundle of same-name nodes.

    Matching is case-insensitive full-name equality only. Substring, fuzzy, and
    alias-like matching are prohibited. If the KG contains multiple DISEASE
    nodes with the same normalized full name, all of them are treated as one
    virtual disease bundle so complementary genes, phenotypes, ontology edges,
    and evaluation labels are not discarded.

    ``element_id`` is a deterministic primary member used only for embedding
    cache storage. Retrieval and context construction use
    ``exact_match_element_ids`` as the full bundle.
    """
    disease = str(disease or "").strip()
    if not disease:
        raise ValueError("Disease name is empty.")

    q = """
    MATCH (d:DISEASE)
    WHERE toLower(trim(d.name)) = toLower(trim($disease))
    RETURN
      d.name AS disease_name,
      d.node_index AS node_index,
      d.node_id AS node_id,
      elementId(d) AS element_id
    ORDER BY elementId(d)
    """
    rows = run_query_auto(driver, q, {"disease": disease}, timeout_sec=60)
    if not rows:
        raise ValueError(
            f"Exact DISEASE node not found: {disease!r}. "
            "Use the exact Neo4j DISEASE.name value."
        )

    candidates = [
        {
            "name": row.get("disease_name") or disease,
            "node_index": row.get("node_index"),
            "node_id": row.get("node_id"),
            "element_id": row.get("element_id"),
        }
        for row in rows
        if row.get("element_id")
    ]
    if not candidates:
        raise ValueError(f"Exact DISEASE matches had no elementId values: {disease!r}")

    candidates.sort(key=lambda x: str(x.get("element_id") or ""))
    primary = candidates[0]
    bundle_ids = [x["element_id"] for x in candidates]
    is_bundle = len(bundle_ids) > 1

    return {
        "query_disease": disease,
        "disease_name": primary.get("name") or disease,
        "node_index": primary.get("node_index"),
        "node_id": primary.get("node_id"),
        "element_id": primary.get("element_id"),
        "exact_match_element_ids": bundle_ids,
        "bundle_element_ids": bundle_ids,
        "duplicate_exact_name_count": len(bundle_ids),
        "resolution_candidates": candidates,
        "resolution_method": (
            "case_insensitive_exact_name_virtual_bundle"
            if is_bundle
            else "case_insensitive_exact_name_single_node"
        ),
    }


def get_disease_context(
    driver,
    disease: str,
    disease_element_id: Optional[str] = None,
    disease_element_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build one merged context from the exact-name disease bundle.

    Same-name member nodes are unioned. Parent/child edges between bundle
    members are excluded from family context, because they represent duplicate
    record linkage rather than a biological hierarchy hop.
    """
    exact_ids = unique_keep_order(
        [x for x in (disease_element_ids or []) if x]
        + ([disease_element_id] if disease_element_id else [])
    )
    q = """
    MATCH (d:DISEASE)
    WHERE (size($disease_element_ids) > 0 AND elementId(d) IN $disease_element_ids)
       OR (size($disease_element_ids) = 0
           AND toLower(trim(d.name)) = toLower(trim($disease)))

    OPTIONAL MATCH (d)-[:associated_with]-(g:GENE)
    WITH d, collect(DISTINCT g.name) AS disease_genes

    OPTIONAL MATCH (d)-[:parent_child]-(fam:DISEASE)
    WHERE NOT elementId(fam) IN $disease_element_ids
      AND toLower(trim(coalesce(fam.name, ''))) <> toLower(trim($disease))
    WITH d, disease_genes, collect(DISTINCT fam.name) AS family_names

    OPTIONAL MATCH (d)-[:phenotype_present]-(ph:PHENO)
    RETURN
      d.name AS disease_name,
      d.node_index AS node_index,
      d.node_id AS node_id,
      elementId(d) AS element_id,
      disease_genes,
      family_names,
      collect(DISTINCT ph.name) AS phenotypes,
      coalesce(d.safe_context_text, '') AS safe_context_text,
      d.safe_context_source AS safe_context_source,
      d.safe_context_embedding_model AS safe_context_embedding_model
    ORDER BY elementId(d)
    """

    rows = run_query_auto(
        driver,
        q,
        {"disease": disease, "disease_element_ids": exact_ids},
        timeout_sec=180,
    )
    if not rows:
        raise ValueError(f"Expected at least one exact DISEASE node for {disease!r}.")

    matched_ids = unique_keep_order([r.get("element_id") for r in rows if r.get("element_id")])
    primary_row = rows[0]
    if disease_element_id:
        primary_row = next(
            (r for r in rows if r.get("element_id") == disease_element_id),
            primary_row,
        )

    safe_texts = unique_keep_order([
        str(r.get("safe_context_text") or "").strip()
        for r in rows
        if str(r.get("safe_context_text") or "").strip()
    ])
    safe_sources = unique_keep_order([
        str(r.get("safe_context_source") or "").strip()
        for r in rows
        if str(r.get("safe_context_source") or "").strip()
    ])

    return {
        "query_disease": disease,
        "matched_disease_names": unique_keep_order([
            r.get("disease_name") or disease for r in rows
        ]),
        "resolved_disease_name": primary_row.get("disease_name") or disease,
        "resolved_node_index": primary_row.get("node_index"),
        "resolved_node_id": primary_row.get("node_id"),
        "resolved_element_id": disease_element_id or primary_row.get("element_id"),
        "bundle_element_ids": matched_ids,
        "duplicate_exact_name_count": len(matched_ids),
        "resolution_method": (
            "case_insensitive_exact_name_virtual_bundle"
            if len(matched_ids) > 1
            else "case_insensitive_exact_name_single_node"
        ),
        "disease_genes": unique_keep_order([
            x for r in rows for x in (r.get("disease_genes") or []) if x
        ])[:80],
        "family_names": unique_keep_order([
            x for r in rows for x in (r.get("family_names") or []) if x
        ])[:30],
        "phenotypes": unique_keep_order([
            x for r in rows for x in (r.get("phenotypes") or []) if x
        ])[:50],
        "safe_context_text": "\n\n".join(safe_texts)[:8000],
        "safe_context_source": safe_sources,
    }

def get_kg_gold_drugs(
    driver,
    disease: str,
    positive_rels: Optional[List[str]] = None,
    disease_element_id: Optional[str] = None,
    disease_element_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    positive_rels = positive_rels or DIRECT_GOLD_RELS
    rel_union = "|".join(positive_rels)
    exact_ids = unique_keep_order(
        [x for x in (disease_element_ids or []) if x]
        + ([disease_element_id] if disease_element_id else [])
    )

    q = f"""
    MATCH (d:DISEASE)
    WHERE (size($disease_element_ids) > 0 AND elementId(d) IN $disease_element_ids)
       OR (size($disease_element_ids) = 0
           AND toLower(trim(d.name)) = toLower(trim($disease)))

    MATCH (dr:DRUG)-[r:{rel_union}]-(d)
    RETURN DISTINCT
      dr.name AS drug,
      type(r) AS relation,
      coalesce(r.Max_Phase, dr.Max_Phase) AS max_phase
    ORDER BY relation, drug
    """

    return run_query_auto(
        driver,
        q,
        {"disease": disease, "disease_element_ids": exact_ids},
        timeout_sec=180,
    )


def get_target_disease_redaction_drugs(
    driver,
    disease: str,
    disease_element_id: Optional[str] = None,
    disease_element_ids: Optional[List[str]] = None,
) -> List[str]:
    """Return direct disease-drug names for text redaction only.

    The returned names are used exclusively to replace treatment mentions in
    disease RAG text and prompts. They are not attached to candidates, priors,
    paths, or scores. All supported positive and negative clinical relation
    types are included so this is not an evaluation-label lookup.
    """
    rels = unique_keep_order(SUPPORTIVE_POSITIVE_RELS_DISCOVERY + NEGATIVE_DD_RELS)
    rows = get_kg_gold_drugs(
        driver,
        disease,
        positive_rels=rels,
        disease_element_id=disease_element_id,
        disease_element_ids=disease_element_ids,
    )
    return unique_keep_order([r.get("drug") for r in rows if r.get("drug")])


def build_gold_drugs(
    driver,
    case: Optional[Dict[str, Any]] = None,
    disease: Optional[str] = None,
    manual_gold_drugs: Optional[List[str]] = None,
    use_kg_gold: bool = True,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Compatible gold builder.

    Supports both:
      build_gold_drugs(driver, case=case)
    and old style:
      build_gold_drugs(driver, disease=disease, manual_gold_drugs=..., use_kg_gold=...)
    """

    if case is not None:
        disease = case["disease"]
        manual_gold_drugs = case.get("manual_gold_drugs") or []
        use_kg_gold = bool(case.get("use_kg_gold", True))
        positive_rels = case.get("positive_rels") or DIRECT_GOLD_RELS
        disease_element_id = case.get("_disease_element_id")
        disease_element_ids = case.get("_disease_exact_element_ids") or []
    else:
        if disease is None:
            raise ValueError("Either case or disease must be provided.")
        manual_gold_drugs = manual_gold_drugs or []
        positive_rels = DIRECT_GOLD_RELS
        disease_element_id = None
        disease_element_ids = []

    gold_records = []

    # 1) Manual curated gold
    for drug in manual_gold_drugs:
        gold_records.append({
            "drug": drug,
            "source": "manual_curated",
            "relation": "manual_positive",
            "max_phase": None,
        })

    # 2) KG positive gold: indication / off_label_use only
    if use_kg_gold:
        for r in get_kg_gold_drugs(
            driver,
            disease,
            positive_rels=positive_rels,
            disease_element_id=disease_element_id,
            disease_element_ids=disease_element_ids,
        ):
            gold_records.append({
                "drug": r.get("drug"),
                "source": "kg_positive",
                "relation": r.get("relation"),
                "max_phase": r.get("max_phase"),
            })

    gold_drugs = unique_keep_order([
        r["drug"] for r in gold_records if r.get("drug")
    ])

    return gold_drugs, gold_records


# ============================================================
# 5. Target-context-first candidate generation
# ============================================================






def _fallback_context_expansion(disease: str, disease_context: Dict[str, Any], genes: List[str]) -> Tuple[List[str], List[str]]:
    """No disease-name-to-mechanism expansion is performed."""
    return [], []



def _rank_fallback_genes(disease: str, disease_context: Dict[str, Any], genes: List[str]) -> List[str]:
    """Preserve KG-provided gene order without gene-name priors."""
    return unique_keep_order([normalize_gene(g) for g in (genes or []) if normalize_gene(g)])













def build_target_context_anchors(
    disease: str,
    disease_context: Dict[str, Any],
    max_genes: int = 40,
    max_terms: int = 60,
    case_area: str = "",
    disease_name_idf: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Deterministic target-disease KG/RAG context with no disease-specific target rules."""
    genes = unique_keep_order([
        normalize_mechanism_gene(g)
        for g in (disease_context.get("disease_genes") or [])
        if str(g or "").strip()
    ])[:max_genes]
    phenotypes = unique_keep_order(disease_context.get("phenotypes") or [])[:max_terms]
    families = unique_keep_order([disease] + (disease_context.get("family_names") or []))[:30]
    # PHENO names are kept as PHENO context and are not passed into BP/PATH string constraints.
    axes = [{
        "axis_name": "disease-only KG context",
        "axis_rationale": "Deterministic target-disease context using only retrieved disease genes and phenotype context.",
        "priority": "high" if genes else "medium",
        "anchor_genes": genes,
        "anchor_biology_terms": [],
        "anchor_pathways": [],
        "anchor_phenotypes": phenotypes,
        "relevant_cell_types_or_tissues": [],
        "expected_drug_actions_or_classes": [],
        "anchor_source": "target_disease_context",
    }]
    return {
        "disease": disease,
        "mechanism_axes": axes,
        "global_anchors": {
            "target_genes": genes,
            "biology_terms": [],
            "pathways": [],
            "phenotype_terms": phenotypes,
            "disease_family_names": families,
        },
        "raw_text": "[TARGET_DISEASE_CONTEXT] disease-only KG/RAG anchors",
        "anchor_source": "target_disease_context",
            }



def priority_weight(priority: str) -> float:
    p = str(priority or "").lower()
    if p == "high":
        return 1.0
    if p == "medium":
        return 0.5
    if p == "low":
        return 0.2
    return 0.3


def normalize_terms_for_cypher(terms: List[str]) -> List[str]:
    out = []
    for t in terms or []:
        s = normalize_name(t)
        if len(s) >= 4 and s not in GENERIC_BIO_TERMS:
            out.append(s)
    return unique_keep_order(out)


def extract_gene_mentions(items):
    """
    Target-context anchor_genes에서 retrieval용 gene symbol만 추출.
    
    Supports strings, structured gene dictionaries, and serialized dictionaries.
    """
    out = []

    for item in items or []:
        # Case 1: string
        if isinstance(item, str):
            s = item.strip()

            # dict가 문자열로 들어온 경우
            if s.startswith("{") and s.endswith("}"):
                try:
                    parsed = ast.literal_eval(s)
                    if isinstance(parsed, dict):
                        item = parsed
                    else:
                        out.append(s)
                        continue
                except Exception:
                    out.append(s)
                    continue
            else:
                out.append(s)
                continue

        # Case 2: dict
        if isinstance(item, dict):
            confidence = str(
                item.get("confidence")
                or item.get("CONFIDENCE")
                or "medium"
            ).lower()

            # low confidence는 retrieval anchor에서 제외
            if confidence == "low":
                continue

            symbol = (
                item.get("symbol")
                or item.get("SYMBOL")
                or item.get("gene_symbol")
                or item.get("GENE_SYMBOL")
            )

            if symbol:
                out.append(symbol)

    return unique_keep_order(
        [normalize_mechanism_gene(x) for x in out if str(x).strip()]
    )
    

def extract_gene_symbols(items):
    out = []

    for item in items or []:
        if isinstance(item, str):
            s = item.strip()

            if s.startswith("{") and s.endswith("}"):
                try:
                    parsed = ast.literal_eval(s)
                    if isinstance(parsed, dict):
                        item = parsed
                    else:
                        out.append(s)
                        continue
                except Exception:
                    out.append(s)
                    continue
            else:
                out.append(s)
                continue

        if isinstance(item, dict):
            symbol = (
                item.get("symbol")
                or item.get("SYMBOL")
                or item.get("gene_symbol")
                or item.get("GENE_SYMBOL")
            )
            if symbol:
                out.append(symbol)

    return unique_keep_order([normalize_mechanism_gene(x) for x in out if x])
    
def _axis_anchor_signature(axis: Dict[str, Any]) -> set:
    """Normalized gene/BP/PATH signature used only for deterministic diversity selection."""
    genes = {normalize_gene(x) for x in extract_gene_mentions(axis.get("anchor_genes") or []) if normalize_gene(x)}
    terms = {
        normalize_name(x)
        for x in ((axis.get("anchor_biology_terms") or []) + (axis.get("anchor_pathways") or []))
        if normalize_name(x) and normalize_name(x) not in GENERIC_BIO_TERMS
    }
    return {f"g:{x}" for x in genes} | {f"t:{x}" for x in terms}


def _axis_overlap(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    sa, sb = _axis_anchor_signature(a), _axis_anchor_signature(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(len(sa | sb), 1)


def _axis_raw_anchor_keys(axis: Dict[str, Any]) -> List[str]:
    """Return unique raw target-context anchors, counting each raw concept at most once.

    Gene anchors and BP/PATH text anchors are kept in separate namespaces so
    alias or cross-label KG resolution cannot inflate the denominator or
    numerator. This function is disease-, drug-, and label-agnostic.
    """
    keys: List[str] = []
    seen = set()
    for gene in extract_gene_mentions(axis.get("anchor_genes") or []):
        norm = normalize_mechanism_gene(gene)
        key = f"g:{norm}" if norm else ""
        if key and key not in seen:
            seen.add(key)
            keys.append(key)
    for term in (axis.get("anchor_biology_terms") or []) + (axis.get("anchor_pathways") or []):
        norm = normalize_name(term)
        key = f"t:{norm}" if norm else ""
        if key and key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def resolve_mechanism_axes_against_kg(
    driver,
    mechanism_anchor_obj: Dict[str, Any],
    timeout_sec: int = 20,
) -> Dict[str, Any]:
    """Resolve target-context GENE/BP/PATH anchors by exact KG node names.

    Raw-anchor resolution and KG expansion are tracked separately. A raw text
    term can resolve to BP, PATH, or both, but it contributes at most one unit
    to the raw resolution fraction. Therefore the fraction is always in [0, 1].
    """
    obj = copy.deepcopy(mechanism_anchor_obj or {})
    axes = obj.get("mechanism_axes", []) or []
    gene_names = unique_keep_order(
        normalize_mechanism_gene(g)
        for axis in axes
        for g in extract_gene_mentions(axis.get("anchor_genes") or [])
        if normalize_mechanism_gene(g)
    )
    term_names = unique_keep_order(
        normalize_name(term)
        for axis in axes
        for term in ((axis.get("anchor_biology_terms") or []) + (axis.get("anchor_pathways") or []))
        if normalize_name(term)
    )
    if not gene_names and not term_names:
        obj["anchor_kg_resolution"] = {
            "status": "no_anchors",
            "uses_exact_name_matching": True,
            "n_axes": len(axes),
        }
        return obj

    query = """
    MATCH (n)
    WHERE n.name IS NOT NULL
      AND (
        (n:GENE AND toUpper(trim(n.name)) IN $gene_names)
        OR
        ((n:BP OR n:PATH) AND toLower(trim(n.name)) IN $term_names)
      )
    RETURN n.name AS name, labels(n) AS labels
    """
    try:
        rows = run_query_auto(
            driver,
            query,
            {
                "gene_names": [str(x).upper() for x in gene_names],
                "term_names": [normalize_name(x) for x in term_names],
            },
            timeout_sec=timeout_sec,
        )
    except Exception as exc:
        obj["anchor_kg_resolution"] = {
            "status": "lookup_failed_original_anchors_retained",
            "uses_exact_name_matching": True,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "n_axes": len(axes),
        }
        return obj

    resolved_genes: Dict[str, str] = {}
    resolved_bp: Dict[str, str] = {}
    resolved_path: Dict[str, str] = {}
    for row in rows or []:
        name = str(row.get("name") or "").strip()
        labels = {str(x) for x in (row.get("labels") or [])}
        if not name:
            continue
        if "GENE" in labels:
            resolved_genes[normalize_mechanism_gene(name)] = name
        if "BP" in labels:
            resolved_bp[normalize_name(name)] = name
        if "PATH" in labels:
            resolved_path[normalize_name(name)] = name

    resolved_axes = []
    for raw_axis in axes:
        axis = copy.deepcopy(raw_axis)
        raw_genes = unique_keep_order(
            normalize_mechanism_gene(g)
            for g in extract_gene_mentions(axis.get("anchor_genes") or [])
            if normalize_mechanism_gene(g)
        )
        raw_bio = unique_keep_order(axis.get("anchor_biology_terms") or [])
        raw_paths = unique_keep_order(axis.get("anchor_pathways") or [])
        axis["raw_anchor_genes"] = raw_genes
        axis["raw_anchor_biology_terms"] = raw_bio
        axis["raw_anchor_pathways"] = raw_paths

        axis_genes = unique_keep_order(
            resolved_genes[g] for g in raw_genes if g in resolved_genes
        )
        axis_bp = unique_keep_order(
            [resolved_bp[normalize_name(x)] for x in raw_bio if normalize_name(x) in resolved_bp]
            + [resolved_bp[normalize_name(x)] for x in raw_paths if normalize_name(x) in resolved_bp]
        )
        axis_paths = unique_keep_order(
            [resolved_path[normalize_name(x)] for x in raw_bio if normalize_name(x) in resolved_path]
            + [resolved_path[normalize_name(x)] for x in raw_paths if normalize_name(x) in resolved_path]
        )
        axis["anchor_genes"] = axis_genes
        axis["anchor_biology_terms"] = axis_bp
        axis["anchor_pathways"] = axis_paths

        raw_keys = _axis_raw_anchor_keys(raw_axis)
        resolved_raw_keys = set()
        for gene in raw_genes:
            if gene in resolved_genes:
                resolved_raw_keys.add(f"g:{gene}")
        for term in raw_bio + raw_paths:
            norm = normalize_name(term)
            if norm in resolved_bp or norm in resolved_path:
                resolved_raw_keys.add(f"t:{norm}")

        raw_count = len(raw_keys)
        resolved_raw_count = len(resolved_raw_keys.intersection(raw_keys))
        expanded_count = len(axis_genes) + len(axis_bp) + len(axis_paths)
        fraction = 0.0 if raw_count == 0 else resolved_raw_count / raw_count

        axis["kg_anchor_resolution_available"] = True
        axis["kg_raw_anchor_count"] = raw_count
        axis["kg_resolved_raw_anchor_count"] = resolved_raw_count
        # Backward-compatible field now has raw-anchor semantics.
        axis["kg_resolved_anchor_count"] = resolved_raw_count
        axis["kg_expanded_anchor_count"] = expanded_count
        axis["kg_anchor_resolution_fraction"] = round(max(0.0, min(1.0, fraction)), 6)
        axis["kg_unresolved_anchor_count"] = max(raw_count - resolved_raw_count, 0)
        resolved_axes.append(axis)

    obj["mechanism_axes"] = resolved_axes
    obj["anchor_kg_resolution"] = {
        "status": "ok",
        "uses_exact_name_matching": True,
        "raw_anchor_resolution_semantics": "unique_raw_anchor_count; alias/cross-label expansion excluded",
        "n_axes": len(resolved_axes),
        "n_raw_genes": len(gene_names),
        "n_resolved_genes": len(resolved_genes),
        "n_raw_terms": len(term_names),
        "n_resolved_bp_terms": len(resolved_bp),
        "n_resolved_path_terms": len(resolved_path),
        "n_axes_with_any_resolved_anchor": sum(
            1 for axis in resolved_axes if int(axis.get("kg_resolved_raw_anchor_count") or 0) > 0
        ),
        "max_axis_resolution_fraction": max(
            [safe_float(axis.get("kg_anchor_resolution_fraction"), 0.0) for axis in resolved_axes]
            or [0.0]
        ),
    }
    return obj


def _axis_selection_specificity(axis: Dict[str, Any]) -> float:
    """Disease-agnostic specificity proxy for deterministic axis ordering."""
    explicit = axis.get("anchor_specificity")
    if explicit is not None:
        return max(0.0, safe_float(explicit, 0.0))
    raw_n = max(1, int(axis.get("kg_raw_anchor_count") or len(_axis_raw_anchor_keys(axis)) or 1))
    specific_n = len(_axis_anchor_signature(axis))
    return max(0.0, min(1.0, specific_n / raw_n))


def select_mechanism_axes(
    mechanism_axes: List[Dict[str, Any]],
    max_axes: int = 3,
    redundancy_threshold: float = 0.80,
) -> List[Dict[str, Any]]:
    """Select a reproducible target-context axis set without disease-specific rules.

    Declared scientific priority is evaluated first. Within each priority tier,
    raw-anchor KG resolution and generic specificity are used, followed by the
    original target-context order. Alias/KG expansion count is never a ranking feature.
    Near-duplicate axes are deferred when a distinct alternative exists.
    """
    priority_rank = {"high": 0, "medium": 1, "low": 2}
    prepared: List[Dict[str, Any]] = []
    for idx, raw in enumerate(mechanism_axes or []):
        axis = copy.deepcopy(raw)
        genes = extract_gene_mentions(axis.get("anchor_genes") or [])
        bio = unique_keep_order(axis.get("anchor_biology_terms") or [])
        paths = unique_keep_order(axis.get("anchor_pathways") or [])
        if not genes and not bio and not paths:
            continue
        axis["_original_axis_index"] = idx
        axis["_anchor_signature_size"] = len(_axis_anchor_signature(axis))
        axis["_kg_resolved_raw_anchor_count"] = int(
            axis.get("kg_resolved_raw_anchor_count")
            if axis.get("kg_anchor_resolution_available")
            else axis.get("_anchor_signature_size") or 0
        )
        axis["_anchor_specificity"] = _axis_selection_specificity(axis)
        prepared.append(axis)

    prepared.sort(
        key=lambda axis: (
            priority_rank.get(str(axis.get("priority") or "medium").lower(), 3),
            -int(axis.get("_kg_resolved_raw_anchor_count") or 0),
            -float(axis.get("_anchor_specificity") or 0.0),
            int(axis.get("_original_axis_index") or 0),
            normalize_name(axis.get("axis_name")),
        )
    )

    selected: List[Dict[str, Any]] = []
    deferred: List[Dict[str, Any]] = []
    budget = max(int(max_axes or 0), 0)
    for axis in prepared:
        overlap = max((_axis_overlap(axis, prior) for prior in selected), default=0.0)
        axis["max_overlap_with_selected_axes"] = round(overlap, 6)
        if selected and overlap >= redundancy_threshold:
            axis["axis_selection_reason"] = "deferred_redundant_axis"
            deferred.append(axis)
            continue
        axis["axis_selection_reason"] = "selected_priority_raw_resolution_specificity"
        selected.append(axis)
        if len(selected) >= budget:
            break

    if len(selected) < budget:
        for axis in deferred:
            axis["axis_selection_reason"] = "selected_after_diversity_defer"
            selected.append(axis)
            if len(selected) >= budget:
                break

    for rank, axis in enumerate(selected, start=1):
        axis["selected_for_initial_graph"] = True
        axis["axis_selection_rank"] = rank
        axis["axis_selection_policy"] = "priority_raw_resolution_specificity_diversity"
        if (
            normalize_name(axis.get("priority")) == "high"
            and int(axis.get("_kg_resolved_raw_anchor_count") or 0) == 0
        ):
            axis["axis_selection_diagnostic"] = "unresolved_high_priority"
    return selected

def axis_anchor_bundle(
    mechanism_anchor_obj: Dict[str, Any],
    max_axes: int = 3,
    use_mechanism_lookup: bool = True,
    preselected_axes: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Convert selected mechanism axes into GraphRAG anchor bundles.

    Only GENE and BP/PATH anchors are passed to the core graph queries. Tissue/cell
    descriptions and expected action classes remain available to the drug embedding
    representation but are not treated as BP/PATH node names.
    """
    bundles = []
    axes = preselected_axes
    if axes is None:
        axes = select_mechanism_axes(
            mechanism_anchor_obj.get("mechanism_axes", []) or [],
            max_axes=max_axes,
        )

    object_anchor_source = mechanism_anchor_obj.get("anchor_source")

    for axis in axes or []:
        raw_gene_mentions = extract_gene_mentions(axis.get("anchor_genes") or [])
        axis_genes = unique_keep_order(
            [normalize_mechanism_gene(g) for g in raw_gene_mentions if g]
        )
        bio_path_terms = unique_keep_order(
            (axis.get("anchor_biology_terms") or [])
            + (axis.get("anchor_pathways") or [])
        )
        expansion_terms = unique_keep_order(
            (axis.get("anchor_biology_terms") or [])
            + (axis.get("anchor_pathways") or [])
            + [axis.get("axis_name") or ""]
        )
        expanded_genes = expand_mechanism_terms_to_genes(
            expansion_terms,
            enabled=use_mechanism_lookup,
        )
        anchor_genes = unique_keep_order(
            axis_genes
            + [normalize_mechanism_gene(g) for g in expanded_genes if g]
        )
        bundles.append({
            "axis_name": axis.get("axis_name") or "",
            "priority": axis.get("priority") or "medium",
            "axis_rationale": axis.get("axis_rationale") or "",
            "anchor_source": axis.get("anchor_source") or object_anchor_source or "target_disease_context",
            "anchor_genes": anchor_genes,
            "raw_anchor_genes": axis_genes,
            "expanded_genes": [normalize_mechanism_gene(g) for g in expanded_genes if g],
            "anchor_terms": normalize_terms_for_cypher(bio_path_terms),
            "anchor_bio_path_terms": normalize_terms_for_cypher(bio_path_terms),
            "anchor_phenotypes": unique_keep_order(axis.get("anchor_phenotypes") or []),
            "near_diseases": unique_keep_order(axis.get("near_diseases") or []),
            "axis_selection_rank": axis.get("axis_selection_rank"),
            "axis_selection_policy": axis.get("axis_selection_policy"),
        })
    return bundles[:max_axes]


def vector_anchor_bundle(
    mechanism_anchor_obj: Dict[str, Any],
    use_mechanism_lookup: bool = False,
) -> List[Dict[str, Any]]:
    """Return the explicit branch-1-2 bundle without mixing it with target-context axes."""
    vector_obj = build_vector_anchor_mechanism_obj(mechanism_anchor_obj)
    return axis_anchor_bundle(
        vector_obj,
        max_axes=1,
        use_mechanism_lookup=use_mechanism_lookup,
        preselected_axes=vector_obj.get("mechanism_axes") or [],
    )

def is_target_specific_bridge_axis(axis: Dict[str, Any]) -> bool:
    """Gate broad bridge motifs using supplied anchor structure only.

    No disease-name or mechanism-keyword dictionary is consulted. A bridge motif
    is eligible only when the target context supplies both a concrete gene anchor and
    a BP/PATH or other biological term.
    """
    genes = axis.get("anchor_genes", []) or []
    terms = (axis.get("anchor_bio_path_terms", []) or []) + (axis.get("anchor_terms", []) or [])
    return bool(genes) and bool(terms)


def should_run_anchor_pattern(
    pattern_name: str,
    axis: Dict[str, Any],
    anchor_genes: List[str],
    anchor_terms: List[str],
    anchor_bio_path_terms: List[str],
    anchor_phenotypes: Optional[List[str]] = None,
    ablation_mode: str = ABLATION_MODE,
) -> bool:
    """Pattern-specific, disease-name-agnostic gating."""
    anchor_phenotypes = anchor_phenotypes or []
    if pattern_name == "drug_target_to_anchor_disease_gene_via_bio":
        if not anchor_genes or ablation_mode == "off":
            return False
        return is_target_specific_bridge_axis(axis) if ablation_mode == "conditional" else True
    if pattern_name == "drug_target_ppi_to_anchor_gene":
        if not anchor_genes or ablation_mode == "off":
            return False
        return is_target_specific_bridge_axis(axis) if ablation_mode == "conditional" else True
    if pattern_name == "drug_to_disease_direct_gene_action":
        return True
    if pattern_name in {
        "drug_to_anchor_gene",
        "drug_to_disease_anchor_gene",
        "drug_to_disease_via_anchor_gene_path",
        "drug_anchor_target_to_disease_via_bio",
    }:
        return bool(anchor_genes)
    if pattern_name in {"drug_to_anchor_bioprocess", "drug_to_disease_via_anchor_bio"}:
        return bool(anchor_terms or anchor_bio_path_terms)
    if pattern_name == "drug_to_disease_via_vector_phenotype_gene":
        return bool(anchor_phenotypes)
    return True

# ============================================================
# 6. Candidate-specific GraphRAG verification
# ============================================================

DRUG_GENE_ACTION_TYPE_WEIGHTS = {
    # Explicit Action_Type values should never score lower than the relation-only
    # fallback for the same relation family. These are DRUG->GENE pharmacology
    # edge refinements, not disease-drug clinical priors.
    "inhibitor": 3.05,
    "blocker": 3.05,
    "antagonist": 2.95,
    "inverse agonist": 2.90,
    "degrader": 3.10,
    "negative allosteric modulator": 2.85,
    "agonist": 3.05,
    "activator": 3.05,
    "opener": 2.90,
    "partial agonist": 2.85,
    "releasing agent": 2.75,
    "positive allosteric modulator": 2.90,
    "positive modulator": 2.75,
    "modulator": 2.60,
    "stabiliser": 2.65,
    "stabilizer": 2.65,
    "binding agent": 2.05,
    "substrate": 1.75,
}

DRUG_GENE_ACTION_RELATION_FALLBACK_WEIGHTS = {
    # Conservative relation-only fallbacks used when Action_Type is missing.
    # Kept below or equal to explicit Action_Type scores to avoid the fallback>
    # specific-evidence inversion.
    "inhibition": 2.75,
    "activation": 2.75,
    "modulation": 2.40,
    "binding": 1.60,
    "target": 1.40,
    "enzyme": 0.75,
    "transporter": 0.70,
    "carrier": 0.60,
}

DRUG_GENE_DIRECTIONAL_RELS = {"inhibition", "activation", "modulation", "binding"}
DRUG_GENE_ADME_RELS = {"enzyme", "transporter", "carrier"}


def normalize_action_type(x: Any) -> str:
    # KG values appear both as `Negative_allosteric_modulator` and natural text.
    return re.sub(r"\s+", " ", str(x or "").strip().lower().replace("_", " "))


def normalize_relation_set(x: Any) -> set:
    vals: List[str] = []
    if isinstance(x, str):
        vals = re.split(r"[|;,]", x)
    elif isinstance(x, (list, tuple, set)):
        vals = list(x)
    elif x is not None:
        vals = [x]
    return {normalize_name(v) for v in vals if str(v or "").strip()}


def _action_type_candidates(action_type: Any, action_types: Any = None) -> List[str]:
    vals: List[str] = []
    vals.extend([normalize_action_type(v) for v in flatten_values(action_type)])
    vals.extend([normalize_action_type(v) for v in flatten_values(action_types)])
    return [v for v in vals if v]


DRUG_GENE_ACTION_TYPE_COMPATIBILITY = {
    "inhibition": {
        "antagonist", "blocker", "degrader", "inhibitor", "inverse agonist",
        "negative allosteric modulator",
    },
    "activation": {
        "activator", "agonist", "opener", "partial agonist",
        "positive allosteric modulator", "positive modulator", "releasing agent",
    },
    "modulation": {"modulator", "stabiliser", "stabilizer"},
    "binding": {"binding agent", "substrate"},
}


def _compatible_action_type_candidates(
    rel: Any,
    action_type: Any = None,
    action_types: Any = None,
) -> List[str]:
    rel_norm = normalize_name(rel)
    candidates = _action_type_candidates(action_type, action_types)
    if rel_norm == "target":
        # A broad target annotation may accompany any explicit pharmacological
        # action type on the same drug-gene pair.
        return candidates
    allowed = DRUG_GENE_ACTION_TYPE_COMPATIBILITY.get(rel_norm)
    if not allowed:
        return []
    return [value for value in candidates if value in allowed]


def _drug_gene_action_base_score(
    rel: Any,
    action_type: Any = None,
    action_types: Any = None,
) -> float:
    rel = normalize_name(rel)
    explicit_scores = [
        float(DRUG_GENE_ACTION_TYPE_WEIGHTS[value])
        for value in _compatible_action_type_candidates(rel, action_type, action_types)
        if value in DRUG_GENE_ACTION_TYPE_WEIGHTS
    ]
    if explicit_scores:
        return max(explicit_scores)
    return float(DRUG_GENE_ACTION_RELATION_FALLBACK_WEIGHTS.get(rel, 0.0))


def first_not_none(*values: Any) -> Any:
    """Return the first non-None value; unlike `or`, preserves numeric zero."""
    for value in values:
        if value is not None:
            return value
    return None


def canonical_drug_gene_relation(row: Dict[str, Any]) -> str:
    relset = normalize_relation_set(row.get("drug_gene_relset"))
    if row.get("first_rel"):
        relset.add(normalize_name(row.get("first_rel")))
    if not relset:
        return normalize_name(row.get("first_rel"))
    priority = {"inhibition": 8, "activation": 8, "modulation": 7, "target": 6, "binding": 5, "enzyme": 3, "transporter": 2, "carrier": 2}
    return max(relset, key=lambda r: (action_score_candidate(r, relation_set=relset), priority.get(r, 0), r))


def canonical_pattern_family(pattern: Any, row: Optional[Dict[str, Any]] = None) -> str:
    p = str(pattern or "")
    bridge_type = str((row or {}).get("anchor_type") or (row or {}).get("bridge_node_type") or "")
    if bridge_type in {"BP", "PATH"}:
        return f"{p}:BIOLOGICAL_BRIDGE"
    return p


def canonical_path_key(row: Dict[str, Any], pattern: Any = None) -> Tuple[Any, ...]:
    nodes = tuple(normalize_name(x) for x in (row.get("path_nodes") or []) if str(x or "").strip())
    return (
        normalize_name(row.get("drug") or row.get("candidate_drug")),
        normalize_gene(row.get("target_gene")),
        normalize_gene(row.get("disease_gene")),
        normalize_name(row.get("anchor_name") or row.get("bridge_node_name")),
        canonical_pattern_family(pattern or row.get("pattern"), row),
        nodes,
    )


def canonicalize_path_rows(rows: List[Dict[str, Any]], pattern: Any = None) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for raw in rows or []:
        row = dict(raw)
        key = canonical_path_key(row, pattern)
        first_rel = normalize_name(row.get("first_rel"))
        relset = normalize_relation_set(row.get("drug_gene_relset"))
        if first_rel in set(DRUG_GENE_RELS):
            relset.add(first_rel)
        clinical_relset = normalize_relation_set(row.get("disease_drug_relset"))
        if first_rel in {
            "indication",
            "off_label_use",
            "tested_indication",
            "studied_for_treatment_of",
            "studied_for_marker_mechanism_of",
            "contraindication",
        }:
            clinical_relset.add(first_rel)
        action_types = unique_keep_order(flatten_values(row.get("drug_gene_action_types")) + flatten_values(row.get("action_type")))
        if key not in grouped:
            row["drug_gene_relset"] = sorted(relset)
            row["disease_drug_relset"] = sorted(clinical_relset)
            row["drug_gene_action_types"] = action_types
            grouped[key] = row
            continue
        cur = grouped[key]
        merged_relset = normalize_relation_set(cur.get("drug_gene_relset")) | relset
        cur["drug_gene_relset"] = sorted(merged_relset)
        cur["drug_gene_action_types"] = unique_keep_order(flatten_values(cur.get("drug_gene_action_types")) + action_types)
        # path_rels is an ordered edge sequence aligned to path_nodes. Never
        # union relation sequences position-blindly: target+inhibition rows for
        # the same topology would otherwise move the second drug-gene relation
        # into a later disease-hierarchy edge position. Keep one representative
        # sequence and rewrite only its first edge after relation canonicalization.
        cur["tested_indication_phases"] = flatten_values(cur.get("tested_indication_phases")) + flatten_values(row.get("tested_indication_phases"))
        cur["disease_drug_relset"] = sorted(normalize_relation_set(cur.get("disease_drug_relset")) | normalize_relation_set(row.get("disease_drug_relset")))
    out = []
    clinical_priority = {
        "indication": 5,
        "off_label_use": 4,
        "tested_indication": 3,
        "studied_for_treatment_of": 2,
        "studied_for_marker_mechanism_of": 1,
        "contraindication": 0,
    }
    for row in grouped.values():
        if row.get("drug_gene_relset"):
            best = max(
                row["drug_gene_relset"],
                key=lambda r: action_score_candidate(
                    r,
                    relation_set=row["drug_gene_relset"],
                    action_types=row.get("drug_gene_action_types"),
                ),
            )
            row["first_rel"] = best
            path_rels = list(row.get("path_rels") or [])
            if path_rels:
                path_rels[0] = best
                row["path_rels"] = path_rels
        elif row.get("disease_drug_relset"):
            relset = normalize_relation_set(row.get("disease_drug_relset"))
            positive = [r for r in relset if r in DISEASE_DRUG_POSITIVE_RELS]
            choices = positive or (["contraindication"] if "contraindication" in relset else [])
            if choices:
                row["first_rel"] = max(
                    choices,
                    key=lambda r: (clinical_priority.get(r, -1), r),
                )
                path_rels = list(row.get("path_rels") or [])
                if path_rels:
                    path_rels[0] = row["first_rel"]
                    row["path_rels"] = path_rels
            if "tested_indication" in relset:
                row["relation_max_phase"] = max_numeric_value(
                    row.get("tested_indication_phases"),
                    default=0.0,
                )
        out.append(row)
    return out


def structural_degree_penalty(row: Dict[str, Any]) -> float:
    if not ENABLE_STRUCTURAL_SCORING:
        return 0.0

    # Baseline structural fields that were already part of the frozen scorer.
    fields = [
        "target_gene_disease_degree",
        "target_gene_ppi_degree",
        "family_child_degree",
    ]

    # Newly activated topology fields are an independent experimental component.
    # They are disabled in the default evidence-hierarchy-only condition so that
    # drug/bridge degree effects cannot confound the BP-bridge correction test.
    if ENABLE_TOPOLOGY_SPECIFICITY:
        fields.extend([
            "drug_gene_degree",
            "disease_gene_degree",
            "bridge_gene_degree",
            "phenotype_disease_degree",
        ])

    penalty = 0.0
    for field in fields:
        degree = max(0.0, safe_float(row.get(field), 0.0))
        if degree > 0:
            penalty += min(math.log1p(degree) / 12.0, 0.7)
    return round(min(penalty, 1.5), 6)


# Family transfer is scored continuously. It is never accepted or rejected by a
# single fixed degree cutoff. Broader ontology parents require more independent
# graph support before family-only evidence can verify a candidate.
FAMILY_HIERARCHY_MIN_SPECIFICITY = 0.20  # retained for backward-compatible reporting only
FAMILY_HIERARCHY_UNKNOWN_SPECIFICITY = 0.0
FAMILY_HIERARCHY_SUPPORT_BASE = 0.40
FAMILY_HIERARCHY_MAX_REQUIRED_SUPPORTS = 4
FAMILY_EVIDENCE_PATTERNS = {"family_label_edge", "sibling_gene_action", "family_clinical_prior"}

# Candidate-floor rescue is a topology-only safety net. It is controlled by the
# same sparse-rescue ablation flag and therefore disappears in no_sparse_area_rescue.
GLOBAL_CANDIDATE_FLOOR = 20
GLOBAL_CANDIDATE_FLOOR_POLICY: Dict[str, Any] = {
    "enabled": True,
    "enabled_patterns": [
        "direct_gene_action",
        "ppi_gene_action",
        "sibling_gene_action",
        "family_clinical_prior",
    ],
    "dense_allowed_patterns": [
        "direct_gene_action",
        "ppi_gene_action",
        "sibling_gene_action",
        "family_clinical_prior",
    ],
    "limits": {
        "direct_gene_action": 120,
        "ppi_gene_action": 160,
        "sibling_gene_action": 180,
        "family_clinical_prior": 120,
    },
    "timeouts": {
        "direct_gene_action": 20,
        "ppi_gene_action": 25,
        "sibling_gene_action": 25,
        "family_clinical_prior": 20,
    },
    "tier_insert_after": {"strong": 10, "medium": 60, "weak": 90},
    "tier_max": {"strong": 3, "medium": 5, "weak": 10},
}

PATTERN_TIMEOUT_RETRY_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "patterns": [
        "drug_to_disease_via_anchor_gene_path",
        "drug_target_to_anchor_disease_gene_via_bio",
        "drug_to_disease_via_anchor_bio",
    ],
    "anchor_gene_limit": 6,
    "anchor_term_limit": 24,
    "result_limit": 12,
    "timeout_sec": 18,
}


def family_hierarchy_specificity(row: Dict[str, Any]) -> float:
    """Return a topology-only specificity score for a disease-family parent.

    The score depends only on the number of DISEASE children attached to the
    parent node. It contains no disease names, drug names, gene allowlists, or
    outcome-derived tuning. Broad ontology roots receive lower specificity.
    """
    degree = safe_float(row.get("family_child_degree"), -1.0)
    if degree < 0:
        return FAMILY_HIERARCHY_UNKNOWN_SPECIFICITY
    return round(1.0 / math.log2(2.0 + max(0.0, degree)), 6)


def family_required_independent_supports(row: Dict[str, Any]) -> int:
    """Return the evidence multiplicity required for family-only verification.

    The requirement increases smoothly as the ontology parent becomes broader.
    No disease names, drug names, gold labels, or outcome-derived thresholds are
    consulted. A narrow parent can verify with one path; broad roots require up to
    four independent sibling/target supports.
    """
    specificity = family_hierarchy_specificity(row)
    row["family_hierarchy_specificity"] = specificity
    effective = max(float(specificity), 0.10)
    required = int(math.ceil(float(FAMILY_HIERARCHY_SUPPORT_BASE) / effective))
    return max(1, min(int(FAMILY_HIERARCHY_MAX_REQUIRED_SUPPORTS), required))


def passes_family_hierarchy_specificity(row: Dict[str, Any]) -> bool:
    """Backward-compatible helper: whether one family path is sufficient alone."""
    return family_required_independent_supports(row) <= 1


FAMILY_PRIMARY_CLINICAL_RELS = {"indication", "off_label_use"}
FAMILY_TESTED_INDICATION_MIN_PHASE = 1.0


def _qualified_tested_indication(row: Dict[str, Any]) -> bool:
    relset = normalize_relation_set(row.get("disease_drug_relset"))
    rel = normalize_name(row.get("first_rel"))
    if rel:
        relset.add(rel)
    if "tested_indication" not in relset:
        return False
    phase = first_not_none(row.get("relation_max_phase"), row.get("tested_indication_phases"))
    return max_numeric_value(phase, default=0.0) >= float(FAMILY_TESTED_INDICATION_MIN_PHASE)


def is_primary_family_clinical_evidence(row: Dict[str, Any]) -> bool:
    """Whether a family clinical edge can count as independent support.

    Indication and off-label-use edges count directly. A tested-indication edge
    counts only when its recorded maximum phase reaches the fixed phase threshold.
    `studied_for_treatment_of` is retained as context but never counts by itself.
    """
    relset = normalize_relation_set(row.get("disease_drug_relset"))
    rel = normalize_name(row.get("first_rel"))
    if rel:
        relset.add(rel)
    return bool(relset & FAMILY_PRIMARY_CLINICAL_RELS) or _qualified_tested_indication(row)


def is_secondary_treatment_context_only(row: Dict[str, Any]) -> bool:
    relset = normalize_relation_set(row.get("disease_drug_relset"))
    rel = normalize_name(row.get("first_rel"))
    if rel:
        relset.add(rel)
    return "studied_for_treatment_of" in relset and not is_primary_family_clinical_evidence(row)


def is_primary_family_evidence_row(row: Dict[str, Any]) -> bool:
    pattern = str(row.get("pattern") or "")
    if pattern == "sibling_gene_action":
        return True
    if pattern in {"family_label_edge", "family_clinical_prior"}:
        return is_primary_family_clinical_evidence(row)
    return False


def primary_prior_support_diseases(prior_row: Dict[str, Any]) -> List[str]:
    """Extract distinct clinically qualified prior diseases for one prior drug.

    Edge-level provenance is preferred. Older cached rows fall back to relation-
    level metadata. Marker/mechanism and treatment-study context never count.
    """
    out: List[str] = []
    for edge in prior_row.get("prior_relation_edges") or []:
        if not isinstance(edge, dict):
            continue
        rel = normalize_name(edge.get("rel"))
        disease = str(edge.get("disease") or "").strip()
        phase = max_numeric_value(edge.get("phase"), default=0.0)
        if not disease:
            continue
        if rel in FAMILY_PRIMARY_CLINICAL_RELS or (
            rel == "tested_indication" and phase >= float(FAMILY_TESTED_INDICATION_MIN_PHASE)
        ):
            out.append(disease)
    if out:
        return unique_keep_order(out)

    rels = normalize_relation_set(prior_row.get("prior_rels"))
    qualifies = bool(rels & FAMILY_PRIMARY_CLINICAL_RELS)
    if not qualifies and "tested_indication" in rels:
        qualifies = max_numeric_value(prior_row.get("max_phases"), default=0.0) >= float(
            FAMILY_TESTED_INDICATION_MIN_PHASE
        )
    return unique_keep_order(prior_row.get("prior_diseases") or []) if qualifies else []


def _family_support_disease_name(row: Dict[str, Any]) -> str:
    """Return the sibling/prior disease concept represented by a family row.

    Multiple targets from the same sibling disease deliberately remain one
    independent support. This prevents a single source disease with many genes
    from satisfying a broad-family multiplicity requirement by itself.
    """
    return normalize_name(row.get("bridge_node_name"))


def _family_parent_name(row: Dict[str, Any]) -> str:
    """Return the ontology parent that defines one coherent family branch.

    Current family paths end in ``... sibling -> parent -> target disease``.
    Explicit metadata is preferred, while path-node inference preserves
    compatibility with cached rows produced before ``family_parent_name`` was
    added to sparse-query outputs.
    """
    explicit = normalize_name(row.get("family_parent_name"))
    if explicit:
        return explicit

    nodes = [str(x or "").strip() for x in (row.get("path_nodes") or [])]
    if len(nodes) >= 3:
        inferred = normalize_name(nodes[-2])
        if inferred:
            return inferred
    return ""


def _family_parent_cluster_key(row: Dict[str, Any]) -> str:
    """Return a stable parent-specific evidence-cluster key.

    Missing parent provenance is never pooled across unrelated sibling diseases;
    it receives a disease-specific unresolved key instead.
    """
    parent = _family_parent_name(row)
    if parent:
        return parent
    disease = _family_support_disease_name(row) or "unknown_support_disease"
    return f"__unresolved_parent__::{disease}"


def _family_evidence_support_key(row: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
    """Deterministic ordering key for accepted family evidence rows."""
    return (
        normalize_name(row.get("family_evidence_cluster_key") or _family_parent_cluster_key(row)),
        _family_support_disease_name(row),
        normalize_gene(row.get("target_gene")),
        str(row.get("pattern") or ""),
        normalize_name(row.get("first_rel")),
    )


def _family_path_score(row: Dict[str, Any]) -> float:
    return safe_float(first_not_none(row.get("path_score"), row.get("score")), 0.0)


def filter_family_only_evidence_paths(paths: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only a coherent, sufficiently supported family-evidence branch.

    Family support is counted *within one ontology parent*. Distinct sibling
    diseases from unrelated parent branches cannot be pooled to cross the
    multiplicity threshold. Multiple genes from one sibling still count once.

    Qualified embedding-prior diseases provide at most one corroboration unit
    per parent cluster. This preserves independent clinical-prior corroboration
    without allowing a long semantic-neighborhood list to overwhelm graph
    coherence. A non-family mechanistic path remains independently sufficient.
    Treatment-study-only relations are annotation/secondary context and never
    verify a candidate alone.
    """
    rows = list(paths or [])
    if not rows:
        return []

    non_family = [r for r in rows if str(r.get("pattern") or "") not in FAMILY_EVIDENCE_PATTERNS]
    family_rows = [r for r in rows if str(r.get("pattern") or "") in FAMILY_EVIDENCE_PATTERNS]
    if not family_rows:
        return rows

    for row in family_rows:
        parent_name = _family_parent_name(row)
        cluster_key = _family_parent_cluster_key(row)
        row["family_parent_name"] = parent_name or None
        row["family_evidence_cluster_key"] = cluster_key
        if is_primary_family_evidence_row(row):
            row["family_evidence_role"] = "primary_family_evidence"
        elif is_secondary_treatment_context_only(row):
            row["family_evidence_role"] = "secondary_treatment_context"
        else:
            row["family_evidence_role"] = "secondary_unqualified_clinical_context"

    if non_family:
        for row in family_rows:
            row["family_evidence_independent_supports"] = 1
            row["family_evidence_required_supports"] = 1
            row["family_evidence_acceptance"] = "supported_by_non_family_path"
            row["family_evidence_selected_cluster"] = True
        return rows

    primary_rows = [r for r in family_rows if is_primary_family_evidence_row(r)]
    if not primary_rows:
        for row in family_rows:
            row["family_evidence_independent_supports"] = 0
            row["family_evidence_required_supports"] = family_required_independent_supports(row)
            row["family_evidence_acceptance"] = "secondary_context_only"
            row["family_evidence_selected_cluster"] = False
        return []

    by_cluster: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in family_rows:
        by_cluster[_family_parent_cluster_key(row)].append(row)

    cluster_stats: List[Dict[str, Any]] = []
    for cluster_key, cluster_rows in by_cluster.items():
        cluster_primary = [r for r in cluster_rows if is_primary_family_evidence_row(r)]
        if not cluster_primary:
            continue

        supporting_diseases = {
            _family_support_disease_name(r)
            for r in cluster_primary
            if _family_support_disease_name(r)
        }
        prior_diseases = {
            normalize_name(x)
            for r in cluster_rows
            for x in (r.get("prior_family_support_diseases") or [])
            if str(x or "").strip()
        }
        prior_diseases.discard("")

        # Semantic/embedding prior provenance is an independent source class, but
        # it is deliberately capped at one support unit inside a graph-coherent
        # parent branch. This blocks multi-source disease-list accumulation.
        prior_novel = sorted(prior_diseases - supporting_diseases)
        prior_corroboration_units = 1 if prior_novel else 0
        support_count = len(supporting_diseases) + prior_corroboration_units
        required = min(family_required_independent_supports(r) for r in cluster_primary)
        specificity = max(family_hierarchy_specificity(r) for r in cluster_primary)
        best_score = max((_family_path_score(r) for r in cluster_primary), default=0.0)
        accepted = support_count >= required

        stat = {
            "cluster_key": cluster_key,
            "rows": cluster_rows,
            "primary_rows": cluster_primary,
            "supporting_diseases": sorted(supporting_diseases),
            "prior_diseases": sorted(prior_diseases),
            "prior_novel_diseases": prior_novel,
            "prior_corroboration_units": prior_corroboration_units,
            "support_count": support_count,
            "required": required,
            "specificity": specificity,
            "best_score": best_score,
            "accepted": accepted,
        }
        cluster_stats.append(stat)

        for row in cluster_rows:
            row["family_evidence_independent_supports"] = support_count
            row["family_evidence_required_supports"] = required
            row["family_evidence_support_diseases"] = stat["supporting_diseases"]
            row["family_evidence_prior_support_diseases"] = stat["prior_diseases"]
            row["family_evidence_prior_corroboration_units"] = prior_corroboration_units
            row["family_evidence_cluster_specificity"] = specificity
            row["family_evidence_cluster_best_score"] = best_score
            row["family_evidence_acceptance"] = (
                "accepted_parent_coherent_support" if accepted
                else "insufficient_parent_coherent_support"
            )
            row["family_evidence_selected_cluster"] = False

    accepted_clusters = [s for s in cluster_stats if s["accepted"]]
    if not accepted_clusters:
        return []

    # Use one strongest coherent branch; never combine support across parents.
    accepted_clusters.sort(
        key=lambda s: (
            -(int(s["support_count"]) - int(s["required"])),
            -float(s["best_score"]),
            -float(s["specificity"]),
            -int(s["support_count"]),
            str(s["cluster_key"]),
        )
    )
    selected = accepted_clusters[0]
    selected_key = str(selected["cluster_key"])
    for row in family_rows:
        if _family_parent_cluster_key(row) == selected_key:
            row["family_evidence_selected_cluster"] = True
            row["family_evidence_acceptance"] = "accepted_parent_coherent_support"
        elif row.get("family_evidence_acceptance") == "accepted_parent_coherent_support":
            row["family_evidence_acceptance"] = "accepted_but_nonselected_parent_cluster"

    return list(selected["rows"])


def has_only_adme_drug_gene_evidence(row: Dict[str, Any]) -> bool:
    """True when the canonical drug-gene evidence is ADME-only.

    target/inhibition/activation/modulation/binding evidence prevents this flag;
    enzyme/transporter/carrier alone is retained only as weak contextual support.
    """
    relset = normalize_relation_set(row.get("drug_gene_relset"))
    first_rel = normalize_name(row.get("first_rel"))
    if first_rel in DRUG_GENE_RELS:
        relset.add(first_rel)
    return bool(relset) and relset.issubset(DRUG_GENE_ADME_RELS)


def action_score_candidate(
    rel: Any,
    action_type: Any = None,
    relation_set: Any = None,
    action_types: Any = None,
) -> float:
    """Score DRUG->GENE pharmacological action evidence.

    Uses relation type, Action_Type, and same drug-gene pair multi-relation
    evidence. `target + inhibition/activation/...` is treated as confirmatory
    evidence rather than as two unrelated rows.
    """
    r = normalize_name(rel)
    relset = normalize_relation_set(relation_set)
    if r:
        relset.add(r)

    base = _drug_gene_action_base_score(r, action_type, action_types)

    # If a returned row is the broad `target` edge but the same DRUG-GENE pair
    # also has a directional edge, lift it near the directional evidence level.
    # Merged Action_Type values are retained during path canonicalization and
    # are used here when they are compatible with the directional relation.
    if r == "target" and (relset & DRUG_GENE_DIRECTIONAL_RELS):
        base = max(
            base,
            max(
                _drug_gene_action_base_score(x, action_type, action_types)
                for x in (relset & DRUG_GENE_DIRECTIONAL_RELS)
            ) - 0.10,
        )

    # Confirmatory bonus: target annotation + directional MoA on the same pair.
    if "target" in relset and r in DRUG_GENE_DIRECTIONAL_RELS:
        base += 0.25

    # Weak context bonus: target annotation + enzyme/transporter/carrier.
    if "target" in relset and r in DRUG_GENE_ADME_RELS:
        base += 0.08

    # Defensive only. Audit found no activation+inhibition conflicts, but keep a
    # small mixed-direction penalty in case the KG changes.
    if {"activation", "inhibition"}.issubset(relset):
        base -= 0.20

    return round(max(0.0, base), 4)


DISEASE_DRUG_VERIFICATION_REL_WEIGHTS = {
    # DISEASE->DRUG therapeutic-support edges. Marker/mechanism-study edges are
    # context-only and are deliberately excluded from candidate generation.
    "indication": 0.65,
    "off_label_use": 0.35,
    "tested_indication": 0.20,  # fallback only; phase-aware below when Max_Phase exists.
    "studied_for_treatment_of": 0.12,
    "contraindication": -1.20,
}

TESTED_INDICATION_PHASE_WEIGHTS = {
    4.0: 0.35,
    3.0: 0.28,
    2.0: 0.20,
    1.0: 0.12,
    0.5: 0.08,
    0.0: 0.05,
}

DISEASE_DRUG_POSITIVE_RELS = {
    "indication",
    "off_label_use",
    "tested_indication",
    "studied_for_treatment_of",
}


def tested_indication_phase_score(max_phase: Any) -> float:
    phase = max_numeric_value(max_phase, default=0.0)
    if phase >= 4.0:
        return TESTED_INDICATION_PHASE_WEIGHTS[4.0]
    if phase >= 3.0:
        return TESTED_INDICATION_PHASE_WEIGHTS[3.0]
    if phase >= 2.0:
        return TESTED_INDICATION_PHASE_WEIGHTS[2.0]
    if phase >= 1.0:
        return TESTED_INDICATION_PHASE_WEIGHTS[1.0]
    if phase >= 0.5:
        return TESTED_INDICATION_PHASE_WEIGHTS[0.5]
    return TESTED_INDICATION_PHASE_WEIGHTS[0.0]


def is_disease_drug_prior_pattern(pattern: Any) -> bool:
    return str(pattern or "") in {
        "family_label_edge",
        "family_disease_drug_prior",
        "sibling_disease_drug_prior",
    }


def disease_drug_prior_verification_score(
    rel: Any,
    rel_max_phase: Any = None,
    relation_set: Any = None,
) -> float:
    """Weak score for transferred DISEASE->DRUG clinical relations.

    This remains separate from action_score_candidate(), which is only for
    DRUG->GENE pharmacological action relations.
    """
    r = normalize_name(rel)
    relset = normalize_relation_set(relation_set)
    if r:
        relset.add(r)

    if r == "tested_indication":
        score = tested_indication_phase_score(rel_max_phase)
    else:
        score = float(DISEASE_DRUG_VERIFICATION_REL_WEIGHTS.get(r, 0.0))

    # Mixed positive + contraindication transfer is not deleted, but is risk-qualified.
    if "contraindication" in relset and (relset & DISEASE_DRUG_POSITIVE_RELS) and r != "contraindication":
        score -= 0.18

    return round(score, 4)

def is_generic_term(x: Any) -> bool:
    return normalize_name(x) in GENERIC_BIO_TERMS


def term_match_bonus(name: Any, terms: List[str]) -> float:
    lname = normalize_name(name)
    if not lname:
        return 0.0

    for t in terms or []:
        nt = normalize_name(t)
        if len(nt) >= 4 and (nt in lname or lname in nt):
            return 0.8
    return 0.0

def neighbor_prior_score(row: Dict[str, Any]) -> float:
    shared_gene_count = safe_float(row.get("shared_gene_count"))
    neighbor_count = safe_float(row.get("neighbor_disease_count"))
    prior_count = safe_float(row.get("indication_prior_count"))

    score = 0.0
    score += min(shared_gene_count, 5) * 0.15
    score += min(neighbor_count, 5) * 0.10
    score += min(prior_count, 5) * 0.10
    return score
    
def get_neighbor_disease_treatment_prior(
    driver,
    disease: str,
    anchor_genes: List[str],
    heldout_diseases: Optional[List[str]] = None,
    limit: int = 300,
) -> List[Dict[str, Any]]:
    """
    Similar/neighbor disease의 indication/off_label drug prior.
    단, target disease / heldout disease의 direct relation은 제외.
    """
    heldout_diseases = heldout_diseases or [disease]

    q = """
    MATCH (target:DISEASE)
    WHERE toLower(target.name) = toLower($disease)

    MATCH (target)-[:associated_with]-(g:GENE)-[:associated_with]-(near_dis:DISEASE)
    WHERE near_dis.name IS NOT NULL
      AND NOT toLower(near_dis.name) IN [x IN $heldout_diseases | toLower(x)]

    OPTIONAL MATCH (near_dis)-[r:indication|off_label_use]-(dr:DRUG)

    WHERE dr.name IS NOT NULL
      AND (
        size($anchor_genes) = 0
        OR g.name IN $anchor_genes
      )

    RETURN
      dr.name AS drug,
      collect(DISTINCT near_dis.name)[0..10] AS neighbor_diseases,
      collect(DISTINCT g.name)[0..20] AS shared_genes,
      collect(DISTINCT type(r)) AS prior_rels,
      count(DISTINCT near_dis) AS neighbor_disease_count,
      count(DISTINCT g) AS shared_gene_count,
      count(DISTINCT r) AS indication_prior_count
    ORDER BY
      indication_prior_count DESC,
      shared_gene_count DESC,
      neighbor_disease_count DESC
    LIMIT $limit
    """

    return run_query_auto(
        driver,
        q,
        {
            "disease": disease,
            "anchor_genes": anchor_genes or [],
            "heldout_diseases": heldout_diseases,
            "limit": limit,
        },
        timeout_sec=180,
    )
    
def anchor_candidate_queries() -> Dict[str, str]:
    drug_gene_rel_union = "|".join(DRUG_GENE_RELS)

    prohibited_path_rels = ", ".join([f"'{x}'" for x in PROHIBITED_PATH_RELS])

    common_return = """
    RETURN
      dr.name AS drug,
      type(r0) AS first_rel,
      r0.Action_Type AS action_type,
      [(dr)-[rr]-(tg) WHERE type(rr) IN ['target', 'inhibition', 'activation', 'binding', 'modulation', 'enzyme', 'transporter', 'carrier'] | type(rr)] AS drug_gene_relset,
      [x IN [(dr)-[rr]-(tg) WHERE type(rr) IN ['target', 'inhibition', 'activation', 'binding', 'modulation', 'enzyme', 'transporter', 'carrier'] | rr.Action_Type] WHERE x IS NOT NULL] AS drug_gene_action_types,
      tg.name AS target_gene,
      anchor_name,
      anchor_type,
      length(p) AS hop_len,
      [n IN nodes(p) | coalesce(n.name, n.node_id, elementId(n))] AS path_nodes,
      [n IN nodes(p) | labels(n)] AS path_labels,
      [r IN relationships(p) | type(r)] AS path_rels,
      dr.Group AS `Group`,
      dr.Max_Phase AS MaxPhase,
      dr.Black_Box AS BlackBox,
      dr.Withdrawn_Flag AS Withdrawn,
      dr.Inorganic_Flag AS Inorganic,
      dr.Molecular_Weight AS MW,
      dr.Polar_Surface_Area AS TPSA,
      coalesce(dr.CX_LogP, dr.XLogP, dr.CLogP, dr.AlogP) AS LogP,
      dr.QED_Weighted AS QED,
      size([(tg)-[:associated_with]-(x:DISEASE) | x]) AS target_gene_disease_degree,
      size([(tg)-[:ppi]-(x:GENE) | x]) AS target_gene_ppi_degree,
      size([(dr)-[rr]-(x:GENE)
            WHERE type(rr) IN ['target', 'inhibition', 'activation', 'binding', 'modulation', 'enzyme', 'transporter', 'carrier'] | x]) AS drug_gene_degree,
      disease_gene_degree,
      bridge_gene_degree,
      phenotype_disease_degree
    """

    common_path_filter = f"""
      ALL(n IN nodes(p)
          WHERE SINGLE(m IN nodes(p) WHERE elementId(m) = elementId(n)))
      AND NONE(r IN relationships(p)
               WHERE type(r) IN [{prohibited_path_rels}])
      AND NONE(n IN nodes(p)
               WHERE any(lbl IN labels(n) WHERE lbl IN ['EXPO']))
    """

    return {
        # Weak rescue: candidate drug has a direct drug-gene action/PK relation to a
        # disease-associated gene. This catches short paths such as
        # It is not treated as strong mechanism evidence by scoring.
        "drug_to_disease_direct_gene_action": f"""
        MATCH (dis:DISEASE)
        WHERE toLower(dis.name) = toLower($disease)

        MATCH p =
          (dr:DRUG)-[r0:{drug_gene_rel_union}]-
          (tg:GENE)-[:associated_with]-
          (dis)

        WHERE tg.name IS NOT NULL
          AND {common_path_filter}

        WITH dr, r0, tg, tg AS dg, p,
             tg.name AS anchor_name,
             'DISEASE_GENE_DIRECT' AS anchor_type,
             size([(tg)-[:associated_with]-(x:DISEASE) | x]) AS disease_gene_degree,
             0 AS bridge_gene_degree,
             0 AS phenotype_disease_degree
        {common_return}
        LIMIT $limit
        """,

        # Candidate drug targets a gene close to the active anchor-gene pool
        "drug_to_anchor_gene": f"""
        MATCH (ag:GENE)
        WHERE ag.name IN $anchor_genes
        
        MATCH p =
          (ag)
          -[:ppi|interacts_with*0..1]-
          (tg:GENE)
          -[r0:{drug_gene_rel_union}]-
          (dr:DRUG)
        
        WHERE tg.name IS NOT NULL
          AND {common_path_filter}
        
        WITH dr, r0, tg, ag, p,
             ag.name AS anchor_name,
             'GENE' AS anchor_type,
             size([(ag)-[:associated_with]-(x:DISEASE) | x]) AS disease_gene_degree,
             0 AS bridge_gene_degree,
             0 AS phenotype_disease_degree
        {common_return}
        LIMIT $limit
        """,

        # Candidate drug target connects to BP/PATH whose name matches the active anchor-term pool
        "drug_to_anchor_bioprocess": f"""
        MATCH (dr:DRUG)-[r0:{drug_gene_rel_union}]-(tg:GENE)
        MATCH p = (dr)-[r0]-(tg)-[:interacts_with]-(bio:BP|PATH)
        WHERE tg.name IS NOT NULL
          AND bio.name IS NOT NULL
          AND any(t IN $anchor_terms
                  WHERE toLower(bio.name) CONTAINS t
                     OR t CONTAINS toLower(bio.name))
          AND NOT toLower(bio.name) IN $generic_bio_terms
          AND {common_path_filter}
        WITH dr, r0, tg, bio, p,
             bio.name AS anchor_name,
             CASE
               WHEN 'BP' IN labels(bio) THEN 'BP'
               WHEN 'PATH' IN labels(bio) THEN 'PATH'
               ELSE head(labels(bio))
             END AS anchor_type,
             0 AS disease_gene_degree,
             size([(bio)-[:interacts_with]-(x:GENE) | x]) AS bridge_gene_degree,
             0 AS phenotype_disease_degree
        {common_return}
        LIMIT $limit
        """,

        # Candidate drug connects to disease via anchor-relevant BP/PATH
        "drug_to_disease_via_anchor_bio": f"""
        MATCH (dis:DISEASE)
        WHERE toLower(dis.name) = toLower($disease)

        MATCH (dr:DRUG)-[r0:{drug_gene_rel_union}]-(tg:GENE)
        MATCH p =
          (dr)-[r0]-(tg)
          -[:interacts_with]-
          (bio:BP|PATH)
          -[:interacts_with]-
          (dg:GENE)
          -[:associated_with]-
          (dis)

        WHERE bio.name IS NOT NULL
          AND any(t IN $anchor_terms
                  WHERE toLower(bio.name) CONTAINS t
                     OR t CONTAINS toLower(bio.name))
          AND NOT toLower(bio.name) IN $generic_bio_terms
          AND {common_path_filter}

        WITH dr, r0, tg, bio, dg, p,
             bio.name AS anchor_name,
             CASE
               WHEN 'BP' IN labels(bio) THEN 'BP'
               WHEN 'PATH' IN labels(bio) THEN 'PATH'
               ELSE head(labels(bio))
             END AS anchor_type,
             size([(dg)-[:associated_with]-(x:DISEASE) | x]) AS disease_gene_degree,
             size([(bio)-[:interacts_with]-(x:GENE) | x]) AS bridge_gene_degree,
             0 AS phenotype_disease_degree
        {common_return}
        LIMIT $limit
        """,

        # Candidate drug connects to disease-associated genes, constrained by the active anchor-gene pool
        "drug_to_disease_anchor_gene": f"""
        MATCH (dis:DISEASE)
        WHERE toLower(dis.name) = toLower($disease)

        MATCH (dr:DRUG)-[r0:{drug_gene_rel_union}]-(tg:GENE)
        MATCH p =
          (dr)-[r0]-(tg)
          -[:ppi|interacts_with*0..1]-
          (dg:GENE)
          -[:associated_with]-
          (dis)

        WHERE dg.name IN $anchor_genes
          AND {common_path_filter}

        WITH dr, r0, tg, dg, p,
             dg.name AS anchor_name,
             'DISEASE_GENE' AS anchor_type,
             size([(dg)-[:associated_with]-(x:DISEASE) | x]) AS disease_gene_degree,
             0 AS bridge_gene_degree,
             0 AS phenotype_disease_degree
        {common_return}
        LIMIT $limit
        """,

        # Candidate drug connects to disease through an anchor gene and pathway/BP
        "drug_to_disease_via_anchor_gene_path": f"""
        MATCH (dis:DISEASE)
        WHERE toLower(dis.name) = toLower($disease)
        
        MATCH p =
          (dr:DRUG)-[r0:{drug_gene_rel_union}]-
          (tg:GENE)
          -[:ppi|interacts_with*0..1]-
          (ag:GENE)
          -[:interacts_with]-
          (bio:BP|PATH)
          -[:interacts_with]-
          (dg:GENE)
          -[:associated_with]-
          (dis)
        
        WHERE ag.name IN $anchor_genes
          AND bio.name IS NOT NULL
          AND (
                tg.name IN $anchor_genes
                OR dg.name IN $anchor_genes
                OR any(t IN $anchor_bio_path_terms
                       WHERE toLower(bio.name) CONTAINS t
                          OR t CONTAINS toLower(bio.name))
              )
          AND NOT toLower(bio.name) IN $generic_bio_terms
          AND {common_path_filter}
        
        WITH dr, r0, tg, ag, bio, dg, p,
             bio.name AS anchor_name,
             CASE
               WHEN 'BP' IN labels(bio) THEN 'BP'
               WHEN 'PATH' IN labels(bio) THEN 'PATH'
               ELSE head(labels(bio))
             END AS anchor_type,
             size([(dg)-[:associated_with]-(x:DISEASE) | x]) AS disease_gene_degree,
             size([(bio)-[:interacts_with]-(x:GENE) | x]) AS bridge_gene_degree,
             0 AS phenotype_disease_degree
        {common_return}
        LIMIT $limit
        """,

        # Candidate drug connects to disease through anchor BP/PATH.
        "drug_target_to_anchor_disease_gene_via_bio": f"""
        MATCH (dis:DISEASE)
        WHERE toLower(dis.name) = toLower($disease)
        
        MATCH p =
          (dr:DRUG)-[r0:{drug_gene_rel_union}]-
          (tg:GENE)
          -[:interacts_with]-
          (bio:BP|PATH)
          -[:interacts_with]-
          (dg:GENE)
          -[:associated_with]-
          (dis)
        
        WHERE dg.name IN $anchor_genes
          AND bio.name IS NOT NULL
          AND (
                tg.name IN $anchor_genes
                OR any(t IN $anchor_bio_path_terms
                       WHERE toLower(bio.name) CONTAINS t
                          OR t CONTAINS toLower(bio.name))
              )
          AND NOT toLower(bio.name) IN $generic_bio_terms
          AND {common_path_filter}
        
        WITH dr, r0, tg, bio, dg, p,
             bio.name AS anchor_name,
             CASE
               WHEN 'BP' IN labels(bio) THEN 'BP'
               WHEN 'PATH' IN labels(bio) THEN 'PATH'
               ELSE head(labels(bio))
             END AS anchor_type,
             size([(dg)-[:associated_with]-(x:DISEASE) | x]) AS disease_gene_degree,
             size([(bio)-[:interacts_with]-(x:GENE) | x]) AS bridge_gene_degree,
             0 AS phenotype_disease_degree
        {common_return}
        LIMIT $limit
        """,

        # Narrow replacement for the broad/slow PPI-to-biology bridge.
        # Candidate drug target is one 1-hop gene-gene interaction away from an active anchor gene
        # that is directly associated with the query disease. This keeps the path disease-grounded
        # and avoids broad BP/PATH expansion noise.
        "drug_target_ppi_to_anchor_gene": f"""
        MATCH (dis:DISEASE)
        WHERE toLower(dis.name) = toLower($disease)

        MATCH (ag:GENE)-[:associated_with]-(dis)
        WHERE ag.name IN $anchor_genes
          AND ag.name IS NOT NULL

        MATCH p =
          (dr:DRUG)-[r0:{drug_gene_rel_union}]-
          (tg:GENE)
          -[:ppi|interacts_with]-
          (ag)

        WHERE tg.name IS NOT NULL
          AND tg.name <> ag.name
          AND {common_path_filter}

        WITH dr, r0, tg, ag, p,
             ag.name AS anchor_name,
             'DISEASE_GENE' AS anchor_type,
             size([(ag)-[:associated_with]-(x:DISEASE) | x]) AS disease_gene_degree,
             0 AS bridge_gene_degree,
             0 AS phenotype_disease_degree
        {common_return}
        LIMIT $limit
        """,

        "drug_to_disease_via_vector_phenotype_gene": f"""
        MATCH (dis:DISEASE)
        WHERE toLower(dis.name) = toLower($disease)

        MATCH (dis)-[:phenotype_present]-(ph:PHENO)
        WHERE ph.name IN $anchor_phenotypes

        MATCH (ph)-[:associated_with]-(ag:GENE)
        MATCH p =
          (dr:DRUG)-[r0:{drug_gene_rel_union}]-
          (tg:GENE)
          -[:ppi|interacts_with*0..1]-
          (ag)
          -[:associated_with]-
          (ph)
          -[:phenotype_present]-
          (dis)

        WHERE tg.name IS NOT NULL
          AND ag.name IS NOT NULL
          AND {common_path_filter}

        WITH dr, r0, tg, ph, ag, p,
             ph.name AS anchor_name,
             'PHENO' AS anchor_type,
             size([(ag)-[:associated_with]-(x:DISEASE) | x]) AS disease_gene_degree,
             0 AS bridge_gene_degree,
             size([(ph)-[:phenotype_present]-(x:DISEASE) | x]) AS phenotype_disease_degree
        {common_return}
        LIMIT $limit
        """
    }

def broad_anchor_penalty(anchor_name: Any) -> float:
    lname = normalize_name(anchor_name)
    if not lname:
        return 0.0

    penalties = []

    # exact-only generic anchors
    if lname in EXACT_TOO_BROAD_ANCHORS:
        penalties.append(2.5)

    # exact broad ontology terms
    if lname in BROAD_ANCHOR_TERMS:
        penalties.append(1.2)

    # substring broad terms, but only if not exact
    elif any(t in lname for t in BROAD_ANCHOR_TERMS):
        penalties.append(0.7)

    # disease-connected but too broad
    if lname in VERY_BROAD_BUT_DISEASE_CONNECTED_ANCHORS:
        penalties.append(2.0)

    if lname in NOISY_BUT_DISEASE_CONNECTED_ANCHORS:
        penalties.append(1.5)

    return max(penalties) if penalties else 0.0


def pattern_specificity_score(pattern: str) -> float:
    if pattern == "drug_to_disease_direct_gene_action":
        # Very short disease-gene--drug action/transporter path. Useful as weak rescue,
        # but not a strong mechanistic path by itself.
        return 0.25

    if pattern == "drug_target_to_anchor_disease_gene_via_bio":
        return 2.4

    if pattern == "drug_anchor_target_to_disease_via_bio":
        return 2.2

    if pattern == "drug_target_ppi_to_anchor_gene":
        # Same base specificity as the old PPI bridge so the pilot isolates topology replacement.
        return 2.1

    if pattern == "drug_to_disease_via_vector_phenotype_gene":
        # Shared vector-neighbor/target phenotype with a bounded gene bridge.
        return 0.8

    if pattern == "drug_to_disease_via_anchor_bio":
        return 2.0

    if pattern == "drug_to_disease_via_anchor_gene_path":
        return 1.8

    if pattern == "drug_to_disease_anchor_gene":
        return 0.4

    if pattern == "drug_to_anchor_gene":
        return -0.8

    if pattern == "drug_to_anchor_bioprocess":
        return -1.5

    return 0.0
    

def score_anchor_candidate_path(
    row: Dict[str, Any],
    axis: Dict[str, Any],
) -> float:
    first_rel = row.get("first_rel")
    pattern = row.get("pattern", "")

    target_gene = normalize_gene(row.get("target_gene"))
    anchor_name = row.get("anchor_name")
    anchor_type = str(row.get("anchor_type") or "")

    anchor_genes = extract_gene_mentions(axis.get("anchor_genes") or [])
    anchor_terms = axis.get("anchor_terms", [])
    
    anchor_term_match = term_match_bonus(anchor_name, anchor_terms)
    row["anchor_term_match"] = anchor_term_match
    
    score = 0.0

    # 1) Disease-connected paths should dominate retrieval ranking
    score += pattern_specificity_score(pattern)

    # 2) Drug-target/action quality
    a = action_score_candidate(first_rel, row.get("action_type"), row.get("drug_gene_relset"), row.get("drug_gene_action_types"))

    # Rescue previous target only if it is directly anchor-supported.
    if str(first_rel or "").lower() == "target":
        if target_gene in anchor_genes:
            a = max(a, 1.5)
        else:
            a = min(a, 0.5)

    score += 1.1 * a

    # 3) Axis priority
    score += priority_weight(axis.get("priority")) * 0.8

    # 4) Anchor support
    if target_gene in anchor_genes:
        score += 1.8

    if normalize_gene(anchor_name) in anchor_genes:
        score += 1.0

    # 5) Anchor type
    if anchor_type in {"BP", "PATH"}:
        tm = anchor_term_match
    
        if tm > 0:
            score += 1.0 + tm
        else:
            score -= 0.7
    elif anchor_type == "DISEASE_GENE_DIRECT":
        # Direct disease-gene action/transporter edge is useful for recall rescue,
        # but should remain weaker than target-context BP/PATH mechanisms.
        score += 0.15
    elif anchor_type == "DISEASE_GENE":
        score += 0.8
    elif anchor_type == "GENE":
        score += 0.2
    elif anchor_type == "PHENO":
        score += 0.1
    elif anchor_type == "ANAT":
        score -= 1.0
    elif anchor_type in {"MF", "CC"}:
        score -= 1.5

    # 6) Hop length
    hop_len = int(row.get("hop_len") or 0)
    if 3 <= hop_len <= 5:
        score += 0.5
    elif hop_len == 2:
        # 2-hop is allowed, but anchor-only 2-hop is often too shallow.
        if pattern in DISEASE_CONNECTED_PATTERNS:
            score += 0.2
        else:
            score -= 0.7
    elif hop_len > 5:
        score -= 0.5

    # 7) Generic / broad anchor penalty
    score -= generic_bridge_penalty(anchor_name)
    score -= broad_anchor_penalty(anchor_name)

    # 8) KG-degree specificity penalty (no fixed hub-gene list)
    score -= structural_degree_penalty(row)

    # 9) Translational properties are intentionally excluded here.
    # Path scoring represents graph-mechanistic evidence only; approval, phase,
    # physicochemical and safety attributes are applied once after all candidate
    # generation and sparse rescue steps.

    return round(score, 4)


def expand_mechanism_terms_to_genes(terms: List[str], enabled: bool = True) -> List[str]:
    """No text-to-gene lookup table is used; genes must come from KG/RAG context."""
    return []



def _retry_anchor_pattern_after_timeout(
    driver,
    pattern_name: str,
    query: str,
    params: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Retry one expensive anchor query and return explicit telemetry.

    The telemetry is persisted so pattern-by-area efficiency can be audited after
    the complete benchmark without parsing console logs. It never changes scores.
    """
    cfg = PATTERN_TIMEOUT_RETRY_CONFIG or {}
    disabled = not bool(cfg.get("enabled", False)) or pattern_name not in set(cfg.get("patterns") or [])
    if disabled:
        return [], {
            "attempted": False,
            "status": "not_configured",
            "rows": 0,
            "timeout_sec": None,
        }
    slim = dict(params or {})
    slim["anchor_genes"] = list(slim.get("anchor_genes") or [])[: int(cfg.get("anchor_gene_limit", 6))]
    slim["anchor_terms"] = list(slim.get("anchor_terms") or [])[: int(cfg.get("anchor_term_limit", 24))]
    slim["anchor_bio_path_terms"] = list(slim.get("anchor_bio_path_terms") or [])[: int(cfg.get("anchor_term_limit", 24))]
    slim["anchor_phenotypes"] = list(slim.get("anchor_phenotypes") or [])[: int(cfg.get("anchor_term_limit", 24))]
    slim["limit"] = min(int(slim.get("limit") or 12), int(cfg.get("result_limit", 12)))
    retry_timeout = int(cfg.get("timeout_sec", 18))
    print(
        f"    [PATTERN RETRY START] {pattern_name} | limit={slim['limit']} | "
        f"anchors={len(slim['anchor_genes'])} | timeout={retry_timeout}s",
        flush=True,
    )
    try:
        rows = run_query_auto(driver, query, slim, timeout_sec=retry_timeout)
        print(
            f"    [PATTERN RETRY DONE] {pattern_name} | rows={len(rows)} | timeout={retry_timeout}s",
            flush=True,
        )
        return rows, {
            "attempted": True,
            "status": "ok",
            "rows": len(rows),
            "timeout_sec": retry_timeout,
        }
    except TimeoutError as exc:
        print(
            f"    [PATTERN RETRY TIMEOUT] {pattern_name} | timeout={retry_timeout}s | {exc}",
            flush=True,
        )
        return [], {
            "attempted": True,
            "status": "timeout",
            "rows": 0,
            "timeout_sec": retry_timeout,
            "error": str(exc),
        }
    except Exception as exc:
        print(
            f"    [PATTERN RETRY ERROR] {pattern_name} | {type(exc).__name__}: {exc}",
            flush=True,
        )
        return [], {
            "attempted": True,
            "status": "error",
            "rows": 0,
            "timeout_sec": retry_timeout,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _vector_path_source_diseases(row: Dict[str, Any], mechanism_anchor_obj: Dict[str, Any]) -> List[str]:
    pool = (mechanism_anchor_obj or {}).get("vector_anchor_pool") or {}
    maps = pool.get("anchor_source_maps") or {}
    anchor_name = row.get("anchor_name")
    anchor_type = normalize_name(row.get("anchor_type"))
    if anchor_type in {"gene", "disease_gene", "disease_gene_direct"}:
        return unique_keep_order((maps.get("gene") or {}).get(normalize_gene(anchor_name), []))
    if anchor_type == "bp":
        return unique_keep_order((maps.get("bio") or {}).get(normalize_name(anchor_name), []))
    if anchor_type == "path":
        return unique_keep_order((maps.get("pathway") or {}).get(normalize_name(anchor_name), []))
    if anchor_type == "pheno":
        return unique_keep_order((maps.get("phenotype") or {}).get(normalize_name(anchor_name), []))
    return []


def retrieve_anchor_guided_drug_candidates(
    driver,
    disease: str,
    mechanism_anchor_obj: Dict[str, Any],
    candidate_k: int = CANDIDATE_K,
    limit_per_axis_pattern: int = PATH_LIMIT_PER_PATTERN,
    ablation_mode: str = ABLATION_MODE,
    use_mechanism_lookup: bool = True,
    case_area: str = "",
    use_target_context_graph: bool = True,
    use_vector_anchor_graph: bool = True,
    use_common_direct_graph: bool = True,
):
    """Generate candidates from target-context and vector-neighbor GraphRAG branches."""
    target_axes = axis_anchor_bundle(
        mechanism_anchor_obj,
        max_axes=3,
        use_mechanism_lookup=use_mechanism_lookup,
    ) if use_target_context_graph else []
    vector_axes = vector_anchor_bundle(
        mechanism_anchor_obj,
        use_mechanism_lookup=False,
    ) if use_vector_anchor_graph else []
    for axis in target_axes:
        axis["anchor_branch"] = "disease_context_fallback_graph"
    for axis in vector_axes:
        axis["anchor_branch"] = "vector_neighbor_anchor_graph"

    # The direct disease-gene motif is shared by both anchor branches and does not
    # depend on a selected axis. Represent it explicitly so it executes exactly once
    # and is not incorrectly attributed to whichever axis happened to be first.
    common_direct_axes = []
    if use_common_direct_graph:
        common_direct_axes = [{
            "axis_name": "Target-disease direct gene",
            "priority": "medium",
            "axis_rationale": "Axis-independent target disease-gene pharmacology motif.",
            "anchor_source": "target_disease_graph",
            "anchor_genes": [],
            "anchor_terms": [],
            "anchor_bio_path_terms": [],
            "anchor_phenotypes": [],
            "anchor_branch": "common_direct_graph",
        }]
    axes = common_direct_axes + target_axes + vector_axes
    queries = anchor_candidate_queries()
    candidate_map: Dict[str, Dict[str, Any]] = {}
    branch_query_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    pattern_query_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "pattern": None,
        "area": normalize_area_name(case_area),
        "calls": 0,
        "rows": 0,
        "timeouts": 0,
        "errors": 0,
        "disabled": 0,
        "skipped": 0,
        "retry_attempts": 0,
        "retry_successes": 0,
        "retry_timeouts": 0,
        "retry_errors": 0,
        "retry_rows": 0,
        "branches": set(),
    })

    for axis in axes:
        axis_name = axis.get("axis_name", "")
        anchor_branch = axis.get("anchor_branch") or "disease_context_fallback_graph"
        anchor_genes = axis.get("anchor_genes", [])
        anchor_terms = axis.get("anchor_terms", [])
        anchor_bio_path_terms = axis.get("anchor_bio_path_terms", anchor_terms)
        anchor_phenotypes = axis.get("anchor_phenotypes", []) or []
        if anchor_branch != "common_direct_graph" and not anchor_genes and not anchor_terms and not anchor_phenotypes:
            continue
        print(f"\n[ANCHOR RETRIEVAL] branch={anchor_branch} axis={axis_name}")
        print(f"  genes={anchor_genes[:10]}")
        print(f"  terms={anchor_terms[:10]}")
        if anchor_phenotypes:
            print(f"  phenotypes={anchor_phenotypes[:10]}")

        for pattern_name, query in queries.items():
            is_direct = pattern_name == "drug_to_disease_direct_gene_action"
            is_vector_pheno = pattern_name == "drug_to_disease_via_vector_phenotype_gene"
            if anchor_branch == "common_direct_graph" and not is_direct:
                continue
            if anchor_branch != "common_direct_graph" and is_direct:
                continue
            if is_vector_pheno and anchor_branch != "vector_neighbor_anchor_graph":
                continue

            pattern_limit = area_pattern_limit(case_area, pattern_name, PATTERN_LIMITS.get(pattern_name, limit_per_axis_pattern))
            pattern_timeout = area_pattern_timeout(case_area, pattern_name, PATTERN_TIMEOUTS.get(pattern_name, VERIFY_TIMEOUT_SEC))
            stat = pattern_query_stats[pattern_name]
            stat["pattern"] = pattern_name
            stat["branches"].add(anchor_branch)
            if pattern_name in DISABLED_ANCHOR_PATTERNS or area_pattern_is_disabled(case_area, pattern_name):
                branch_query_stats[anchor_branch]["disabled"] += 1
                stat["disabled"] += 1
                continue
            if not should_run_anchor_pattern(
                pattern_name=pattern_name,
                axis=axis,
                anchor_genes=anchor_genes,
                anchor_terms=anchor_terms,
                anchor_bio_path_terms=anchor_bio_path_terms,
                anchor_phenotypes=anchor_phenotypes,
                ablation_mode=ablation_mode,
            ):
                branch_query_stats[anchor_branch]["skipped"] += 1
                stat["skipped"] += 1
                continue

            params = {
                "disease": disease,
                "anchor_genes": anchor_genes,
                "anchor_terms": anchor_terms,
                "anchor_bio_path_terms": anchor_bio_path_terms,
                "anchor_phenotypes": anchor_phenotypes,
                "generic_bio_terms": list(GENERIC_BIO_TERMS),
                "broad_hub_genes": list(BROAD_HUB_GENES),
                "limit": pattern_limit,
            }
            branch_query_stats[anchor_branch]["calls"] += 1
            stat["calls"] += 1
            print(f"    [PATTERN START] {pattern_name} | branch={anchor_branch} | limit={pattern_limit} | timeout={pattern_timeout}s", flush=True)
            try:
                rows = run_query_auto(driver, query, params, timeout_sec=pattern_timeout)
                print(f"    [PATTERN DONE] {pattern_name} | branch={anchor_branch} | rows={len(rows)}", flush=True)
            except TimeoutError as exc:
                branch_query_stats[anchor_branch]["timeouts"] += 1
                stat["timeouts"] += 1
                print(f"    [PATTERN TIMEOUT] {pattern_name} | branch={anchor_branch} | {exc}", flush=True)
                rows, retry_meta = _retry_anchor_pattern_after_timeout(driver, pattern_name, query, params)
                if retry_meta.get("attempted"):
                    stat["retry_attempts"] += 1
                    stat["retry_rows"] += int(retry_meta.get("rows") or 0)
                    if retry_meta.get("status") == "ok":
                        stat["retry_successes"] += 1
                    elif retry_meta.get("status") == "timeout":
                        stat["retry_timeouts"] += 1
                    elif retry_meta.get("status") == "error":
                        stat["retry_errors"] += 1
            except Exception as exc:
                branch_query_stats[anchor_branch]["errors"] += 1
                stat["errors"] += 1
                print(f"[ANCHOR RETRIEVAL ERROR] branch={anchor_branch} axis={axis_name} pattern={pattern_name}: {type(exc).__name__}: {exc}", flush=True)
                rows = []

            rows = canonicalize_path_rows(rows, pattern_name)
            branch_query_stats[anchor_branch]["rows"] += len(rows)
            stat["rows"] += len(rows)
            for row in rows:
                drug = str(row.get("drug") or "").strip()
                if not drug:
                    continue
                nd = normalize_name(drug)
                evidence_branch = anchor_branch
                effective_axis_name = "Target-disease direct gene" if is_direct else axis_name
                effective_axis = dict(axis)
                effective_axis["axis_name"] = effective_axis_name
                if is_direct:
                    effective_axis["priority"] = "medium"
                row["pattern"] = pattern_name
                row["anchor_branch"] = evidence_branch
                row["axis_independent_pattern"] = bool(is_direct)
                row["anchor_source_diseases"] = _vector_path_source_diseases(row, mechanism_anchor_obj) if anchor_branch == "vector_neighbor_anchor_graph" else []
                path_score = score_anchor_candidate_path(row, effective_axis)

                if nd not in candidate_map:
                    candidate_map[nd] = {
                        "drug": drug,
                        "path_count": 0,
                        "supporting_axes": set(),
                        "supporting_patterns": set(),
                        "supporting_anchor_branches": set(),
                        "best_path_score": None,
                        "best_path": None,
                        "all_paths": [],
                        "seen_path_keys": set(),
                    }
                candidate = candidate_map[nd]
                candidate["supporting_patterns"].add(pattern_name)
                candidate["supporting_anchor_branches"].add(evidence_branch)
                if not is_direct:
                    candidate["supporting_axes"].add(axis_name)
                path_key = canonical_path_key(row, pattern_name)
                if path_key in candidate["seen_path_keys"]:
                    continue
                candidate["seen_path_keys"].add(path_key)
                candidate["path_count"] += 1
                path_row = {
                    **row,
                    "axis_name": effective_axis_name,
                    "path_score": path_score,
                    "axis_priority": effective_axis.get("priority"),
                }
                candidate["all_paths"].append(path_row)
                if candidate["best_path_score"] is None or path_score > candidate["best_path_score"]:
                    candidate["best_path_score"] = path_score
                    candidate["best_path"] = path_row

    # Candidate frequency within a target/anchor class is not used as evidence.
    # Otherwise large drug classes and graph hubs receive an artificial popularity
    # bonus unrelated to mechanistic support.
    candidates: List[Dict[str, Any]] = []
    for candidate in candidate_map.values():
        if not any(pattern in DISEASE_CONNECTED_PATTERNS for pattern in candidate["supporting_patterns"]):
            continue
        best_score = candidate["best_path_score"] or 0.0
        best = candidate.get("best_path") or {}
        n_axes = len(candidate["supporting_axes"])
        n_patterns = len(candidate["supporting_patterns"])
        n_branches = len({b for b in candidate["supporting_anchor_branches"] if b != "common_direct_graph"})
        total_score = (
            best_score
            + 0.8
            + min(math.log1p(candidate["path_count"]), 2.0) * 0.10
            + min(n_axes, 3) * 0.15
            + min(n_patterns, 2) * 0.10
            + max(0, min(n_branches - 1, 1)) * 0.10
        )
        paths = sorted(candidate["all_paths"], key=lambda row: (safe_float(row.get("path_score")), -int(row.get("hop_len") or 99)), reverse=True)
        branches = sorted(candidate["supporting_anchor_branches"])
        candidates.append({
            "drug": candidate["drug"],
            "rank": None,
            "candidate_score": round(total_score, 4),
            "best_path_score": best_score,
            "path_count": candidate["path_count"],
            "supporting_axes": sorted(candidate["supporting_axes"]),
            "supporting_patterns": sorted(candidate["supporting_patterns"]),
            "supporting_anchor_branches": branches,
            "source_disease_context_fallback_graph": "disease_context_fallback_graph" in branches,
            "source_vector_neighbor_anchor_graph": "vector_neighbor_anchor_graph" in branches,
            "source_common_direct_graph": "common_direct_graph" in branches,
            "best_path": best,
            "top_paths": paths[:10],
        })

    candidates.sort(key=lambda row: (-safe_float(row.get("candidate_score")), -safe_float(row.get("best_path_score")), -safe_float(row.get("path_count")), normalize_name(row.get("drug"))))
    candidates = candidates[:candidate_k]
    for rank, candidate in enumerate(candidates, start=1):
        candidate["rank"] = rank

    output = []
    for candidate in candidates:
        best = candidate.get("best_path") or {}
        branches = candidate.get("supporting_anchor_branches") or []
        output.append({
            "drug": candidate["drug"],
            "rank": candidate["rank"],
            "confidence": "medium",
            "mechanism_axis": "; ".join(candidate.get("supporting_axes", [])[:3]),
            "suggested_targets": unique_keep_order([best.get("target_gene"), best.get("anchor_name") if best.get("anchor_type") in {"GENE", "DISEASE_GENE", "DISEASE_GENE_DIRECT"} else None]),
            "suggested_biology_terms": unique_keep_order([best.get("anchor_name") if best.get("anchor_type") in {"BP", "PATH", "PHENO"} else None] + candidate.get("supporting_axes", [])),
            "rationale": f"GraphRAG branches={branches}; best_anchor={best.get('anchor_name')}; score={candidate.get('candidate_score')}; path_count={candidate.get('path_count')}.",
            "candidate_score": candidate["candidate_score"],
            "candidate_source": "+".join(branches) if branches else "anchor_guided_graphrag",
            "candidate_sources": branches,
            "best_anchor_path": best,
            "top_anchor_paths": candidate.get("top_paths", []),
            "path_count": candidate.get("path_count"),
            "supporting_patterns": candidate.get("supporting_patterns", []),
            "supporting_axes": candidate.get("supporting_axes", []),
            "source_graph_candidate": True,
            "source_disease_context_fallback_graph": candidate.get("source_disease_context_fallback_graph", False),
            "source_vector_neighbor_anchor_graph": candidate.get("source_vector_neighbor_anchor_graph", False),
            "source_common_direct_graph": candidate.get("source_common_direct_graph", False),
        })

    return {
        "disease": disease,
        "mechanism_axes": mechanism_anchor_obj.get("mechanism_axes", []),
        "selected_target_context_axes": target_axes,
        "vector_anchor_pool": mechanism_anchor_obj.get("vector_anchor_pool") or {},
        "global_anchors": mechanism_anchor_obj.get("global_anchors", {}),
        "vector_guided_graphrag": mechanism_anchor_obj.get("vector_guided_graphrag", {"enabled": False}),
        "anchor_branch_query_stats": {key: dict(value) for key, value in branch_query_stats.items()},
        "anchor_pattern_query_stats": {
            key: {**{k: v for k, v in value.items() if k != "branches"}, "branches": sorted(value.get("branches") or [])}
            for key, value in pattern_query_stats.items()
        },
        "candidate_drugs": output,
        "raw_candidates": candidates,
    }


def score_verified_path(
    row: Dict[str, Any],
    anchor_genes: List[str],
    anchor_terms: List[str],
    mechanism_anchor_genes: Optional[List[str]] = None,
    disease_genes: Optional[List[str]] = None,
) -> float:
    first_rel = row.get("first_rel")
    target_gene = normalize_gene(row.get("target_gene"))
    disease_gene = normalize_gene(row.get("disease_gene"))
    bridge_name = row.get("bridge_node_name")
    bridge_type = str(row.get("bridge_node_type") or "")
    pattern = row.get("pattern", "")

    anchors = {normalize_gene(g) for g in anchor_genes or [] if g}
    mechanism_anchors = {normalize_gene(g) for g in mechanism_anchor_genes or [] if g}
    disease_gene_set = {normalize_gene(g) for g in disease_genes or [] if g}

    score = 0.0

    # Evidence hierarchy for the prior-seed verifier. A direct target-gene to
    # disease-gene interaction is more specific than an unmatched shared BP/PATH.
    # This rule is global and topology/semantics based; it contains no disease,
    # drug, gene, or benchmark-specific names.
    if ENABLE_VERIFIER_EVIDENCE_HIERARCHY and pattern == "gene_gene":
        score += 0.8

    if pattern in {"drug_to_disease_via_anchor_bio", "drug_to_disease_anchor_gene"}:
        score += 1.0
    elif pattern == "pheno_gene_bridge":
        # Infection/rare-disease rescue: target disease often has phenotypes but no associated genes.
        # Keep below direct disease-gene/BP evidence, but above generic fallback paths.
        score += 0.6
    elif pattern in {"family_label_edge"}:
        # Ontology/label-transfer rescue: prior drug has a treatment/literature edge to
        # a parent/child disease, not the held-out target disease itself. This is useful
        # for high-recall drug-repurposing discovery, but it is NOT mechanistic evidence.
        # Keep it as a weak contextual prior so it cannot outrank true drug-gene-disease
        # GraphRAG paths by itself.
        row["verification_evidence_class"] = "non_mechanistic_context_prior"
        specificity = family_hierarchy_specificity(row)
        row["family_hierarchy_specificity"] = specificity
        score += 0.75 * specificity
    elif pattern in {"drug_to_anchor_gene", "drug_to_anchor_bioprocess"}:
        score -= 0.3

    # Relation evidence. Most verification rows use first_rel as a DRUG-GENE
    # molecular action relation. The family_label_edge verifier is different:
    # first_rel is a transferred DISEASE-DRUG clinical relation from a related
    # disease. Keep these domains separate to avoid scoring indication/off-label
    # edges as target/action evidence.
    if is_disease_drug_prior_pattern(pattern):
        score += disease_drug_prior_verification_score(first_rel, first_not_none(row.get("relation_max_phase"), row.get("tested_indication_phases"), row.get("disease_drug_max_phases")), row.get("disease_drug_relset"))
    else:
        a = action_score_candidate(first_rel, row.get("action_type"), row.get("drug_gene_relset"), row.get("drug_gene_action_types"))

        if str(first_rel or "").lower() == "target":
            # target is broad; rescue only with target-context/candidate anchor support
            if target_gene in mechanism_anchors:
                a = max(a, 1.6)
            elif disease_gene in mechanism_anchors:
                a = max(a, 1.3)
            elif target_gene in disease_gene_set or disease_gene in disease_gene_set:
                a = max(a, 1.1)
            else:
                a = min(a, 0.8)

        score += 2.0 * a

    # anchor support
    if target_gene in mechanism_anchors:
        score += 1.8
    elif target_gene in disease_gene_set:
        score += 0.5

    if disease_gene in mechanism_anchors:
        score += 0.8
    elif disease_gene in disease_gene_set:
        score += 0.3

    # biology bridge support
    lname_bridge = normalize_name(bridge_name)
    
    if bridge_type in {"BP", "PATH"}:
        tm = term_match_bonus(bridge_name, anchor_terms)
        if ENABLE_VERIFIER_EVIDENCE_HIERARCHY and pattern == "bio_bridge":
            row["verification_evidence_class"] = "indirect_biology_bridge"
            # An unmatched shared biology node is supporting context, not strong
            # target-disease verification. Matched disease/phenotype terms retain
            # positive support; unmatched bridges are conservatively downweighted.
            if tm > 0:
                score += 0.5 + tm
            else:
                score += 0.0
        else:
            score += 1.0 + tm
    
    elif bridge_type == "PHENO":
        # phenotype bridge can be useful in sparse infectious diseases.
        score += 0.45
        score += 0.5 * term_match_bonus(bridge_name, anchor_terms)

    # hop appropriateness
    hop_len = int(row.get("hop_len") or 0)
    if 2 <= hop_len <= 6:
        score += 0.5
    elif hop_len == 1:
        score -= 1.0
    elif hop_len > 6:
        score -= 0.3

    # generic penalties
    score -= generic_bridge_penalty(bridge_name)

    score -= structural_degree_penalty(row)

    # Translational properties are excluded from graph verification scores.
    # They are applied once, globally, after candidate generation and rescue.

    return round(score, 4)


def verification_queries() -> Dict[str, str]:
    drug_gene_rel_union = "|".join(DRUG_GENE_RELS)
    prohibited_rels = ", ".join([f"'{x}'" for x in PROHIBITED_PATH_RELS])

    common_where = f"""
    ALL(n IN nodes(p)
        WHERE SINGLE(m IN nodes(p) WHERE elementId(m) = elementId(n)))
    AND NONE(r IN relationships(p)
             WHERE type(r) IN [{prohibited_rels}])
    AND NONE(n IN nodes(p)
             WHERE any(lbl IN labels(n) WHERE lbl IN ['EXPO']))
    """

    common_return = """
    RETURN
      dr.name AS candidate_drug,
      type(r0) AS first_rel,
      r0.Action_Type AS action_type,
      [(dr)-[rr]-(g1) WHERE type(rr) IN ['target', 'inhibition', 'activation', 'binding', 'modulation', 'enzyme', 'transporter', 'carrier'] | type(rr)] AS drug_gene_relset,
      [x IN [(dr)-[rr]-(g1) WHERE type(rr) IN ['target', 'inhibition', 'activation', 'binding', 'modulation', 'enzyme', 'transporter', 'carrier'] | rr.Action_Type] WHERE x IS NOT NULL] AS drug_gene_action_types,
      g1.name AS target_gene,
      disease_gene.name AS disease_gene,
      bridge_node_name,
      bridge_node_type,
      length(p) AS hop_len,
      [n IN nodes(p) | coalesce(n.name, n.node_id, elementId(n))] AS path_nodes,
      [n IN nodes(p) | labels(n)] AS path_labels,
      [r IN relationships(p) | type(r)] AS path_rels,
      dr.Group AS `Group`,
      dr.Max_Phase AS MaxPhase,
      dr.Black_Box AS BlackBox,
      dr.Withdrawn_Flag AS Withdrawn,
      dr.Inorganic_Flag AS Inorganic,
      dr.Molecular_Weight AS MW,
      dr.Polar_Surface_Area AS TPSA,
      coalesce(dr.CX_LogP, dr.XLogP, dr.CLogP, dr.AlogP) AS LogP,
      dr.QED_Weighted AS QED,
      size([(g1)-[:associated_with]-(x:DISEASE) | x]) AS target_gene_disease_degree,
      size([(g1)-[:ppi]-(x:GENE) | x]) AS target_gene_ppi_degree,
      size([(dr)-[rr]-(x:GENE)
            WHERE type(rr) IN ['target', 'inhibition', 'activation', 'binding', 'modulation', 'enzyme', 'transporter', 'carrier'] | x]) AS drug_gene_degree,
      disease_gene_degree,
      bridge_gene_degree,
      phenotype_disease_degree
    ORDER BY
      candidate_drug ASC,
      first_rel ASC,
      target_gene ASC,
      bridge_node_name ASC,
      bridge_node_type ASC,
      hop_len ASC
    """

    return {
        "gene_gene": f"""
        MATCH (dr:DRUG)
        WHERE toLower(dr.name) = toLower($drug)

        MATCH (dis:DISEASE)
        WHERE toLower(dis.name) = toLower($disease)

        MATCH p =
          (dr)-[r0:{drug_gene_rel_union}]-
          (g1:GENE)
          -[:ppi]-
          (disease_gene:GENE)
          -[:associated_with]-
          (dis)

        WITH dr, r0, g1, disease_gene, p,
             null AS bridge_node_name,
             'GENE_GENE' AS bridge_node_type,
             size([(disease_gene)-[:associated_with]-(x:DISEASE) | x]) AS disease_gene_degree,
             0 AS bridge_gene_degree,
             0 AS phenotype_disease_degree
        WHERE {common_where}
        {common_return}
        LIMIT $limit
        """,

        "bio_bridge": f"""
        MATCH (dr:DRUG)
        WHERE toLower(dr.name) = toLower($drug)

        MATCH (dis:DISEASE)
        WHERE toLower(dis.name) = toLower($disease)

        MATCH p =
          (dr)-[r0:{drug_gene_rel_union}]-
          (g1:GENE)
          -[:interacts_with]-
          (bio:BP|PATH)
          -[:interacts_with]-
          (disease_gene:GENE)
          -[:associated_with]-
          (dis)

        WITH dr, r0, g1, disease_gene, bio, p,
             bio.name AS bridge_node_name,
             CASE
               WHEN 'BP' IN labels(bio) THEN 'BP'
               WHEN 'PATH' IN labels(bio) THEN 'PATH'
               ELSE head(labels(bio))
             END AS bridge_node_type,
             size([(disease_gene)-[:associated_with]-(x:DISEASE) | x]) AS disease_gene_degree,
             size([(bio)-[:interacts_with]-(x:GENE) | x]) AS bridge_gene_degree,
             0 AS phenotype_disease_degree
        WHERE {common_where}
          AND NOT toLower(coalesce(bridge_node_name, '')) IN $generic_bio_terms
        {common_return}
        LIMIT $limit
        """,

        "pheno_gene_bridge": f"""
        MATCH (dr:DRUG)
        WHERE toLower(dr.name) = toLower($drug)

        MATCH (dis:DISEASE)
        WHERE toLower(dis.name) = toLower($disease)

        MATCH p =
          (dr)-[r0:{drug_gene_rel_union}]-
          (g1:GENE)
          -[:ppi]-
          (disease_gene:GENE)
          -[:associated_with]-
          (ph:PHENO)
          -[:phenotype_present]-
          (dis)

        WITH dr, r0, g1, disease_gene, ph, p,
             ph.name AS bridge_node_name,
             'PHENO' AS bridge_node_type,
             size([(disease_gene)-[:associated_with]-(x:DISEASE) | x]) AS disease_gene_degree,
             0 AS bridge_gene_degree,
             size([(ph)-[:phenotype_present]-(x:DISEASE) | x]) AS phenotype_disease_degree
        WHERE {common_where}
          AND NOT toLower(coalesce(bridge_node_name, '')) IN $generic_bio_terms
        {common_return}
        LIMIT $limit
        """,

        "sibling_gene_action": f"""
        MATCH (dr:DRUG)
        WHERE toLower(dr.name) = toLower($drug)
        MATCH (dis:DISEASE)
        WHERE toLower(dis.name) = toLower($disease)
        MATCH p = (dr)-[r0:{drug_gene_rel_union}]-(g1:GENE)-[:associated_with]-(sib:DISEASE)-[:parent_child]->(parent:DISEASE)<-[:parent_child]-(dis)
        WHERE sib <> dis
          AND toLower(trim(sib.name)) <> toLower(trim(dis.name))
          AND toLower(trim(parent.name)) <> toLower(trim(dis.name))
        WITH dr, r0, g1, sib, parent, p,
             g1 AS disease_gene,
             sib.name AS bridge_node_name,
             'SIBLING_DISEASE_GENE' AS bridge_node_type,
             size([(g1)-[:associated_with]-(x:DISEASE) | x]) AS target_gene_disease_degree,
             size([(g1)-[:associated_with]-(x:DISEASE) | x]) AS disease_gene_degree,
             0 AS bridge_gene_degree,
             0 AS phenotype_disease_degree,
             size([(parent)<-[:parent_child]-(x:DISEASE) | x]) AS family_child_degree
        WHERE {common_where}
        {common_return}
        LIMIT $limit
        """,

        "family_label_edge": f"""
        MATCH (dr:DRUG)
        WHERE toLower(dr.name) = toLower($drug)

        MATCH (dis:DISEASE)
        WHERE toLower(dis.name) = toLower($disease)

        MATCH p =
          (dr)<-[r0:indication|off_label_use|tested_indication|studied_for_treatment_of]-(fam:DISEASE)-[:parent_child]->(parent:DISEASE)<-[:parent_child]-(dis)
        WHERE fam <> dis
          AND toLower(trim(fam.name)) <> toLower(trim(dis.name))
          AND toLower(trim(parent.name)) <> toLower(trim(dis.name))

        WITH dr, r0, fam, parent, dis, p,
             fam.name AS bridge_node_name,
             'DISEASE_LABEL_FAMILY' AS bridge_node_type,
             null AS target_gene,
             null AS disease_gene
        WHERE toLower(fam.name) <> toLower(dis.name)
          AND ALL(n IN nodes(p)
              WHERE SINGLE(m IN nodes(p) WHERE elementId(m) = elementId(n)))
        RETURN
          dr.name AS candidate_drug,
          type(r0) AS first_rel,
          r0.Max_Phase AS relation_max_phase,
          [(fam)-[rr]-(dr) WHERE type(rr) IN ['indication', 'off_label_use', 'tested_indication', 'studied_for_treatment_of', 'contraindication'] | type(rr)] AS disease_drug_relset,
          [x IN [(fam)-[rr:tested_indication]-(dr) | rr.Max_Phase] WHERE x IS NOT NULL] AS tested_indication_phases,
          target_gene AS target_gene,
          disease_gene AS disease_gene,
          bridge_node_name,
          bridge_node_type,
          parent.name AS family_parent_name,
          length(p) AS hop_len,
          [n IN nodes(p) | coalesce(n.name, n.node_id, elementId(n))] AS path_nodes,
          [n IN nodes(p) | labels(n)] AS path_labels,
          [r IN relationships(p) | type(r)] AS path_rels,
          dr.Group AS `Group`,
          dr.Max_Phase AS MaxPhase,
          dr.Black_Box AS BlackBox,
          dr.Withdrawn_Flag AS Withdrawn,
          dr.Inorganic_Flag AS Inorganic,
          dr.Molecular_Weight AS MW,
          dr.Polar_Surface_Area AS TPSA,
          coalesce(dr.CX_LogP, dr.XLogP, dr.CLogP, dr.AlogP) AS LogP,
          dr.QED_Weighted AS QED,
          size([(parent)<-[:parent_child]-(x:DISEASE) | x]) AS family_child_degree
        ORDER BY
          CASE type(r0)
            WHEN 'indication' THEN 0
            WHEN 'off_label_use' THEN 1
            WHEN 'tested_indication' THEN 2
            ELSE 3
          END,
          hop_len ASC,
          bridge_node_name ASC
        LIMIT $limit
        """
    }


def verify_candidate_drug(
    driver,
    disease: str,
    drug: str,
    anchor_genes: List[str],
    anchor_terms: List[str],
    mechanism_anchor_genes: Optional[List[str]] = None,
    disease_genes: Optional[List[str]] = None,
    limit_per_pattern: int = PATH_LIMIT_PER_PATTERN,
) -> Dict[str, Any]:
    queries = verification_queries()
    enabled_patterns = enabled_prior_verify_patterns_for_current_case(disease)
    if enabled_patterns:
        allowed = set(enabled_patterns)
        queries = {k: v for k, v in queries.items() if k in allowed}

    all_paths = []
    pattern_timing = []

    for pattern_name, q in queries.items():
        t0 = time.time()
        try:
            rows = run_query_auto(
                driver,
                q,
                {
                    "disease": disease,
                    "drug": drug,
                    "limit": limit_per_pattern,
                    "generic_bio_terms": list(GENERIC_BIO_TERMS),
                    "family_label_broad_context_excludes": list(FAMILY_LABEL_BROAD_CONTEXT_EXCLUDES),
                },
                timeout_sec=VERIFY_TIMEOUT_SEC,
            )
        except Exception as e:
            rows = []
            print(f"[VERIFY ERROR] disease={disease} drug={drug} pattern={pattern_name}: {type(e).__name__}: {e}")

        elapsed = time.time() - t0
        pattern_timing.append({
            "pattern": pattern_name,
            "seconds": round(elapsed, 3),
            "n_rows": len(rows),
        })
        if elapsed >= 5.0:
            print(
                f"[VERIFY TIMING] disease={disease} drug={drug} "
                f"pattern={pattern_name} rows={len(rows)} sec={elapsed:.1f}",
                flush=True,
            )

        rows = canonicalize_path_rows(rows, pattern_name)
        for row in rows:
            if pattern_name == "sibling_gene_action" and has_only_adme_drug_gene_evidence(row):
                continue
            row["pattern"] = pattern_name
            row["path_score"] = score_verified_path(
                row,
                anchor_genes=anchor_genes,
                anchor_terms=anchor_terms,
                mechanism_anchor_genes=mechanism_anchor_genes or [],
                disease_genes=disease_genes or [],
            )
            all_paths.append(row)

    all_paths = filter_family_only_evidence_paths(all_paths)
    all_paths.sort(
        key=lambda r: (
            r.get("path_score", 0.0),
            action_score_candidate(r.get("first_rel"), r.get("action_type"), r.get("drug_gene_relset"), r.get("drug_gene_action_types")),
            -int(r.get("hop_len") or 99),
        ),
        reverse=True,
    )

    best = all_paths[0] if all_paths else None

    return {
        "drug": drug,
        "path_found": bool(best),
        "n_paths": len(all_paths),
        "best_score": best.get("path_score") if best else None,
        "best_pattern": best.get("pattern") if best else None,
        "best_target_gene": best.get("target_gene") if best else None,
        "best_bridge_node": best.get("bridge_node_name") if best else None,
        "best_bridge_type": best.get("bridge_node_type") if best else None,
        "best_disease_gene": best.get("disease_gene") if best else None,
        "best_hop_len": best.get("hop_len") if best else None,
        "best_path_nodes": best.get("path_nodes") if best else [],
        "best_path_rels": best.get("path_rels") if best else [],
        "best_verification_evidence_class": best.get("verification_evidence_class") if best else None,
        "top_paths": all_paths[:5],
        "pattern_timing": pattern_timing,
    }


def _norm_required_anchor_set(items: Optional[List[str]]) -> set:
    return {normalize_name(x) for x in (items or []) if str(x or "").strip()}


def _row_matches_required_vector_anchor(
    row: Dict[str, Any],
    required_anchor_genes: Optional[List[str]] = None,
    required_anchor_terms: Optional[List[str]] = None,
) -> bool:
    """Return True if a verified path explicitly passes through vector-neighbor anchors.

    This implements branch (3): target disease -- vector-neighbor anchor -- prior drug seed.
    The prior drug is not accepted merely because it came from a similar disease; the
    target-disease path must contain at least one vector-neighbor gene/BP/PATH/PHENO anchor.
    """
    gene_set = _norm_required_anchor_set(required_anchor_genes)
    term_set = _norm_required_anchor_set(required_anchor_terms)

    if not gene_set and not term_set:
        return False

    path_nodes = [normalize_name(x) for x in (row.get("path_nodes") or [])]
    target_gene = normalize_name(row.get("target_gene"))
    disease_gene = normalize_name(row.get("disease_gene"))
    bridge_node = normalize_name(row.get("bridge_node_name"))

    if gene_set:
        if target_gene in gene_set or disease_gene in gene_set:
            return True
        if any(n in gene_set for n in path_nodes):
            return True

    if term_set:
        if bridge_node in term_set:
            return True
        if any(n in term_set for n in path_nodes):
            return True

    return False


def verify_candidate_drug_anchor_constrained(
    driver,
    disease: str,
    drug: str,
    anchor_genes: List[str],
    anchor_terms: List[str],
    mechanism_anchor_genes: Optional[List[str]] = None,
    disease_genes: Optional[List[str]] = None,
    limit_per_pattern: int = PATH_LIMIT_PER_PATTERN,
    require_vector_anchor_evidence: bool = False,
    required_anchor_genes: Optional[List[str]] = None,
    required_anchor_terms: Optional[List[str]] = None,
    verification_source: str = "prior_seed_general_graph_verified",
    prior_family_support_diseases: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Verify a fixed prior drug seed against the target disease.

    Two modes:
      - require_vector_anchor_evidence=False: branch (2), drug seed -> target disease path.
      - require_vector_anchor_evidence=True: branch (3), drug seed -> vector-neighbor anchor -> target disease path.

    This differs from GraphDRx because prior drugs are not promoted unless a target-disease
    KG path is found. In constrained mode, the path must explicitly touch a vector-neighbor
    gene/BP/PATH/PHENO anchor.
    """
    queries = verification_queries()
    enabled_patterns = enabled_prior_verify_patterns_for_current_case(disease)
    if enabled_patterns:
        allowed = set(enabled_patterns)
        queries = {k: v for k, v in queries.items() if k in allowed}
    all_paths = []
    pattern_timing = []
    n_paths_before_filter = 0

    required_anchor_genes = required_anchor_genes or []
    required_anchor_terms = required_anchor_terms or []
    prior_family_support_diseases = unique_keep_order(prior_family_support_diseases or [])

    for pattern_name, q in queries.items():
        t0 = time.time()
        try:
            rows = run_query_auto(
                driver,
                q,
                {
                    "disease": disease,
                    "drug": drug,
                    "limit": limit_per_pattern,
                    "generic_bio_terms": list(GENERIC_BIO_TERMS),
                    "family_label_broad_context_excludes": list(FAMILY_LABEL_BROAD_CONTEXT_EXCLUDES),
                },
                timeout_sec=VERIFY_TIMEOUT_SEC,
            )
        except Exception as e:
            rows = []
            print(f"[VERIFY ERROR] disease={disease} drug={drug} pattern={pattern_name}: {type(e).__name__}: {e}")

        elapsed = time.time() - t0
        pattern_timing.append({
            "pattern": pattern_name,
            "seconds": round(elapsed, 3),
            "n_rows": len(rows),
        })
        if elapsed >= 5.0:
            print(
                f"[VERIFY TIMING] disease={disease} drug={drug} "
                f"pattern={pattern_name} rows={len(rows)} sec={elapsed:.1f}",
                flush=True,
            )

        rows = canonicalize_path_rows(rows, pattern_name)
        n_paths_before_filter += len(rows)

        for row in rows:
            if pattern_name == "sibling_gene_action" and has_only_adme_drug_gene_evidence(row):
                continue
            row["pattern"] = pattern_name
            row["verification_source"] = verification_source
            if pattern_name in FAMILY_EVIDENCE_PATTERNS and prior_family_support_diseases:
                row["prior_family_support_diseases"] = list(prior_family_support_diseases)
            row["path_has_required_vector_anchor"] = _row_matches_required_vector_anchor(
                row,
                required_anchor_genes=required_anchor_genes,
                required_anchor_terms=required_anchor_terms,
            )

            if require_vector_anchor_evidence and not row["path_has_required_vector_anchor"]:
                continue

            row["path_score"] = score_verified_path(
                row,
                anchor_genes=anchor_genes,
                anchor_terms=anchor_terms,
                mechanism_anchor_genes=mechanism_anchor_genes or [],
                disease_genes=disease_genes or [],
            )
            # Small evidence bonus for the stricter path class. This only affects ordering
            # among graph-verified paths; it does not promote path-free prior drugs.
            if require_vector_anchor_evidence:
                row["path_score"] = round(float(row.get("path_score") or 0.0) + 0.35, 6)
            all_paths.append(row)

    all_paths = filter_family_only_evidence_paths(all_paths)
    all_paths.sort(
        key=lambda r: (
            r.get("path_score", 0.0),
            bool(r.get("path_has_required_vector_anchor")),
            action_score_candidate(r.get("first_rel"), r.get("action_type"), r.get("drug_gene_relset"), r.get("drug_gene_action_types")),
            -int(r.get("hop_len") or 99),
        ),
        reverse=True,
    )

    best = all_paths[0] if all_paths else None

    return {
        "drug": drug,
        "path_found": bool(best),
        "n_paths": len(all_paths),
        "n_paths_before_filter": n_paths_before_filter,
        "require_vector_anchor_evidence": bool(require_vector_anchor_evidence),
        "verification_source": verification_source,
        "required_anchor_gene_count": len(required_anchor_genes or []),
        "required_anchor_term_count": len(required_anchor_terms or []),
        "prior_family_support_diseases": list(prior_family_support_diseases),
        "prior_family_support_count": len(prior_family_support_diseases),
        "best_score": best.get("path_score") if best else None,
        "best_pattern": best.get("pattern") if best else None,
        "best_target_gene": best.get("target_gene") if best else None,
        "best_bridge_node": best.get("bridge_node_name") if best else None,
        "best_bridge_type": best.get("bridge_node_type") if best else None,
        "best_disease_gene": best.get("disease_gene") if best else None,
        "best_hop_len": best.get("hop_len") if best else None,
        "best_path_nodes": best.get("path_nodes") if best else [],
        "best_path_rels": best.get("path_rels") if best else [],
        "best_verification_evidence_class": best.get("verification_evidence_class") if best else None,
        "best_family_evidence_independent_supports": best.get("family_evidence_independent_supports") if best else None,
        "best_family_evidence_required_supports": best.get("family_evidence_required_supports") if best else None,
        "best_family_evidence_support_diseases": best.get("family_evidence_support_diseases", []) if best else [],
        "best_family_evidence_prior_support_diseases": best.get("family_evidence_prior_support_diseases", []) if best else [],
        "best_family_evidence_prior_corroboration_units": best.get("family_evidence_prior_corroboration_units") if best else None,
        "best_family_parent_name": best.get("family_parent_name") if best else None,
        "best_family_evidence_cluster_key": best.get("family_evidence_cluster_key") if best else None,
        "best_family_evidence_role": best.get("family_evidence_role") if best else None,
        "best_path_has_required_vector_anchor": bool(best.get("path_has_required_vector_anchor")) if best else False,
        "top_paths": all_paths[:5],
        "pattern_timing": pattern_timing,
    }





# ============================================================
# 6.4 Vector-guided GraphRAG anchor expansion
# ============================================================
def get_vector_similar_disease_kg_context(
    driver,
    similar_diseases: List[Dict[str, Any]],
    top_n: int = 20,
    max_genes: int = 80,
    max_terms: int = 100,
    case_area: str = "",
) -> Dict[str, Any]:
    """Retrieve an ordered, provenance-preserving vector-neighbor anchor pool.

    Similar-disease rank is preserved explicitly; Neo4j row order is never used as
    an implicit proxy for embedding similarity. No near-disease drug relation is
    read in this branch.
    """
    near_rows = []
    seen = set()
    for rank, row in enumerate(similar_diseases or [], start=1):
        name = str(row.get("disease") or "").strip()
        key = normalize_name(name)
        if not name or key in seen:
            continue
        seen.add(key)
        near_rows.append({
            "name": name,
            "rank": rank,
            "similarity": safe_float(row.get("similarity_score"), 0.0),
        })
        if len(near_rows) >= max(int(top_n or 0), 0):
            break

    if not near_rows:
        return {
            "enabled": True,
            "near_diseases": [],
            "near_disease_rows": [],
            "vector_genes": [],
            "vector_bio_terms": [],
            "vector_phenotypes": [],
            "vector_pathways": [],
            "gene_source_diseases": {},
            "bio_term_source_diseases": {},
            "pathway_source_diseases": {},
            "phenotype_source_diseases": {},
            "source": "vector_similar_disease_kg_context",
        }

    q = """
    UNWIND $near_rows AS nr
    MATCH (near:DISEASE)
    WHERE near.name = nr.name

    OPTIONAL MATCH (near)-[:associated_with]-(g:GENE)
    WITH nr, near, collect(DISTINCT g.name) AS genes

    OPTIONAL MATCH (near)-[:phenotype_present]-(ph:PHENO)
    WITH nr, near, genes, collect(DISTINCT ph.name) AS phenotypes

    OPTIONAL MATCH (near)-[:associated_with]-(g2:GENE)-[:interacts_with]-(bio:BP|PATH)
    WHERE bio.name IS NOT NULL
      AND NOT toLower(bio.name) IN $generic_bio_terms
    WITH nr, near, genes, phenotypes,
         collect(DISTINCT CASE WHEN 'BP' IN labels(bio) THEN bio.name ELSE NULL END) AS bp_terms,
         collect(DISTINCT CASE WHEN 'PATH' IN labels(bio) THEN bio.name ELSE NULL END) AS path_terms

    RETURN nr.rank AS similarity_rank,
           nr.similarity AS similarity_score,
           near.name AS disease,
           genes, phenotypes, bp_terms, path_terms
    ORDER BY similarity_rank ASC, disease ASC
    """
    rows = run_query_auto(
        driver,
        q,
        {
            "near_rows": near_rows,
            "max_genes": int(max_genes),
            "max_terms": int(max_terms),
            "generic_bio_terms": list(GENERIC_BIO_TERMS),
        },
        timeout_sec=180,
    )
    rows = sorted(
        rows or [],
        key=lambda r: (int(r.get("similarity_rank") or 10**9), normalize_name(r.get("disease"))),
    )

    genes: List[str] = []
    phenotypes: List[str] = []
    bp_terms: List[str] = []
    path_terms: List[str] = []
    gene_sources: Dict[str, List[str]] = defaultdict(list)
    phenotype_sources: Dict[str, List[str]] = defaultdict(list)
    bp_sources: Dict[str, List[str]] = defaultdict(list)
    path_sources: Dict[str, List[str]] = defaultdict(list)
    use_vector_phenotypes = bool(structural_area_option(case_area, "use_vector_phenotypes", False))

    for row in rows:
        source_disease = str(row.get("disease") or "").strip()
        for value in row.get("genes") or []:
            gene = normalize_mechanism_gene(value)
            if not gene:
                continue
            genes.append(gene)
            if source_disease and source_disease not in gene_sources[gene]:
                gene_sources[gene].append(source_disease)
        if use_vector_phenotypes:
            for value in row.get("phenotypes") or []:
                term = str(value or "").strip()
                if not term:
                    continue
                phenotypes.append(term)
                key = normalize_name(term)
                if source_disease and source_disease not in phenotype_sources[key]:
                    phenotype_sources[key].append(source_disease)
        for value in row.get("bp_terms") or []:
            term = str(value or "").strip()
            if not term or normalize_name(term) in GENERIC_BIO_TERMS:
                continue
            bp_terms.append(term)
            key = normalize_name(term)
            if source_disease and source_disease not in bp_sources[key]:
                bp_sources[key].append(source_disease)
        for value in row.get("path_terms") or []:
            term = str(value or "").strip()
            if not term or normalize_name(term) in GENERIC_BIO_TERMS:
                continue
            path_terms.append(term)
            key = normalize_name(term)
            if source_disease and source_disease not in path_sources[key]:
                path_sources[key].append(source_disease)

    # Rank anchors without gold labels or disease-name rules. The strongest
    # embedding-neighbor support is primary, distinct-neighbor multiplicity is
    # secondary, and lexical order is the deterministic final tie-breaker.
    sim_by_disease = {normalize_name(r["name"]): safe_float(r.get("similarity"), 0.0) for r in near_rows}
    rank_by_disease = {normalize_name(r["name"]): int(r.get("rank") or 10**9) for r in near_rows}

    def rank_anchor_values(values, source_map, normalizer, cap):
        display = {}
        for value in values or []:
            key = normalizer(value)
            if key and key not in display:
                display[key] = str(value).strip()
        ranked = []
        for key, shown in display.items():
            sources = unique_keep_order(source_map.get(key, []))
            max_similarity = max((sim_by_disease.get(normalize_name(d), 0.0) for d in sources), default=0.0)
            best_rank = min((rank_by_disease.get(normalize_name(d), 10**9) for d in sources), default=10**9)
            ranked.append((-max_similarity, -len(sources), best_rank, normalize_name(shown), shown))
        ranked.sort()
        return [row[-1] for row in ranked[: max(int(cap or 0), 0)]]

    genes = rank_anchor_values(genes, gene_sources, normalize_mechanism_gene, max_genes)
    phenotypes = rank_anchor_values(phenotypes, phenotype_sources, normalize_name, max_terms)
    bp_terms = rank_anchor_values(bp_terms, bp_sources, normalize_name, max_terms)
    path_terms = rank_anchor_values(path_terms, path_sources, normalize_name, max_terms)

    return {
        "enabled": True,
        "near_diseases": [row["name"] for row in near_rows],
        "near_disease_rows": near_rows,
        "vector_genes": genes,
        "vector_bio_terms": bp_terms,
        "vector_pathways": path_terms,
        "vector_phenotypes": phenotypes,
        "gene_source_diseases": {g: gene_sources.get(g, []) for g in genes},
        "bio_term_source_diseases": {normalize_name(t): bp_sources.get(normalize_name(t), []) for t in bp_terms},
        "pathway_source_diseases": {normalize_name(t): path_sources.get(normalize_name(t), []) for t in path_terms},
        "phenotype_source_diseases": {normalize_name(t): phenotype_sources.get(normalize_name(t), []) for t in phenotypes},
        "source": "vector_similar_disease_kg_context",
        "use_vector_phenotypes": use_vector_phenotypes,
        "ordering_policy": "max_neighbor_similarity_then_distinct_neighbor_support_then_neighbor_rank_then_name",
    }


def augment_mechanism_anchors_with_vector_context(
    driver,
    disease: str,
    mechanism_anchor_obj: Dict[str, Any],
    embedding_prior: Dict[str, Any],
    top_n: int = 20,
    max_genes: int = 80,
    max_terms: int = 100,
    case_area: str = "",
) -> Dict[str, Any]:
    """Attach a separate vector-neighbor anchor pool without mixing axis lists."""
    out = copy.deepcopy(mechanism_anchor_obj or {})
    ctx = get_vector_similar_disease_kg_context(
        driver,
        embedding_prior.get("similar_diseases") or [],
        top_n=top_n,
        max_genes=max_genes,
        max_terms=max_terms,
        case_area=case_area,
    )
    vector_genes = ctx.get("vector_genes") or []
    vector_bio = ctx.get("vector_bio_terms") or []
    vector_paths = ctx.get("vector_pathways") or []
    vector_phenotypes = ctx.get("vector_phenotypes") or []

    out["vector_anchor_pool"] = {
        "axis_name": "Vector-guided similar-disease KG neighborhood",
        "axis_rationale": (
            "Embedding-similar diseases supply ordered KG gene, BP, PATH, and "
            "area-gated phenotype anchors; their drug relations are not used in this branch."
        ),
        "priority": "high",
        "anchor_source": "disease_vector_neighbor_kg",
        "anchor_genes": vector_genes,
        "anchor_biology_terms": vector_bio,
        "anchor_pathways": vector_paths,
        "anchor_phenotypes": vector_phenotypes,
        "relevant_cell_types_or_tissues": [],
        "expected_drug_actions_or_classes": [],
        "near_diseases": ctx.get("near_diseases") or [],
        "anchor_source_maps": {
            "gene": ctx.get("gene_source_diseases") or {},
            "bio": ctx.get("bio_term_source_diseases") or {},
            "pathway": ctx.get("pathway_source_diseases") or {},
            "phenotype": ctx.get("phenotype_source_diseases") or {},
        },
    }

    # Keep a combined representation for prior-path scoring and drug-embedding
    # reranking, while preserving the two candidate-generation pools separately.
    ga = dict(out.get("global_anchors") or {})
    ga["target_genes"] = unique_keep_order((ga.get("target_genes") or []) + vector_genes)[:150]
    ga["biology_terms"] = unique_keep_order((ga.get("biology_terms") or []) + vector_bio)[:200]
    ga["pathways"] = unique_keep_order((ga.get("pathways") or []) + vector_paths)[:150]
    ga["phenotype_terms"] = unique_keep_order((ga.get("phenotype_terms") or []) + vector_phenotypes)[:100]
    ga["disease_family_names"] = unique_keep_order((ga.get("disease_family_names") or []) + [disease])[:60]
    out["global_anchors"] = ga
    out["vector_guided_graphrag"] = {
        "enabled": bool(vector_genes or vector_bio or vector_paths or vector_phenotypes),
        "mode": "separate_vector_anchor_branch",
        "top_n": top_n,
        "max_genes": max_genes,
        "max_terms": max_terms,
        **ctx,
    }
    return out


def build_vector_anchor_mechanism_obj(mechanism_anchor_obj: Dict[str, Any]) -> Dict[str, Any]:
    """Build the explicit branch-1-2 retrieval object from the vector anchor pool."""
    pool = copy.deepcopy((mechanism_anchor_obj or {}).get("vector_anchor_pool") or {})
    if not pool:
        return {"mechanism_axes": [], "global_anchors": {}, "vector_anchor_pool": {}}
    pool["selected_for_initial_graph"] = True
    pool["axis_selection_rank"] = 1
    pool["axis_selection_policy"] = "single_vector_neighbor_pool"
    return {
        "disease": mechanism_anchor_obj.get("disease"),
        "mechanism_axes": [pool],
        "global_anchors": {
            "target_genes": pool.get("anchor_genes") or [],
            "biology_terms": pool.get("anchor_biology_terms") or [],
            "pathways": pool.get("anchor_pathways") or [],
            "phenotype_terms": pool.get("anchor_phenotypes") or [],
            "disease_family_names": [],
        },
        "vector_anchor_pool": pool,
        "vector_guided_graphrag": mechanism_anchor_obj.get("vector_guided_graphrag") or {},
    }


# ============================================================
# 6.4 Prior-seeded GraphRAG verification
# ============================================================
def _verification_row_to_candidate(
    v: Dict[str, Any],
    prior_row: Dict[str, Any],
    prior_rank: int,
) -> Dict[str, Any]:
    """Convert a verified prior-drug path into the normal candidate schema.

    This is the key GraphDRx change: disease embedding is used only as a
    candidate-generation seed. The candidate is promoted into the main list only
    if a drug -> target/action -> biology/gene -> disease GraphRAG path is found.
    """
    drug = prior_row.get("drug") or v.get("drug")
    best_score = safe_float(v.get("best_score"), 0.0)
    emb_score = safe_float(prior_row.get("embedding_prior_score"), 0.0)
    path_nodes = v.get("best_path_nodes") or []
    path_rels = v.get("best_path_rels") or []

    # Map verification row to the best_anchor_path schema used elsewhere.
    bridge_name = v.get("best_bridge_node")
    bridge_type = v.get("best_bridge_type")
    bp = {
        "pattern": v.get("best_pattern"),
        "first_rel": None,
        "target_gene": v.get("best_target_gene"),
        "anchor_name": bridge_name or v.get("best_disease_gene"),
        "anchor_type": bridge_type or "GENE_GENE",
        "disease_gene": v.get("best_disease_gene"),
        "hop_len": v.get("best_hop_len"),
        "path_nodes": path_nodes,
        "path_rels": path_rels,
        "path_score": best_score,
        "verification_source": "prior_seed_graph_verification",
        "family_evidence_independent_supports": v.get("best_family_evidence_independent_supports"),
        "family_evidence_required_supports": v.get("best_family_evidence_required_supports"),
        "family_evidence_support_diseases": v.get("best_family_evidence_support_diseases", []),
        "family_evidence_prior_support_diseases": v.get("best_family_evidence_prior_support_diseases", []),
        "family_evidence_prior_corroboration_units": v.get("best_family_evidence_prior_corroboration_units"),
        "family_parent_name": v.get("best_family_parent_name"),
        "family_evidence_cluster_key": v.get("best_family_evidence_cluster_key"),
        "family_evidence_role": v.get("best_family_evidence_role"),
    }
    if path_rels:
        bp["first_rel"] = path_rels[0]

    # This stage stores a graph-only candidate score. The semantic disease prior
    # is attached exactly once later by merge_graph_candidates_with_embedding_priors.
    source_label = v.get("verification_source") or "prior_seed_general_graph_verified"
    vector_anchor_verified = bool(v.get("best_path_has_required_vector_anchor"))
    candidate_score = round((0.85 * best_score) + 0.35 + (0.20 if vector_anchor_verified else 0.0), 6)

    return {
        "drug": drug,
        "rank": None,
        "confidence": "verified_prior_seed",
        "mechanism_axis": source_label,
        "suggested_targets": unique_keep_order([v.get("best_target_gene"), v.get("best_disease_gene")]),
        "suggested_biology_terms": unique_keep_order([bridge_name] if bridge_type in {"BP", "PATH"} else []),
        "rationale": (
            "Disease-embedding prior was used as a GraphRAG seed and re-verified "
            f"against the target disease. pattern={v.get('best_pattern')}; "
            f"target={v.get('best_target_gene')}; bridge={bridge_name}; "
            f"path_score={best_score}; prior_score={emb_score}."
        ),
        "candidate_score": candidate_score,
        "candidate_source": source_label,
        "candidate_sources": [source_label],
        "best_anchor_path": bp,
        "prior_seed_verification_source": source_label,
        "source_prior_seed_general_verified": source_label == "prior_seed_general_graph_verified",
        "source_prior_seed_vector_anchor_verified": source_label == "prior_seed_vector_anchor_graph_verified",
        "source_vector_anchor_overlap": vector_anchor_verified,
        "best_path_score": best_score,
        "source_graph_candidate": True,
        "source_embedding_prior": True,
        "source_prior_graph_verified": True,
        "source_prior_drug_target_graph": True,
        "prior_seed_rank": prior_rank,
        "embedding_prior_score": emb_score,
        "relation_component": prior_row.get("relation_component"),
        "relation_component_mode": prior_row.get("relation_component_mode"),
        "relation_component_weights": prior_row.get("relation_component_weights", []),
        "similarity_component": prior_row.get("similarity_component"),
        "count_component": prior_row.get("count_component"),
        "prior_diseases": prior_row.get("prior_diseases", []),
        "prior_rels": prior_row.get("prior_rels", []),
        "prior_primary_support_diseases": v.get("prior_primary_support_diseases", []),
        "prior_family_support_count": v.get("prior_family_support_count", 0),
        "family_evidence_independent_supports": v.get("best_family_evidence_independent_supports"),
        "family_evidence_required_supports": v.get("best_family_evidence_required_supports"),
        "family_evidence_support_diseases": v.get("best_family_evidence_support_diseases", []),
        "family_evidence_prior_support_diseases": v.get("best_family_evidence_prior_support_diseases", []),
        "family_evidence_prior_corroboration_units": v.get("best_family_evidence_prior_corroboration_units"),
        "family_parent_name": v.get("best_family_parent_name"),
        "family_evidence_cluster_key": v.get("best_family_evidence_cluster_key"),
        "family_evidence_role": v.get("best_family_evidence_role"),
        "contraindication_prior_hit": prior_row.get("contraindication_prior_hit", False),
        "Group": prior_row.get("Group"),
        "MaxPhase": prior_row.get("drug_max_phase", prior_row.get("MaxPhase")),
        "BlackBox": prior_row.get("BlackBox"),
        "Withdrawn": prior_row.get("Withdrawn"),
        "Inorganic": prior_row.get("Inorganic"),
        "MW": prior_row.get("MW"),
        "TPSA": prior_row.get("TPSA"),
        "LogP": prior_row.get("LogP"),
        "QED": prior_row.get("QED"),
    }


def verify_embedding_prior_candidates_as_graphrag_seeds(
    driver,
    disease: str,
    embedding_prior: Dict[str, Any],
    mechanism_obj: Dict[str, Any],
    disease_context: Dict[str, Any],
    top_n: int = 80,
    limit_per_pattern: int = 8,
    skip_existing_graph_drugs: bool = True,
    require_vector_anchor_evidence: bool = False,
    required_anchor_genes: Optional[List[str]] = None,
    required_anchor_terms: Optional[List[str]] = None,
    verification_source_label: str = "prior_seed_general_graph_verified",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Re-query GraphRAG for disease-prior drugs and keep only path-verified hits.

    Without this, source_embedding_prior=True / source_graph_candidate=False drugs are
    not GraphRAG candidates. GraphDRx uses the disease prior as a seed list and then
    verifies each drug-disease pair through bounded GraphRAG patterns.
    """
    priors = embedding_prior.get("candidate_priors") or []
    graph_keys = {normalize_name(c.get("drug")) for c in mechanism_obj.get("candidate_drugs", []) if c.get("drug")}

    # Prior-drug verification is an independent branch. Acceptance and path scoring
    # use target-disease graph context, not target-context/vector anchors. This keeps the
    # disease-embedding drug-seed branch separable from the two anchor branches.
    disease_genes = disease_context.get("disease_genes", []) or []
    use_prior_phenotypes = bool(
        structural_area_option(CURRENT_CASE_AREA, "use_prior_verification_phenotypes", False)
    )
    phenotypes = (disease_context.get("phenotypes", []) or []) if use_prior_phenotypes else []
    anchor_genes = unique_keep_order(disease_genes)
    anchor_terms = unique_keep_order(phenotypes + [disease])
    mechanism_anchor_genes: List[str] = []

    verified_candidates: List[Dict[str, Any]] = []
    verification_rows: List[Dict[str, Any]] = []
    n_tested = 0
    n_skipped_existing = 0

    for prior_rank, prior_row in enumerate(priors[:max(int(top_n or 0), 0)], start=1):
        drug = prior_row.get("drug")
        if not drug:
            continue
        key = normalize_name(drug)
        if skip_existing_graph_drugs and key in graph_keys:
            n_skipped_existing += 1
            continue

        n_tested += 1
        qualified_prior_diseases = primary_prior_support_diseases(prior_row)
        v = verify_candidate_drug_anchor_constrained(
            driver=driver,
            disease=disease,
            drug=drug,
            anchor_genes=anchor_genes,
            mechanism_anchor_genes=mechanism_anchor_genes,
            disease_genes=disease_genes,
            anchor_terms=anchor_terms,
            limit_per_pattern=limit_per_pattern,
            require_vector_anchor_evidence=require_vector_anchor_evidence,
            required_anchor_genes=required_anchor_genes or [],
            required_anchor_terms=required_anchor_terms or [],
            verification_source=verification_source_label,
            prior_family_support_diseases=qualified_prior_diseases,
        )
        v["prior_seed_rank"] = prior_rank
        v["embedding_prior_score"] = prior_row.get("embedding_prior_score")
        v["prior_diseases"] = prior_row.get("prior_diseases", [])
        v["prior_rels"] = prior_row.get("prior_rels", [])
        v["prior_primary_support_diseases"] = qualified_prior_diseases
        verification_rows.append(v)

        print(
            f"  [PRIOR-SEED VERIFY] {verification_source_label} prior#{prior_rank} {drug}: "
            f"path_found={v.get('path_found')} | "
            f"best_score={v.get('best_score')} | "
            f"pattern={v.get('best_pattern')} | "
            f"target={v.get('best_target_gene')} | bridge={v.get('best_bridge_node')} | "
            f"vector_anchor={v.get('best_path_has_required_vector_anchor')} | "
            f"prior_support={len(qualified_prior_diseases)}",
            flush=True,
        )

        if v.get("path_found"):
            verified_candidates.append(_verification_row_to_candidate(v, prior_row, prior_rank))

    # Aggregate verifier timing by pattern for fast ablation decisions.
    pattern_timing_summary = {}
    for row in verification_rows:
        for t in row.get("pattern_timing", []) or []:
            pat = t.get("pattern")
            if not pat:
                continue
            cur = pattern_timing_summary.setdefault(pat, {"calls": 0, "seconds": 0.0, "rows": 0})
            cur["calls"] += 1
            cur["seconds"] += float(t.get("seconds") or 0.0)
            cur["rows"] += int(t.get("n_rows") or 0)
    for pat, cur in pattern_timing_summary.items():
        cur["seconds"] = round(cur["seconds"], 3)
        cur["sec_per_call"] = round(cur["seconds"] / max(cur["calls"], 1), 3)

    summary = {
        "enabled": True,
        "mode": verification_source_label,
        "top_n": top_n,
        "limit_per_pattern": limit_per_pattern,
        "configured_default_patterns": DEFAULT_PRIOR_VERIFY_PATTERNS,
        "effective_enabled_patterns": enabled_prior_verify_patterns_for_current_case(disease),
        "enabled_patterns": enabled_prior_verify_patterns_for_current_case(disease),
        "verify_timeout_sec": VERIFY_TIMEOUT_SEC,
        "require_vector_anchor_evidence": bool(require_vector_anchor_evidence),
        "required_anchor_gene_count": len(required_anchor_genes or []),
        "required_anchor_term_count": len(required_anchor_terms or []),
        "n_prior_candidates": len(priors),
        "n_tested": n_tested,
        "n_skipped_existing_graph_drugs": n_skipped_existing,
        "n_path_found": len(verified_candidates),
        "n_path_not_found": n_tested - len(verified_candidates),
        "pattern_timing_summary": pattern_timing_summary,
        "verified_rows": verification_rows[:200],
    }
    return verified_candidates, summary


def get_required_vector_anchor_sets_from_mechanism_obj(mechanism_anchor_obj: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Collect vector-neighbor anchors for branch (3) prior-seed verification."""
    vctx = (mechanism_anchor_obj or {}).get("vector_guided_graphrag") or {}
    genes = unique_keep_order([normalize_mechanism_gene(g) for g in (vctx.get("vector_genes") or []) if str(g or "").strip()])
    terms = unique_keep_order(
        (vctx.get("vector_bio_terms") or [])
        + (vctx.get("vector_pathways") or [])
        + (vctx.get("vector_phenotypes") or [])
    )
    terms = [t for t in terms if normalize_name(t) not in GENERIC_BIO_TERMS]
    return genes, terms


def deduplicate_candidates_keep_best(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate by drug while preserving evidence from every retrieval branch.

    The highest candidate_score remains the ranking base, but source flags, branch
    provenance, paths, axes, patterns, and prior metadata are merged rather than lost.
    """
    by_key: Dict[str, Dict[str, Any]] = {}
    list_fields = {
        "supporting_anchor_branches", "supporting_axes", "supporting_patterns",
        "suggested_targets", "suggested_biology_terms", "prior_diseases", "prior_rels",
        "family_evidence_support_diseases", "family_evidence_prior_support_diseases",
    }
    bool_prefixes = ("source_",)

    def values_as_list(v: Any) -> List[Any]:
        if v is None:
            return []
        if isinstance(v, (list, tuple, set)):
            return list(v)
        return [v]

    def merge_rows(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        # Keep the stronger row as the scalar/base representation.
        if safe_float(b.get("candidate_score"), 0.0) > safe_float(a.get("candidate_score"), 0.0):
            base, other = dict(b), a
        else:
            base, other = dict(a), b

        for field in list_fields:
            merged = unique_keep_order(values_as_list(a.get(field)) + values_as_list(b.get(field)))
            if merged:
                base[field] = merged

        for key in set(a) | set(b):
            if key.startswith(bool_prefixes):
                base[key] = bool(a.get(key)) or bool(b.get(key))

        paths = []
        for field in ("top_anchor_paths", "top_paths"):
            paths.extend(values_as_list(a.get(field)))
            paths.extend(values_as_list(b.get(field)))
        if a.get("best_anchor_path"):
            paths.append(a.get("best_anchor_path"))
        if b.get("best_anchor_path"):
            paths.append(b.get("best_anchor_path"))
        seen_paths = set()
        merged_paths = []
        for row in paths:
            if not isinstance(row, dict):
                continue
            key = canonical_path_key(row, str(row.get("pattern") or ""))
            if key in seen_paths:
                continue
            seen_paths.add(key)
            merged_paths.append(row)
        merged_paths.sort(key=lambda r: safe_float(r.get("path_score"), 0.0), reverse=True)
        if merged_paths:
            base["top_anchor_paths"] = merged_paths[:10]

        sources = unique_keep_order(
            values_as_list(a.get("candidate_sources"))
            + values_as_list(a.get("candidate_source"))
            + values_as_list(b.get("candidate_sources"))
            + values_as_list(b.get("candidate_source"))
        )
        if sources:
            base["candidate_sources"] = sources
            base["candidate_source"] = "+".join(str(x) for x in sources if x)

        base["path_count"] = max(
            int(a.get("path_count") or 0),
            int(b.get("path_count") or 0),
            len(merged_paths),
        )
        return base

    for c in candidates or []:
        drug = c.get("drug") or c.get("drug_name")
        key = normalize_name(drug)
        if not key:
            continue
        if key not in by_key:
            by_key[key] = dict(c)
        else:
            by_key[key] = merge_rows(by_key[key], c)

    out = list(by_key.values())
    out.sort(key=lambda x: (-safe_float(x.get("candidate_score"), 0.0), normalize_name(x.get("drug"))))
    return out


# ============================================================
# 6.5 Disease-agnostic general rerank
# ============================================================

def general_offline_rerank_score(
    c: Dict[str, Any],
    drug_sim_weight: float = OFFLINE_RERANK_DRUG_SIM_WEIGHT,
) -> float:
    """Integrate independent evidence channels exactly once.

    The graph score already contains relation, path, anchor-specificity and
    bounded evidence-diversity terms. This stage adds the semantic disease
    prior, drug-representation similarity, and disease-specific side-effect
    counter-evidence once. It deliberately does
    not rescore graph topology or translational properties.
    """
    cfg = GENERAL_RERANK_CONFIG or {}
    graph_score = safe_float(
        c.get("base_graph_score"),
        safe_float(c.get("candidate_score_original"), safe_float(c.get("candidate_score"))),
    )
    disease_prior = safe_float(c.get("embedding_prior_score"), 0.0)
    drug_similarity = safe_float(c.get("drug_embedding_similarity"), 0.0)
    side_effect_risk = safe_float(c.get("side_effect_risk_penalty"), 0.0)

    prior_weight = float(cfg.get("disease_prior_weight", 1.0))
    configured_drug_weight = float(cfg.get("drug_embedding_weight", drug_sim_weight))
    risk_weight = float(cfg.get("side_effect_risk_weight", 0.1))
    score = (
        graph_score
        + prior_weight * disease_prior
        + configured_drug_weight * drug_similarity
        - risk_weight * side_effect_risk
    )
    return round(score, 6)


def apply_general_offline_rerank(
    candidates: List[Dict[str, Any]],
    drug_sim_weight: float = OFFLINE_RERANK_DRUG_SIM_WEIGHT,
) -> List[Dict[str, Any]]:
    """Apply final disease-agnostic rerank to existing candidate list."""
    ranked = copy.deepcopy(candidates or [])

    for c in ranked:
        c["candidate_score_before_general_rerank"] = c.get("candidate_score")
        c["rank_before_general_rerank"] = c.get("rank")
        c["general_rerank_score"] = general_offline_rerank_score(
            c,
            drug_sim_weight=drug_sim_weight,
        )
        c["candidate_score"] = c["general_rerank_score"]

    ranked.sort(
        key=lambda x: (
            -safe_float(x.get("general_rerank_score")),
            -safe_float(x.get("candidate_score_before_general_rerank")),
            normalize_name(x.get("drug")),
        )
    )

    for i, c in enumerate(ranked, start=1):
        c["rank"] = i
        c["general_rerank_rank"] = i

    return ranked


# ============================================================
# 6.6 Global translational calibration
# ============================================================

def _candidate_lookup_value(c: Dict[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(c, dict) and c.get(key) is not None:
        return c.get(key)
    p = (c or {}).get("best_anchor_path") or (c or {}).get("best_path") or {}
    if isinstance(p, dict) and p.get(key) is not None:
        return p.get(key)
    return default


SPARSE_PATTERN_EVIDENCE_FAMILIES = {
    "direct_gene_action": "exact_disease_gene",
    "ppi_gene_action": "disease_gene_ppi",
    "sibling_gene_action": "disease_family_gene",
    "family_clinical_prior": "disease_family_clinical",
}

RANK_PARTITION_ORDER = {
    "therapeutic_priority_head": 0,
    "exploratory_tail": 1,
    "known_target_disease_contraindication": 2,
}


def inverse_degree_specificity(degree: Any) -> Optional[float]:
    """Continuous topology-only specificity; no fixed disease/gene blacklist."""
    if degree is None:
        return None
    value = max(0.0, safe_float(degree, 0.0))
    return round(1.0 / math.log2(2.0 + value), 6)


def candidate_direction_confidence(candidate: Dict[str, Any]) -> Tuple[str, str]:
    """Describe direction evidence without inventing disease-state polarity.

    `associated_with` is unsigned. A directional drug-gene edge therefore gives
    only partial confidence unless an explicit disease-state sign is present.
    """
    path = (candidate or {}).get("best_anchor_path") or (candidate or {}).get("best_path") or {}
    drug_rel = normalize_name(path.get("first_rel"))
    relset = normalize_relation_set(path.get("drug_gene_relset"))
    if drug_rel:
        relset.add(drug_rel)
    disease_sign = normalize_name(
        first_not_none(
            path.get("disease_state_sign"),
            path.get("disease_gene_direction"),
            candidate.get("disease_state_sign"),
        )
    )
    explicit_signs = {"up", "down", "upregulated", "downregulated", "gain_of_function", "loss_of_function"}
    directional = bool(relset & {"inhibition", "activation", "modulation"})
    if disease_sign in explicit_signs and directional:
        return "known", "explicit_disease_state_sign_and_directional_drug_action"
    if directional:
        return "partial", "directional_drug_action_but_unsigned_disease_association"
    return "unknown", "no_therapeutic_direction_inference_from_associated_with"


def _candidate_sparse_evidence_families(candidate: Dict[str, Any]) -> Tuple[List[str], bool]:
    patterns = unique_keep_order(
        list(candidate.get("supporting_patterns") or [])
        + [candidate.get("area_sparse_rescue_pattern")]
    )
    families = sorted({
        SPARSE_PATTERN_EVIDENCE_FAMILIES[p]
        for p in patterns
        if p in SPARSE_PATTERN_EVIDENCE_FAMILIES
    })
    path = candidate.get("area_sparse_rescue_evidence") or candidate.get("best_anchor_path") or candidate.get("best_path") or {}
    relset = normalize_relation_set(path.get("drug_gene_relset"))
    first_rel = normalize_name(path.get("first_rel"))
    if first_rel:
        relset.add(first_rel)
    exact_directional = (
        "direct_gene_action" in patterns
        and bool(relset & {"inhibition", "activation", "modulation"})
    )
    eligible = bool(exact_directional or len(families) >= 2)
    return families, eligible


def annotate_candidate_scientific_diagnostics(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add interpretable, disease-agnostic diagnostics without changing rank."""
    rows = copy.deepcopy(candidates or [])
    for candidate in rows:
        path = candidate.get("best_anchor_path") or candidate.get("best_path") or {}
        confidence, basis = candidate_direction_confidence(candidate)
        candidate["direction_confidence"] = confidence
        candidate["direction_confidence_basis"] = basis
        candidate["disease_state_direction_inferred"] = bool(confidence == "known")
        candidate["target_specificity"] = inverse_degree_specificity(path.get("target_gene_disease_degree"))
        bridge_degrees = [
            path.get("bridge_gene_degree"),
            path.get("phenotype_disease_degree"),
            path.get("family_child_degree"),
        ]
        bridge_specificities = [inverse_degree_specificity(x) for x in bridge_degrees if x is not None]
        candidate["anchor_specificity"] = min(bridge_specificities) if bridge_specificities else None
        candidate.setdefault("target_disease_contraindication", False)
        candidate.setdefault("target_disease_contraindication_provenance", [])
        candidate.setdefault("translational_risk_tier", "no_direct_target_disease_contraindication_found")
        candidate.setdefault("rank_partition", "therapeutic_priority_head")

        if candidate.get("source_sparse_rescue"):
            families, eligible = _candidate_sparse_evidence_families(candidate)
            candidate["independent_evidence_families"] = families
            candidate["sparse_priority_eligible"] = eligible
            candidate["rank_partition"] = (
                "therapeutic_priority_head" if eligible else "exploratory_tail"
            )
        else:
            candidate.setdefault("independent_evidence_families", [])
            candidate.setdefault("sparse_priority_eligible", None)
    return rows


def get_target_disease_contraindication_map(
    driver,
    disease: str,
    candidate_names: Iterable[str],
    disease_element_id: Optional[str] = None,
    disease_element_ids: Optional[List[str]] = None,
    timeout_sec: int = 30,
) -> Dict[str, List[Dict[str, Any]]]:
    """Retrieve direct target-disease contraindication edges for risk annotation."""
    names = unique_keep_order([normalize_name(x) for x in candidate_names if normalize_name(x)])
    if not names:
        return {}
    exact_ids = unique_keep_order(
        [x for x in (disease_element_ids or []) if x]
        + ([disease_element_id] if disease_element_id else [])
    )
    query = """
    MATCH (d:DISEASE)
    WHERE (size($disease_element_ids) > 0 AND elementId(d) IN $disease_element_ids)
       OR (size($disease_element_ids) = 0
           AND toLower(trim(d.name)) = toLower(trim($disease)))
    MATCH (d)-[r:contraindication]-(dr:DRUG)
    WHERE toLower(trim(dr.name)) IN $candidate_names
    RETURN toLower(trim(dr.name)) AS drug_key,
           dr.name AS drug,
           d.name AS disease,
           type(r) AS relation,
           r.Max_Phase AS max_phase
    ORDER BY drug_key, disease
    """
    rows = run_query_auto(
        driver,
        query,
        {
            "disease": disease,
            "disease_element_ids": exact_ids,
            "candidate_names": names,
        },
        timeout_sec=timeout_sec,
    )
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows or []:
        key = normalize_name(row.get("drug_key") or row.get("drug"))
        if not key:
            continue
        provenance = {
            "relation": row.get("relation") or "contraindication",
            "disease": row.get("disease") or disease,
            "drug": row.get("drug"),
            "max_phase": row.get("max_phase"),
        }
        if provenance not in out[key]:
            out[key].append(provenance)
    return dict(out)


def annotate_target_disease_contraindications(
    candidates: List[Dict[str, Any]],
    contraindication_map: Dict[str, List[Dict[str, Any]]],
    translational_partition: bool = False,
) -> List[Dict[str, Any]]:
    rows = copy.deepcopy(candidates or [])
    for candidate in rows:
        provenance = list((contraindication_map or {}).get(normalize_name(candidate.get("drug"))) or [])
        candidate["target_disease_contraindication"] = bool(provenance)
        candidate["target_disease_contraindication_provenance"] = provenance
        if provenance:
            candidate["translational_risk_tier"] = "known_target_disease_contraindication"
            if translational_partition:
                candidate["rank_partition"] = "known_target_disease_contraindication"
        else:
            candidate.setdefault("translational_risk_tier", "no_direct_target_disease_contraindication_found")
    return rows


def infer_abstention_reason(mechanism_obj: Dict[str, Any], final_candidates: List[Dict[str, Any]]) -> Optional[str]:
    """Return one deterministic reason code only when no candidate is retrieved."""
    if final_candidates:
        return None
    axes = mechanism_obj.get("mechanism_axes") or []
    high_axes = [a for a in axes if normalize_name(a.get("priority")) == "high"]
    resolved_high_axes = [a for a in high_axes if a.get("kg_anchor_resolution_available")]
    if resolved_high_axes and all(
        int(a.get("kg_resolved_raw_anchor_count") or a.get("kg_resolved_anchor_count") or 0) == 0
        for a in resolved_high_axes
    ):
        return "no_resolvable_high_priority_axis"

    stats = mechanism_obj.get("anchor_pattern_query_stats") or {}
    total_rows = sum(int((s or {}).get("rows") or 0) for s in stats.values())
    total_timeouts = sum(
        int((s or {}).get("timeouts") or 0) + int((s or {}).get("retry_timeouts") or 0)
        for s in stats.values()
    )
    if total_rows == 0 and total_timeouts > 0:
        return "query_timeout_exhausted"
    if total_rows == 0:
        return "all_disease_connected_patterns_zero"

    branches = mechanism_obj.get("retrieval_branches") or {}
    if branches.get("prior_drug_target_graph"):
        n_prior = sum(1 for c in mechanism_obj.get("candidate_drugs_before_drug_embedding", []) if c.get("source_prior_drug_target_graph"))
        if n_prior == 0:
            return "prior_verification_zero"
    sparse = mechanism_obj.get("sparse_area_rescue") or {}
    if sparse.get("enabled") and int(sparse.get("n_selected") or 0) == 0:
        return "sparse_rescue_zero"
    return "all_disease_connected_patterns_zero"


def global_translational_adjustment(c: Dict[str, Any]) -> float:
    """Disease- and area-independent feasibility adjustment."""
    cfg = TRANSLATIONAL_CALIBRATION_CONFIG or {}
    score = 0.0
    group = normalize_name(_candidate_lookup_value(c, "Group", ""))
    withdrawn = _candidate_lookup_value(c, "Withdrawn_Flag", _candidate_lookup_value(c, "Withdrawn", False))
    black_box = safe_float(_candidate_lookup_value(c, "Black_Box", _candidate_lookup_value(c, "BlackBox", 0.0)), 0.0)
    mw = safe_float(_candidate_lookup_value(c, "Molecular_Weight", _candidate_lookup_value(c, "MW", 0.0)), 0.0)
    tpsa = safe_float(_candidate_lookup_value(c, "Polar_Surface_Area", _candidate_lookup_value(c, "TPSA", 0.0)), 0.0)
    logp = safe_float(_candidate_lookup_value(c, "LogP", 0.0), 0.0)
    qed = safe_float(_candidate_lookup_value(c, "QED_Weighted", _candidate_lookup_value(c, "QED", 0.0)), 0.0)
    if group == "approved":
        score += float(cfg.get("approved_bonus", 0.08))
    if group == "withdrawn" or withdrawn is True or normalize_name(withdrawn) == "true":
        score -= float(cfg.get("withdrawn_penalty", 0.8))
    if black_box >= 1:
        score -= float(cfg.get("black_box_penalty", 0.15))
    if mw > float(cfg.get("mw_upper", 900.0)):
        score -= float(cfg.get("extreme_property_penalty", 0.12))
    if tpsa > float(cfg.get("tpsa_upper", 250.0)):
        score -= float(cfg.get("extreme_property_penalty", 0.12))
    if abs(logp) > float(cfg.get("abs_logp_upper", 8.0)):
        score -= float(cfg.get("extreme_property_penalty", 0.12))
    if qed > 0 and qed < float(cfg.get("qed_lower", 0.1)):
        score -= float(cfg.get("extreme_property_penalty", 0.12))
    if bool(c.get("target_disease_contraindication")):
        score -= float(cfg.get("target_disease_contraindication_penalty", 1.0))
    return round(score, 6)


def apply_area_property_postrank_calibration(candidates: List[Dict[str, Any]], area: str = "") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Compatibility name; applies one global calibration to every area."""
    ranked = copy.deepcopy(candidates or [])
    if not ENABLE_TRANSLATIONAL_CALIBRATION:
        for i, c in enumerate(ranked, 1):
            c["rank"] = i
        return ranked, {"enabled": False, "mode": "disabled", "area_used_for_scoring": False}
    for c in ranked:
        base = safe_float(c.get("candidate_score"), 0.0)
        adj = global_translational_adjustment(c)
        c["candidate_score_before_translational_calibration"] = base
        c["translational_adjustment"] = adj
        c["candidate_score"] = round(base + adj, 6)
    ranked.sort(
        key=lambda c: (
            int(RANK_PARTITION_ORDER.get(c.get("rank_partition"), 9)),
            -safe_float(c.get("candidate_score")),
            int(c.get("rank") or 999999),
            normalize_name(c.get("drug")),
        )
    )
    for i, c in enumerate(ranked, 1):
        c["rank_before_translational_calibration"] = c.get("rank")
        c["rank"] = i
    return ranked, {
        "enabled": True,
        "mode": "global_translational_calibration",
        "area": normalize_area_name(area),
        "area_used_for_scoring": False,
        "n_candidates": len(ranked),
        "top10": [c.get("drug") for c in ranked[:10]],
    }

# ============================================================
# 7. Evaluation# ============================================================
# 7. Evaluation
# ============================================================

def average_precision_at_k(gold_drugs: List[str], ranked_names: List[str], k: int = 100) -> float:
    """Truncated average precision (AUPRC-like ranking metric) at K.

    Positives not recovered within top-K contribute zero through the denominator.
    This is the rank-only equivalent of AUPRC/AP for a top-K candidate list and is
    directly comparable across GraphDRx and TxGNN after sorting each method by score and
    truncating to the same K.
    """
    gold_norms = {normalize_name(g) for g in (gold_drugs or []) if str(g or "").strip()}
    if not gold_norms or k <= 0:
        return 0.0
    seen = set()
    hits = 0
    precision_sum = 0.0
    for i, name in enumerate((ranked_names or [])[:k], start=1):
        nn = normalize_name(name)
        if nn in gold_norms and nn not in seen:
            hits += 1
            precision_sum += hits / i
            seen.add(nn)
    return precision_sum / len(gold_norms)


def evaluate_case(
    disease: str,
    gold_drugs: List[str],
    mechanism_obj: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate retrieval ranking only. Zero-gold cases remain reportable but non-evaluable."""
    ranked_rows = mechanism_obj.get("candidate_drugs_retrieval_final", mechanism_obj.get("candidate_drugs", []))
    ranked_names = [c["drug"] for c in ranked_rows]
    gold_ranks = {gold: rank_of_name(ranked_names, gold) for gold in gold_drugs}
    ranks = list(gold_ranks.values())

    def matched_golds_at_k(k: int) -> List[str]:
        return [gold for gold, rank in gold_ranks.items() if rank is not None and rank <= k]

    metrics = {
        "disease": disease,
        "n_gold": len(gold_drugs),
        "evaluable": bool(gold_drugs),
        "gold_drugs": gold_drugs,
        "retrieval_gold_ranks": gold_ranks,
        "retrieval_mrr": reciprocal_rank(ranks) if gold_drugs else None,
    }

    for k in HIT_KS:
        matched = matched_golds_at_k(k)
        metrics[f"retrieval_hit_at_{k}"] = (len(matched) > 0) if gold_drugs else None
        metrics[f"retrieval_gold_hit_count_at_{k}"] = len(matched)
        metrics[f"retrieval_precision_at_{k}"] = (len(matched) / k) if gold_drugs and k else None
        metrics[f"retrieval_average_precision_at_{k}"] = (
            average_precision_at_k(gold_drugs, ranked_names, k) if gold_drugs else None
        )
        metrics[f"retrieval_gold_recall_at_{k}"] = (
            len(matched) / len(gold_drugs) if gold_drugs else None
        )
        metrics[f"retrieval_matched_gold_at_{k}"] = matched

    return metrics


def _json_for_csv(v):
    if isinstance(v, set):
        v = sorted(v)
    if isinstance(v, (list, dict, tuple, set)):
        return json.dumps(v, ensure_ascii=False)
    return v


def _as_retrieval_summary_row(r: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        "case_id": r.get("case_id"),
        "area": r.get("area"),
        "disease": r.get("disease"),
        "n_gold": r.get("n_gold"),
        "evaluable": r.get("evaluable"),
        "gold_drugs": r.get("gold_drugs", []),
    }
    for k in HIT_KS:
        for metric in (
            "hit", "gold_hit_count", "precision",
            "average_precision", "gold_recall", "matched_gold",
        ):
            key = f"retrieval_{metric}_at_{k}"
            row[key] = r.get(key)
    row["retrieval_mrr"] = r.get("retrieval_mrr")
    row["retrieval_gold_ranks"] = r.get("retrieval_gold_ranks", {})
    return row


def write_summary_csv(case_results: List[Dict[str, Any]], out_path: Path):
    rows = [_as_retrieval_summary_row(r) for r in (case_results or [])]
    fieldnames = ["case_id", "area", "disease", "n_gold", "evaluable", "gold_drugs"]
    for k in HIT_KS:
        fieldnames.extend([
            f"retrieval_hit_at_{k}",
            f"retrieval_gold_hit_count_at_{k}",
            f"retrieval_precision_at_{k}",
            f"retrieval_average_precision_at_{k}",
            f"retrieval_gold_recall_at_{k}",
            f"retrieval_matched_gold_at_{k}",
        ])
    fieldnames.extend(["retrieval_mrr", "retrieval_gold_ranks"])
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _json_for_csv(row.get(k)) for k in fieldnames})


def extract_prefixed_metric_rows(case_metrics: List[Dict[str, Any]], prefix: str) -> List[Dict[str, Any]]:
    out = []
    prefix_str = f"{prefix}_"
    for r in case_metrics or []:
        row = {key.replace(prefix_str, "", 1): value for key, value in r.items() if key.startswith(prefix_str)}
        row["case_id"] = r.get("case_id")
        row["area"] = r.get("area")
        if row:
            out.append(row)
    return out


def aggregate_metrics(case_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_cases = len(case_results)
    if total_cases == 0:
        return {}

    evaluable = [r for r in case_results if int(r.get("n_gold") or 0) > 0]
    n = len(evaluable)
    zero_gold = [str(r.get("disease") or "") for r in case_results if int(r.get("n_gold") or 0) == 0]
    total_gold_labels = sum(int(r.get("n_gold") or 0) for r in evaluable)

    out = {
        "n_cases": total_cases,
        "n_evaluable_cases": n,
        "n_zero_gold_cases": total_cases - n,
        "zero_gold_diseases": zero_gold,
        "total_gold_labels": total_gold_labels,
        "metric_denominator_note": "Hit, macro recall, MAP, precision, and MRR use only cases with n_gold > 0.",
    }

    for k in HIT_KS:
        total_hits = sum(int(r.get(f"retrieval_gold_hit_count_at_{k}") or 0) for r in evaluable)
        hit_cases = sum(bool(r.get(f"retrieval_hit_at_{k}")) for r in evaluable)
        out[f"retrieval_hits_at_{k}"] = hit_cases
        out[f"retrieval_hits_at_{k}_rate"] = (hit_cases / n) if n else None
        out[f"retrieval_mean_gold_recall_at_{k}"] = (
            sum(float(r.get(f"retrieval_gold_recall_at_{k}") or 0.0) for r in evaluable) / n
            if n else None
        )
        out[f"retrieval_total_gold_hits_at_{k}"] = total_hits
        out[f"retrieval_micro_gold_recall_at_{k}"] = (
            total_hits / total_gold_labels if total_gold_labels else None
        )
        out[f"retrieval_mean_precision_at_{k}"] = (
            sum(float(r.get(f"retrieval_precision_at_{k}") or 0.0) for r in evaluable) / n
            if n else None
        )
        out[f"retrieval_micro_precision_at_{k}"] = (total_hits / (n * k)) if n and k else None
        out[f"retrieval_mean_average_precision_at_{k}"] = (
            sum(float(r.get(f"retrieval_average_precision_at_{k}") or 0.0) for r in evaluable) / n
            if n else None
        )

    out["retrieval_mean_mrr"] = (
        sum(float(r.get("retrieval_mrr") or 0.0) for r in evaluable) / n if n else None
    )
    return out


def _clean_aggregate_reporting_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow copy for stable CSV reporting."""
    return dict(row or {})


def write_rows_csv(rows: List[Dict[str, Any]], out_path: Path, preferred: Optional[List[str]] = None):
    """Write arbitrary dict rows with stable reproducible column order."""
    rows = [_clean_aggregate_reporting_row(r) for r in (rows or [])]
    preferred = preferred or []

    keys = []
    for k in preferred:
        if k not in keys:
            keys.append(k)
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)

    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            row = {}
            for k in keys:
                row[k] = _json_for_csv(r.get(k))
            writer.writerow(row)



def atomic_write_json(path: Path, payload: Any, *, indent: Optional[int] = None) -> None:
    """Durably replace a JSON file only after a complete temporary write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(
                payload,
                f,
                ensure_ascii=False,
                indent=indent,
                separators=None if indent is not None else (",", ":"),
            )
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def append_jsonl_record(path: Path, payload: Dict[str, Any]) -> None:
    """Append one complete disease result; previous lines survive an interrupted write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def load_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    """Load complete JSONL records and truncate only a malformed trailing record."""
    path = Path(path)
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("rb+") as f:
        while True:
            record_start = f.tell()
            raw = f.readline()
            if not raw:
                break
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                remainder = f.read()
                if remainder.strip():
                    raise
                print(f"[RESUME WARNING] truncating incomplete trailing JSONL record: {path}", flush=True)
                f.seek(record_start)
                f.truncate()
                break
            if isinstance(obj, dict):
                records.append(obj)
    return records


def _compact_path_for_csv(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return (
        candidate.get("best_anchor_path")
        or candidate.get("best_path")
        or candidate.get("area_sparse_rescue_evidence")
        or {}
    )


def compact_top_candidate_rows(all_results: List[Dict[str, Any]], top_k: int = 200) -> List[Dict[str, Any]]:
    """One compact row per disease-drug candidate, including its best KG path."""
    rows: List[Dict[str, Any]] = []
    limit = max(int(top_k or 0), 0)
    for result in all_results or []:
        mechanism = result.get("retrieval") or {}
        retrieval = (
            mechanism.get("candidate_drugs_retrieval_final")
            or mechanism.get("candidate_drugs_final")
            or mechanism.get("candidate_drugs")
            or []
        )
        translational = mechanism.get("candidate_drugs_translational_priority") or []
        translational_by_drug = {
            normalize_name(c.get("drug")): c for c in translational if c.get("drug")
        }
        for fallback_rank, candidate in enumerate(retrieval[:limit], start=1):
            drug = candidate.get("drug")
            if not drug:
                continue
            path = _compact_path_for_csv(candidate)
            downstream = translational_by_drug.get(normalize_name(drug), {})
            rows.append({
                "case_id": (result.get("metrics") or {}).get("case_id"),
                "area": result.get("area"),
                "disease": result.get("disease"),
                "rank": int(candidate.get("rank") or fallback_rank),
                "drug": drug,
                "retrieval_score": first_not_none(
                    candidate.get("retrieval_score"),
                    candidate.get("general_rerank_score"),
                    candidate.get("candidate_score"),
                ),
                "translational_rank": downstream.get("rank"),
                "translational_score": first_not_none(
                    downstream.get("translational_priority_score"),
                    downstream.get("candidate_score"),
                ),
                "base_graph_score": first_not_none(
                    candidate.get("base_graph_score"),
                    candidate.get("candidate_score_original"),
                    candidate.get("best_path_score"),
                ),
                "embedding_prior_score": candidate.get("embedding_prior_score"),
                "drug_embedding_similarity": candidate.get("drug_embedding_similarity"),
                "source_anchor_branch": candidate.get("source_anchor_branch"),
                "source_disease_context_fallback_graph": candidate.get("source_disease_context_fallback_graph"),
                "source_vector_neighbor_anchor_graph": candidate.get("source_vector_neighbor_anchor_graph"),
                "source_semantic_prior_support": candidate.get("source_semantic_prior_support"),
                "source_prior_graph_verified": candidate.get("source_prior_graph_verified"),
                "source_common_direct_graph": candidate.get("source_common_direct_graph"),
                "prior_verification_status": candidate.get("prior_verification_status"),
                "prior_diseases": candidate.get("prior_diseases") or [],
                "prior_primary_support_diseases": candidate.get("prior_primary_support_diseases") or [],
                "prior_rels": candidate.get("prior_rels") or [],
                "structural_evidence_signature": candidate.get("structural_evidence_signature"),
                "structural_evidence_group": candidate.get("structural_evidence_group"),
                "structural_evidence_reason": candidate.get("structural_evidence_reason"),
                "structural_evidence_tier": candidate.get("structural_evidence_tier"),
                "rank_before_structural_evidence_ranking": candidate.get("rank_before_structural_evidence_ranking"),
                "rank_shift_structural_evidence_ranking": candidate.get("rank_shift_structural_evidence_ranking"),
                "candidate_sources": candidate.get("candidate_sources") or [candidate.get("candidate_source")],
                "supporting_patterns": candidate.get("supporting_patterns") or [path.get("pattern")],
                "supporting_axes": candidate.get("supporting_axes") or [candidate.get("mechanism_axis")],
                "direction_confidence": candidate.get("direction_confidence"),
                "direct_contraindication": candidate.get("direct_target_disease_contraindication"),
                "best_path_pattern": path.get("pattern"),
                "best_path_first_rel": path.get("first_rel"),
                "best_path_target_gene": path.get("target_gene"),
                "best_path_disease_gene": path.get("disease_gene"),
                "best_path_anchor": first_not_none(path.get("anchor_name"), path.get("bridge_node_name")),
                "best_path_anchor_type": first_not_none(path.get("anchor_type"), path.get("bridge_node_type")),
                "best_path_hop_len": path.get("hop_len"),
                "best_path_score": first_not_none(path.get("path_score"), candidate.get("best_path_score")),
                "drug_gene_degree": path.get("drug_gene_degree"),
                "target_gene_disease_degree": path.get("target_gene_disease_degree"),
                "target_gene_ppi_degree": path.get("target_gene_ppi_degree"),
                "best_path_nodes": path.get("path_nodes") or [],
                "best_path_rels": path.get("path_rels") or [],
            })
    return rows


def write_compact_top_candidates_csv(all_results: List[Dict[str, Any]], out_path: Path, top_k: int = 200) -> None:
    rows = compact_top_candidate_rows(all_results, top_k=top_k)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    write_rows_csv(
        rows,
        tmp,
        preferred=[
            "case_id", "area", "disease", "rank", "drug", "retrieval_score",
            "translational_rank", "translational_score", "base_graph_score",
            "embedding_prior_score", "drug_embedding_similarity",
            "source_anchor_branch", "source_disease_context_fallback_graph", "source_vector_neighbor_anchor_graph",
            "source_semantic_prior_support", "source_prior_graph_verified",
            "source_common_direct_graph", "prior_verification_status",
            "prior_diseases", "prior_primary_support_diseases", "prior_rels",
            "structural_evidence_signature", "structural_evidence_group",
            "structural_evidence_reason", "structural_evidence_tier",
            "rank_before_structural_evidence_ranking", "rank_shift_structural_evidence_ranking",
            "candidate_sources", "supporting_patterns", "supporting_axes", "direction_confidence",
            "direct_contraindication", "best_path_pattern", "best_path_first_rel",
            "best_path_target_gene", "best_path_disease_gene", "best_path_anchor",
            "best_path_anchor_type", "best_path_hop_len", "best_path_score",
            "drug_gene_degree", "target_gene_disease_degree", "target_gene_ppi_degree",
            "best_path_nodes", "best_path_rels",
        ],
    )
    os.replace(tmp, out_path)


def aggregate_stage_metrics(all_results: List[Dict[str, Any]], key: str = "stage_metrics") -> Dict[str, Dict[str, Any]]:
    """Aggregate GraphRAG-only / disease-prior / drug-embedding / final snapshots."""
    by_stage: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for res in all_results or []:
        for stage_name, row in (res.get(key) or {}).items():
            if isinstance(row, dict):
                by_stage[stage_name].append(row)
    return {stage: aggregate_metrics(rows) for stage, rows in by_stage.items()}


def flatten_stage_metric_rows(all_results: List[Dict[str, Any]], key: str = "stage_metrics") -> List[Dict[str, Any]]:
    """Return one CSV-ready row per disease × stage."""
    rows: List[Dict[str, Any]] = []
    for res in all_results or []:
        disease = res.get("disease")
        runtime_sec = res.get("runtime_sec")
        metrics = res.get("metrics") or {}
        for stage_name, m in (res.get(key) or {}).items():
            if not isinstance(m, dict):
                continue
            row = dict(m)
            row["stage"] = stage_name
            row["disease"] = row.get("disease") or disease
            row["case_id"] = row.get("case_id") or metrics.get("case_id")
            row["area"] = row.get("area") or metrics.get("area")
            row["runtime_sec"] = runtime_sec
            rows.append(row)
    preferred = ["stage", "case_id", "area", "disease", "runtime_sec"]
    out = []
    for row in rows:
        ordered = {k: row.get(k) for k in preferred if k in row}
        for k, v in row.items():
            if k not in ordered:
                ordered[k] = v
        out.append(ordered)
    return out


def flatten_stage_aggregate_rows(stage_aggregate: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for stage_name, m in (stage_aggregate or {}).items():
        row = {"stage": stage_name}
        if isinstance(m, dict):
            row.update(m)
        rows.append(row)
    return rows


def aggregate_stage_metrics_by_area(all_results: List[Dict[str, Any]], key: str = "stage_metrics") -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in flatten_stage_metric_rows(all_results, key=key):
        grouped[(str(row.get("stage") or "unknown"), normalize_area_name(row.get("area")))] .append(row)
    rows = []
    for (stage, area), values in sorted(grouped.items()):
        out = {"stage": stage, "area": area}
        out.update(aggregate_metrics(values))
        rows.append(out)
    return rows


def _candidate_patterns(c: Dict[str, Any]) -> List[str]:
    patterns = c.get("supporting_patterns") or []
    if isinstance(patterns, str):
        patterns = [patterns]
    if not patterns:
        bp = c.get("best_path") or c.get("best_anchor_path") or {}
        p = bp.get("pattern")
        if p:
            patterns = [p]
    return [str(p) for p in patterns if str(p or "").strip()]


def pattern_metric_rows_for_result(res: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compute disease-level strict-gold metrics for each supporting GraphRAG pattern.

    This is post-hoc and cheap: it reuses raw GraphRAG candidates from the same run.
    A candidate counts for a pattern if that pattern is in supporting_patterns. Ranking
    within each pattern follows the already-computed GraphRAG score/rank.
    """
    retrieval_obj = res.get("retrieval") or {}
    raw_candidates = retrieval_obj.get("raw_candidates") or retrieval_obj.get("candidate_drugs_graphrag_only") or []
    metrics = res.get("metrics") or {}
    gold = metrics.get("strict_gold_drugs") or metrics.get("gold_drugs") or []
    disease = res.get("disease") or metrics.get("disease")
    area = metrics.get("area")
    case_id = metrics.get("case_id")
    runtime_sec = res.get("runtime_sec")

    by_pattern: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    best_path_counts: Dict[str, int] = defaultdict(int)
    query_stats = retrieval_obj.get("anchor_pattern_query_stats") or {}

    for c in raw_candidates or []:
        if not isinstance(c, dict):
            continue
        bp = c.get("best_path") or c.get("best_anchor_path") or {}
        best_pattern = str(bp.get("pattern") or "")
        if best_pattern:
            best_path_counts[best_pattern] += 1
        for p in _candidate_patterns(c):
            by_pattern[p].append(dict(c))

    # Preserve explicitly executed zero-yield patterns. These rows are required
    # for post-hoc pattern-by-area efficiency pruning after the full benchmark.
    # Disabled or structurally skipped patterns are not represented as executed.
    for pattern, stat in query_stats.items():
        if int((stat or {}).get("calls") or 0) > 0:
            by_pattern.setdefault(str(pattern), [])

    rows: List[Dict[str, Any]] = []
    for pattern, candidates in sorted(by_pattern.items()):
        candidates = sorted(
            candidates,
            key=lambda x: (
                int(x.get("rank") or 999999),
                -safe_float(x.get("candidate_score")),
                normalize_name(x.get("drug")),
            ),
        )
        for i, c in enumerate(candidates, start=1):
            c["rank"] = i

        row = evaluate_case(
            disease=str(disease or ""),
            gold_drugs=gold,
            mechanism_obj={"candidate_drugs": candidates},
        )
        stat = query_stats.get(pattern) or {}
        row.update({
            "pattern": pattern,
            "case_id": case_id,
            "area": area,
            "disease": disease,
            "runtime_sec": runtime_sec,
            "n_pattern_candidates": len(candidates),
            "n_best_path_candidates": best_path_counts.get(pattern, 0),
            "query_calls": int(stat.get("calls") or 0),
            "query_rows": int(stat.get("rows") or 0),
            "query_timeouts": int(stat.get("timeouts") or 0),
            "query_errors": int(stat.get("errors") or 0),
            "query_retry_attempts": int(stat.get("retry_attempts") or 0),
            "query_retry_successes": int(stat.get("retry_successes") or 0),
            "query_retry_timeouts": int(stat.get("retry_timeouts") or 0),
            "query_retry_errors": int(stat.get("retry_errors") or 0),
            "query_retry_rows": int(stat.get("retry_rows") or 0),
            "query_branches": ";".join(stat.get("branches") or []),
            "zero_yield_executed": bool(int(stat.get("calls") or 0) > 0 and len(candidates) == 0),
            "pattern_metric_note": "posthoc candidate metrics plus persisted query telemetry; executed zero-yield patterns are retained",
        })
        rows.append(row)
    return rows


def flatten_pattern_metric_rows(all_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for res in all_results or []:
        rows.extend(pattern_metric_rows_for_result(res))
    preferred = [
        "pattern", "case_id", "area", "disease", "runtime_sec",
        "n_pattern_candidates", "n_best_path_candidates", "query_calls",
        "query_rows", "query_timeouts", "query_retry_attempts",
        "query_retry_successes", "query_retry_timeouts", "query_errors",
        "zero_yield_executed",
    ]
    out = []
    for row in rows:
        ordered = {k: row.get(k) for k in preferred if k in row}
        for k, v in row.items():
            if k not in ordered:
                ordered[k] = v
        out.append(ordered)
    return out


def aggregate_pattern_metric_rows(pattern_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_pattern: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in pattern_rows or []:
        p = str(row.get("pattern") or "unknown")
        by_pattern[p].append(row)
    out = {}
    for p, rows in by_pattern.items():
        agg = aggregate_metrics(rows)
        agg["total_pattern_candidates"] = sum(int(r.get("n_pattern_candidates") or 0) for r in rows)
        agg["total_best_path_candidates"] = sum(int(r.get("n_best_path_candidates") or 0) for r in rows)
        agg["total_query_calls"] = sum(int(r.get("query_calls") or 0) for r in rows)
        agg["total_query_rows"] = sum(int(r.get("query_rows") or 0) for r in rows)
        agg["total_query_timeouts"] = sum(int(r.get("query_timeouts") or 0) for r in rows)
        agg["total_query_retry_attempts"] = sum(int(r.get("query_retry_attempts") or 0) for r in rows)
        agg["total_query_retry_successes"] = sum(int(r.get("query_retry_successes") or 0) for r in rows)
        agg["total_query_retry_timeouts"] = sum(int(r.get("query_retry_timeouts") or 0) for r in rows)
        agg["zero_yield_executions"] = sum(1 for r in rows if bool(r.get("zero_yield_executed")))
        out[p] = agg
    return out


def flatten_pattern_aggregate_rows(pattern_aggregate: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for pattern, m in (pattern_aggregate or {}).items():
        row = {"pattern": pattern}
        if isinstance(m, dict):
            row.update(m)
        rows.append(row)
    return rows


def aggregate_pattern_metrics_by_area(pattern_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in pattern_rows or []:
        grouped[(str(row.get("pattern") or "unknown"), normalize_area_name(row.get("area")))] .append(row)
    rows = []
    for (pattern, area), values in sorted(grouped.items()):
        out = {"pattern": pattern, "area": area}
        out.update(aggregate_metrics(values))
        out["total_pattern_candidates"] = sum(int(r.get("n_pattern_candidates") or 0) for r in values)
        out["total_best_path_candidates"] = sum(int(r.get("n_best_path_candidates") or 0) for r in values)
        out["total_query_calls"] = sum(int(r.get("query_calls") or 0) for r in values)
        out["total_query_rows"] = sum(int(r.get("query_rows") or 0) for r in values)
        out["total_query_timeouts"] = sum(int(r.get("query_timeouts") or 0) for r in values)
        out["total_query_errors"] = sum(int(r.get("query_errors") or 0) for r in values)
        out["total_query_retry_attempts"] = sum(int(r.get("query_retry_attempts") or 0) for r in values)
        out["total_query_retry_successes"] = sum(int(r.get("query_retry_successes") or 0) for r in values)
        out["total_query_retry_timeouts"] = sum(int(r.get("query_retry_timeouts") or 0) for r in values)
        out["zero_yield_executions"] = sum(1 for r in values if bool(r.get("zero_yield_executed")))
        rows.append(out)
    return rows



# ============================================================
# 8. Main runner
# ============================================================


# ============================================================================
# Topology-only sparse rescue
# ============================================================================

SPARSE_RESCUE_DEFAULT_AREAS: set = set()
SPARSE_RESCUE_PROTECT_TOP_N_DEFAULT = 90
SPARSE_RESCUE_TOP_K_DEFAULT = 100
SPARSE_RESCUE_TIER_INSERT_AFTER_DEFAULT = {"strong": 10, "medium": 60, "weak": 90}
SPARSE_RESCUE_TIER_MAX_DEFAULT = {"strong": 3, "medium": 5, "weak": 10}
SPARSE_RESCUE_TIER_INSERT_AFTER_BY_AREA: Dict[str, Dict[str, int]] = {}
SPARSE_RESCUE_TIER_MAX_BY_AREA: Dict[str, Dict[str, int]] = {}

SPARSE_DISEASE_DRUG_PRIOR_REL_WEIGHTS = {
    "indication": 0.30,
    "off_label_use": 0.18,
    "tested_indication": 0.12,
    "studied_for_treatment_of": 0.08,
    "contraindication": -0.8,
}
SPARSE_DRUG_TARGET_ACTION_REL_WEIGHTS = {
    # Sparse rescue is a bounded recall mechanism. These relation-only
    # fallbacks intentionally remain on a smaller scale than main path scores.
    "inhibition": 0.42,
    "activation": 0.42,
    "modulation": 0.32,
    "binding": 0.18,
    "target": 0.15,
    "enzyme": 0.12,
    "transporter": 0.10,
    "carrier": 0.05,
}

SPARSE_DRUG_ACTION_TYPE_WEIGHTS = {
    "inhibitor": 0.47,
    "blocker": 0.47,
    "antagonist": 0.45,
    "inverse agonist": 0.44,
    "degrader": 0.50,
    "negative allosteric modulator": 0.43,
    "agonist": 0.47,
    "activator": 0.47,
    "opener": 0.44,
    "partial agonist": 0.42,
    "releasing agent": 0.40,
    "positive allosteric modulator": 0.44,
    "positive modulator": 0.40,
    "modulator": 0.36,
    "stabiliser": 0.37,
    "stabilizer": 0.37,
    "binding agent": 0.22,
    "substrate": 0.19,
}

SPARSE_TESTED_INDICATION_PHASE_WEIGHTS = {
    4.0: 0.24,
    3.0: 0.20,
    2.0: 0.16,
    1.0: 0.10,
    0.5: 0.07,
    0.0: 0.04,
}
SPARSE_RELATION_SEMANTIC_LABELS = {
    "indication": "clinical_indication", "off_label_use": "off_label_context",
    "tested_indication": "clinical_development_context",
    "studied_for_treatment_of": "literature_treatment_context",
    "studied_for_marker_mechanism_of": "context_only_not_treatment_evidence",
    "contraindication": "risk_counter_evidence",
    "inhibition": "directional_pharmacology", "activation": "directional_pharmacology",
    "modulation": "directional_pharmacology", "target": "target_annotation",
    "binding": "binding_evidence", "enzyme": "adme_relation",
    "transporter": "adme_relation", "carrier": "adme_relation",
}


def _sparse_extract_anchor_genes_from_retrieval(mechanism_obj: Dict[str, Any]) -> List[str]:
    out = []
    for axis in (mechanism_obj or {}).get("mechanism_axes", []) or []:
        out.extend(extract_gene_mentions(axis.get("anchor_genes") or []))
    ga = (mechanism_obj or {}).get("global_anchors") or {}
    out.extend(extract_gene_mentions(ga.get("target_genes") or []))
    return unique_keep_order([normalize_gene(x) for x in out if normalize_gene(x)])


def _sparse_area_policy(area: Any) -> Dict[str, Any]:
    return dict((STRUCTURAL_AREA_POLICY.get(normalize_area_name(area), {}) or {}).get("sparse_rescue") or {})


def _merge_sparse_policy(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key].update(value)
        elif key in {"enabled_patterns", "dense_allowed_patterns"}:
            out[key] = unique_keep_order(list(out.get(key) or []) + list(value or []))
        else:
            out[key] = copy.deepcopy(value)
    return out


def _effective_sparse_policy(area: Any, force_candidate_floor: bool = False) -> Dict[str, Any]:
    area_policy = _sparse_area_policy(area)
    if not force_candidate_floor:
        return area_policy
    # Start with the global safety net and retain any area-level budgets, but an
    # area-level enabled=false setting must not disable the global floor trigger.
    merged = _merge_sparse_policy(GLOBAL_CANDIDATE_FLOOR_POLICY, area_policy)
    merged["enabled"] = True
    return merged


def _sparse_direct_gene_count(driver, disease_name: str, timeout_sec: int = 20) -> int:
    q = """
    MATCH (d:DISEASE) WHERE toLower(d.name)=toLower($disease)
    OPTIONAL MATCH (d)-[:associated_with]-(g:GENE)
    RETURN count(DISTINCT g) AS n
    """
    rows = run_query_auto(driver, q, {"disease": disease_name}, timeout_sec=timeout_sec)
    return int((rows[0] if rows else {}).get("n") or 0)


def _sparse_queries() -> Dict[str, str]:
    rels = "|".join(DRUG_GENE_RELS)
    common_drug = """
      dr.Group AS `Group`, dr.Max_Phase AS MaxPhase, dr.Black_Box AS BlackBox,
      dr.Withdrawn_Flag AS Withdrawn, dr.Inorganic_Flag AS Inorganic,
      dr.Molecular_Weight AS MW, dr.Polar_Surface_Area AS TPSA,
      coalesce(dr.CX_LogP,dr.XLogP,dr.CLogP,dr.AlogP) AS LogP,
      dr.QED_Weighted AS QED, dr.selected_CID AS PubChemCID,
      dr.selected_ChID AS ChEMBLID, dr.InChIKey AS InChIKey,
      dr.Drug_Type AS DrugType
    """
    return {
        "direct_gene_action": f"""
        MATCH (d:DISEASE) WHERE toLower(d.name)=toLower($disease)
        MATCH (d)-[:associated_with]-(dg:GENE)
        MATCH (dr:DRUG)-[r:{rels}]-(dg)
        RETURN dr.name AS drug, type(r) AS first_rel, r.Action_Type AS action_type,
          [(dr)-[rr]-(dg) WHERE type(rr) IN $drug_gene_rels | type(rr)] AS drug_gene_relset,
          [x IN [(dr)-[rr]-(dg) WHERE type(rr) IN $drug_gene_rels | rr.Action_Type] WHERE x IS NOT NULL] AS drug_gene_action_types,
          dg.name AS target_gene, dg.name AS disease_gene, null AS bridge_node_name,
          'DIRECT_DISEASE_GENE' AS bridge_node_type, 2 AS hop_len,
          [dr.name,dg.name,d.name] AS path_nodes, [type(r),'associated_with'] AS path_rels,
          size([(dg)-[:associated_with]-(x:DISEASE) | x]) AS target_gene_disease_degree,
          size([(dg)-[:ppi]-(x:GENE) | x]) AS target_gene_ppi_degree,
          {common_drug}
        LIMIT $limit
        """,
        "ppi_gene_action": f"""
        MATCH (d:DISEASE) WHERE toLower(d.name)=toLower($disease)
        MATCH (d)-[:associated_with]-(dg:GENE)-[:ppi]-(tg:GENE)
        MATCH (dr:DRUG)-[r:{rels}]-(tg)
        RETURN dr.name AS drug, type(r) AS first_rel, r.Action_Type AS action_type,
          [(dr)-[rr]-(tg) WHERE type(rr) IN $drug_gene_rels | type(rr)] AS drug_gene_relset,
          [x IN [(dr)-[rr]-(tg) WHERE type(rr) IN $drug_gene_rels | rr.Action_Type] WHERE x IS NOT NULL] AS drug_gene_action_types,
          tg.name AS target_gene, dg.name AS disease_gene, dg.name AS bridge_node_name,
          'PPI_DISEASE_GENE' AS bridge_node_type, 3 AS hop_len,
          [dr.name,tg.name,dg.name,d.name] AS path_nodes, [type(r),'ppi','associated_with'] AS path_rels,
          size([(tg)-[:associated_with]-(x:DISEASE) | x]) AS target_gene_disease_degree,
          size([(tg)-[:ppi]-(x:GENE) | x]) AS target_gene_ppi_degree,
          {common_drug}
        LIMIT $limit
        """,
        "sibling_gene_action": f"""
        MATCH (d:DISEASE) WHERE toLower(d.name)=toLower($disease)
        MATCH (d)-[:parent_child]->(parent:DISEASE)<-[:parent_child]-(sib:DISEASE)
        WHERE sib <> d
          AND toLower(trim(sib.name)) <> toLower(trim(d.name))
          AND toLower(trim(parent.name)) <> toLower(trim(d.name))
        MATCH (sib)-[:associated_with]-(dg:GENE)
        MATCH (dr:DRUG)-[r:{rels}]-(dg)
        RETURN dr.name AS drug, type(r) AS first_rel, r.Action_Type AS action_type,
          [(dr)-[rr]-(dg) WHERE type(rr) IN $drug_gene_rels | type(rr)] AS drug_gene_relset,
          [x IN [(dr)-[rr]-(dg) WHERE type(rr) IN $drug_gene_rels | rr.Action_Type] WHERE x IS NOT NULL] AS drug_gene_action_types,
          dg.name AS target_gene, dg.name AS disease_gene, sib.name AS bridge_node_name,
          parent.name AS family_parent_name,
          'SIBLING_DISEASE_GENE' AS bridge_node_type, 4 AS hop_len,
          [dr.name,dg.name,sib.name,parent.name,d.name] AS path_nodes,
          [type(r),'associated_with','parent_child','parent_child'] AS path_rels,
          size([(dg)-[:associated_with]-(x:DISEASE) | x]) AS target_gene_disease_degree,
          size([(dg)-[:ppi]-(x:GENE) | x]) AS target_gene_ppi_degree,
          size([(parent)<-[:parent_child]-(x:DISEASE) | x]) AS family_child_degree,
          {common_drug}
        LIMIT $limit
        """,
        "family_clinical_prior": """
        MATCH (d:DISEASE) WHERE toLower(d.name)=toLower($disease)
        MATCH (d)-[:parent_child]->(parent:DISEASE)<-[:parent_child]-(sib:DISEASE)
        WHERE sib <> d
          AND toLower(trim(sib.name)) <> toLower(trim(d.name))
          AND toLower(trim(parent.name)) <> toLower(trim(d.name))
        MATCH (sib)-[r:indication|off_label_use|tested_indication|studied_for_treatment_of]-(dr:DRUG)
        RETURN dr.name AS drug, type(r) AS first_rel, r.Max_Phase AS relation_max_phase,
          [(sib)-[rr]-(dr) WHERE type(rr) IN ['indication','off_label_use','tested_indication','studied_for_treatment_of','contraindication'] | type(rr)] AS disease_drug_relset,
          [x IN [(sib)-[rr:tested_indication]-(dr) | rr.Max_Phase] WHERE x IS NOT NULL] AS tested_indication_phases,
          null AS target_gene, null AS disease_gene, sib.name AS bridge_node_name,
          parent.name AS family_parent_name,
          'SIBLING_CLINICAL_PRIOR' AS bridge_node_type, 3 AS hop_len,
          [dr.name,sib.name,parent.name,d.name] AS path_nodes,
          [type(r),'parent_child','parent_child'] AS path_rels,
          size([(parent)<-[:parent_child]-(x:DISEASE) | x]) AS family_child_degree,
          dr.Group AS `Group`, dr.Max_Phase AS MaxPhase, dr.Black_Box AS BlackBox,
          dr.Withdrawn_Flag AS Withdrawn, dr.Inorganic_Flag AS Inorganic,
          dr.Molecular_Weight AS MW, dr.Polar_Surface_Area AS TPSA,
          coalesce(dr.CX_LogP,dr.XLogP,dr.CLogP,dr.AlogP) AS LogP,
          dr.QED_Weighted AS QED, dr.selected_CID AS PubChemCID,
          dr.selected_ChID AS ChEMBLID, dr.InChIKey AS InChIKey,
          dr.Drug_Type AS DrugType
        LIMIT $limit
        """,
    }


def _nonempty_identity_value(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(text) and normalize_name(text) not in {"nan", "none", "null", "unknown", "not available"}


def pharmacological_identity_evidence(row: Dict[str, Any]) -> List[str]:
    """Return non-name evidence that a DRUG-labeled node is pharmacological.

    The gate uses stable chemical/biologic identifiers or structured molecular
    annotations. Group/approval status alone is deliberately insufficient because
    heterogeneous source records can place non-drug experimental entities under a
    DRUG label.
    """
    evidence: List[str] = []
    if _nonempty_identity_value(row.get("ChEMBLID")):
        evidence.append("chembl_id")
    cid = row.get("PubChemCID")
    if _nonempty_identity_value(cid):
        try:
            if float(cid) > 0:
                evidence.append("pubchem_cid")
        except Exception:
            evidence.append("pubchem_cid")
    if _nonempty_identity_value(row.get("InChIKey")):
        evidence.append("inchikey")
    if _nonempty_identity_value(row.get("DrugType")):
        evidence.append("drug_type")

    mw = safe_float(row.get("MW"), 0.0)
    molecular_fields = [safe_float(row.get("TPSA"), float("nan")), safe_float(row.get("LogP"), float("nan")), safe_float(row.get("QED"), float("nan"))]
    if math.isfinite(mw) and mw > 0 and any(math.isfinite(v) for v in molecular_fields):
        evidence.append("structured_molecular_properties")
    if math.isfinite(mw) and mw > 0 and bool(safe_float(row.get("Inorganic"), 0.0)):
        evidence.append("inorganic_molecular_record")
    return unique_keep_order(evidence)


def passes_pharmacological_identity_gate(row: Dict[str, Any]) -> bool:
    evidence = pharmacological_identity_evidence(row)
    row["pharmacological_identity_evidence"] = evidence
    row["pharmacological_identity_gate"] = bool(evidence)
    return bool(evidence)


def _sparse_tested_indication_phase_weight(max_phase: Any) -> float:
    phase = max_numeric_value(max_phase, default=0.0)
    if phase >= 4.0:
        return SPARSE_TESTED_INDICATION_PHASE_WEIGHTS[4.0]
    if phase >= 3.0:
        return SPARSE_TESTED_INDICATION_PHASE_WEIGHTS[3.0]
    if phase >= 2.0:
        return SPARSE_TESTED_INDICATION_PHASE_WEIGHTS[2.0]
    if phase >= 1.0:
        return SPARSE_TESTED_INDICATION_PHASE_WEIGHTS[1.0]
    if phase >= 0.5:
        return SPARSE_TESTED_INDICATION_PHASE_WEIGHTS[0.5]
    return SPARSE_TESTED_INDICATION_PHASE_WEIGHTS[0.0]


def _sparse_drug_action_weight(row: Dict[str, Any]) -> float:
    rel = normalize_name(row.get("first_rel"))
    relset = normalize_relation_set(row.get("drug_gene_relset"))
    if rel:
        relset.add(rel)

    explicit_scores = [
        float(SPARSE_DRUG_ACTION_TYPE_WEIGHTS[value])
        for value in _compatible_action_type_candidates(
            rel,
            row.get("action_type"),
            row.get("drug_gene_action_types"),
        )
        if value in SPARSE_DRUG_ACTION_TYPE_WEIGHTS
    ]
    if explicit_scores:
        weight = max(explicit_scores)
    else:
        weight = float(SPARSE_DRUG_TARGET_ACTION_REL_WEIGHTS.get(rel, 0.0))

    if rel == "target" and (relset & DRUG_GENE_DIRECTIONAL_RELS):
        directional_scores = [
            max(
                [
                    float(SPARSE_DRUG_ACTION_TYPE_WEIGHTS[value])
                    for value in _compatible_action_type_candidates(
                        directional_rel,
                        row.get("action_type"),
                        row.get("drug_gene_action_types"),
                    )
                    if value in SPARSE_DRUG_ACTION_TYPE_WEIGHTS
                ]
                or [float(SPARSE_DRUG_TARGET_ACTION_REL_WEIGHTS.get(directional_rel, 0.0))]
            )
            for directional_rel in (relset & DRUG_GENE_DIRECTIONAL_RELS)
        ]
        if directional_scores:
            weight = max(weight, max(directional_scores) - 0.02)

    if "target" in relset and rel in DRUG_GENE_DIRECTIONAL_RELS:
        weight += 0.04
    if "target" in relset and rel in DRUG_GENE_ADME_RELS:
        weight += 0.02
    if {"activation", "inhibition"}.issubset(relset):
        weight -= 0.03
    return round(max(0.0, weight), 4)


def _sparse_disease_drug_prior_weight(row: Dict[str, Any]) -> float:
    rel = normalize_name(row.get("first_rel"))
    relset = normalize_relation_set(row.get("disease_drug_relset"))
    if rel:
        relset.add(rel)

    if rel == "tested_indication":
        phase = first_not_none(
            row.get("relation_max_phase"),
            row.get("tested_indication_phases"),
        )
        weight = _sparse_tested_indication_phase_weight(phase)
    else:
        weight = float(SPARSE_DISEASE_DRUG_PRIOR_REL_WEIGHTS.get(rel, 0.0))

    if "contraindication" in relset and (relset & DISEASE_DRUG_POSITIVE_RELS) and rel != "contraindication":
        weight -= 0.25
    return round(weight, 4)


def _sparse_pattern_score(row: Dict[str, Any], pattern: str) -> float:
    base = {"direct_gene_action": 1.1, "ppi_gene_action": 0.65, "sibling_gene_action": 0.55, "family_clinical_prior": 0.10}.get(pattern, 0.0)
    if pattern in {"sibling_gene_action", "family_clinical_prior"}:
        specificity = family_hierarchy_specificity(row)
        row["family_hierarchy_specificity"] = specificity
        base *= specificity
    if pattern == "family_clinical_prior":
        rel_score = _sparse_disease_drug_prior_weight(row)
    else:
        rel_score = _sparse_drug_action_weight(row)
    score = base + rel_score - structural_degree_penalty(row)
    # Translational properties are applied once after sparse rescue.
    return round(score, 6)


def _sparse_tier(pattern: str, score: float, row: Optional[Dict[str, Any]] = None) -> str:
    if pattern == "family_clinical_prior":
        return "weak"
    if pattern == "sibling_gene_action" and has_only_adme_drug_gene_evidence(row or {}):
        return "weak"
    if pattern == "direct_gene_action" and score >= 2.0:
        return "strong"
    if pattern in {"direct_gene_action", "ppi_gene_action", "sibling_gene_action"} and score >= 1.0:
        return "medium"
    return "weak"


def find_area_sparse_rescue_candidates(
    driver,
    disease_name: str,
    area: str,
    timeout_sec: int = 25,
    verbose: bool = False,
    anchor_genes: Optional[Iterable[str]] = None,
    force_candidate_floor: bool = False,
) -> List[Dict[str, Any]]:
    area_norm = normalize_area_name(area)
    policy = _effective_sparse_policy(area_norm, force_candidate_floor=force_candidate_floor)
    if not policy.get("enabled", False):
        return []
    direct_gene_count = _sparse_direct_gene_count(driver, disease_name, timeout_sec=min(timeout_sec, 20))
    trigger_max = policy.get("max_direct_gene_count")
    enabled_patterns = list(policy.get("enabled_patterns") or [])
    if not force_candidate_floor and trigger_max is not None and direct_gene_count > int(trigger_max):
        enabled_patterns = [p for p in enabled_patterns if p in set(policy.get("dense_allowed_patterns") or [])]
    queries = _sparse_queries()
    evidence_by_drug: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    identity_rejected = 0
    identity_rejected_names: List[str] = []
    for pattern in enabled_patterns:
        q = queries.get(pattern)
        if not q:
            continue
        limit = int((policy.get("limits") or {}).get(pattern, 200))
        try:
            rows = run_query_auto(
                driver,
                q,
                {"disease": disease_name, "drug_gene_rels": DRUG_GENE_RELS, "limit": limit},
                timeout_sec=int((policy.get("timeouts") or {}).get(pattern, timeout_sec)),
            )
        except TimeoutError as exc:
            print(f"[SPARSE RESCUE TIMEOUT] disease={disease_name} pattern={pattern}: {exc}", flush=True)
            rows = []
        except Exception as exc:
            print(f"[SPARSE RESCUE ERROR] disease={disease_name} pattern={pattern}: {type(exc).__name__}: {exc}", flush=True)
            rows = []
        rows = canonicalize_path_rows(rows, pattern)
        for row in rows:
            drug = str(row.get("drug") or "").strip()
            if not drug:
                continue
            if not passes_pharmacological_identity_gate(row):
                identity_rejected += 1
                identity_rejected_names.append(drug)
                continue
            if pattern == "sibling_gene_action" and has_only_adme_drug_gene_evidence(row):
                # Retain as weak context rather than allowing it to verify a candidate alone.
                row["adme_only_family_evidence"] = True
            score = _sparse_pattern_score(row, pattern)
            evidence_by_drug[normalize_name(drug)].append({**row, "pattern": pattern, "score": score})

    if identity_rejected:
        print(
            f"[SPARSE IDENTITY GATE] disease={disease_name} rejected_rows={identity_rejected} "
            f"rejected_entities={len(set(normalize_name(x) for x in identity_rejected_names))}",
            flush=True,
        )

    candidates: List[Dict[str, Any]] = []
    for key, evidence_rows in evidence_by_drug.items():
        accepted_rows = filter_family_only_evidence_paths(evidence_rows)
        if not accepted_rows:
            continue
        accepted_rows.sort(key=lambda r: (-safe_float(r.get("score"), -999.0), _family_evidence_support_key(r)))
        best = accepted_rows[0]
        drug = str(best.get("drug") or "").strip()
        score = safe_float(best.get("score"), 0.0)
        tier = _sparse_tier(str(best.get("pattern") or ""), score, best)
        if all(has_only_adme_drug_gene_evidence(r) for r in accepted_rows if str(r.get("pattern")) == "sibling_gene_action") and any(str(r.get("pattern")) == "sibling_gene_action" for r in accepted_rows):
            tier = "weak"
        sparse_patterns = unique_keep_order([str(r.get("pattern") or "") for r in accepted_rows if str(r.get("pattern") or "").strip()])
        candidates.append({
            "drug": drug,
            "candidate_score": score,
            "candidate_source": "sparse_area_rescue",
            "candidate_sources": ["sparse_area_rescue"],
            "supporting_anchor_branches": ["sparse_area_rescue"],
            "supporting_axes": ["topology_only_sparse_rescue"],
            "supporting_patterns": sparse_patterns,
            "mechanism_axis": "topology_only_sparse_rescue",
            "area_sparse_rescue_score": score,
            "area_sparse_rescue_pattern": best.get("pattern"),
            "area_sparse_rescue_tier": tier,
            "sparse_rescue_tier": tier,
            "sparse_rescue_source_area": area_norm,
            "area_sparse_rescue_evidence": best,
            "area_sparse_rescue_support": accepted_rows[1:],
            "best_anchor_path": best,
            "source_graph_candidate": True,
            "source_sparse_rescue": True,
            "direct_disease_gene_count": direct_gene_count,
            "family_evidence_independent_supports": max((int(r.get("family_evidence_independent_supports") or 0) for r in accepted_rows), default=0),
            "family_evidence_required_supports": max((int(r.get("family_evidence_required_supports") or 0) for r in accepted_rows), default=0),
            "family_evidence_support_diseases": best.get("family_evidence_support_diseases", []),
            "family_evidence_prior_support_diseases": best.get("family_evidence_prior_support_diseases", []),
            "family_evidence_prior_corroboration_units": best.get("family_evidence_prior_corroboration_units", 0),
            "family_parent_name": best.get("family_parent_name"),
            "family_evidence_cluster_key": best.get("family_evidence_cluster_key"),
            "pharmacological_identity_gate": bool(best.get("pharmacological_identity_gate")),
            "pharmacological_identity_evidence": best.get("pharmacological_identity_evidence", []),
        })
    candidates.sort(key=lambda c: (-safe_float(c.get("area_sparse_rescue_score")), normalize_name(c.get("drug"))))
    for i, c in enumerate(candidates, 1):
        c["rank"] = i
    return candidates


def apply_area_sparse_tail_rescue(
    driver,
    disease_name: str,
    area: str,
    candidates: List[Dict[str, Any]],
    protect_top_n: Optional[int] = None,
    top_k: int = 100,
    timeout_sec: int = 25,
    anchor_genes: Optional[Iterable[str]] = None,
    force_candidate_floor: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    area_norm = normalize_area_name(area)
    policy = _effective_sparse_policy(area_norm, force_candidate_floor=force_candidate_floor)
    base = copy.deepcopy(candidates or [])
    sparse = find_area_sparse_rescue_candidates(
        driver, disease_name, area_norm, timeout_sec, False, anchor_genes,
        force_candidate_floor=force_candidate_floor,
    )
    existing = {normalize_name(c.get("drug")) for c in base}
    sparse = [c for c in sparse if normalize_name(c.get("drug")) not in existing]
    insert_after = dict(SPARSE_RESCUE_TIER_INSERT_AFTER_DEFAULT)
    insert_after.update(policy.get("tier_insert_after") or {})
    tier_caps = dict(SPARSE_RESCUE_TIER_MAX_DEFAULT)
    tier_caps.update(policy.get("tier_max") or {})
    if protect_top_n is not None:
        insert_after["weak"] = max(int(insert_after.get("weak", 90)), int(protect_top_n))
    selected = []
    for tier in ("strong", "medium", "weak"):
        cap = max(0, int(tier_caps.get(tier, 0)))
        tier_rows = [c for c in sparse if c.get("area_sparse_rescue_tier") == tier][:cap]
        if tier == "weak":
            # Weak rescue is fill-only: it may occupy vacant top-K tail slots,
            # but it never displaces an already ranked candidate.
            vacant = max(0, int(top_k) - len(base))
            tier_rows = tier_rows[:vacant]
            for row in tier_rows:
                base.append(copy.deepcopy(row))
        else:
            at = min(max(0, int(insert_after.get(tier, len(base)))), len(base))
            for offset, row in enumerate(tier_rows):
                base.insert(min(at + offset, len(base)), copy.deepcopy(row))
        selected.extend(tier_rows)
    dedup = []
    seen = set()
    for row in base:
        key = normalize_name(row.get("drug"))
        if not key or key in seen:
            continue
        seen.add(key)
        dedup.append(row)
        if len(dedup) >= int(top_k):
            break
    for i, c in enumerate(dedup, 1):
        c["rank"] = i
    report = {
        "enabled": bool(policy.get("enabled", False)),
        "status": "ok",
        "area": area_norm,
        "basis": "aggregate_kg_topology_only",
        "trigger": "candidate_floor" if force_candidate_floor else "area_sparse_policy",
        "candidate_floor": int(GLOBAL_CANDIDATE_FLOOR),
        "candidate_count_before_rescue": len(candidates or []),
        "n_sparse_candidates": len(sparse),
        "n_selected": len(selected),
        "selected_drugs": [c.get("drug") for c in selected],
        "selected_patterns": [c.get("area_sparse_rescue_pattern") for c in selected],
        "selected_tiers": [c.get("area_sparse_rescue_tier") for c in selected],
        "uses_disease_name_rules": False,
        "uses_drug_or_gene_allowlists": False,
        "weak_rescue_fill_only": True,
    }
    return dedup, report

        

# ============================================================
# Stable finalization
# ============================================================

def apply_graphdrx_area_final_ranking(
    mechanism_obj: Dict[str, Any],
    area: str,
    cardio_head_n: int = 60,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = copy.deepcopy(mechanism_obj.get("candidate_drugs", []) or [])
    out, seen = [], set()
    for row in rows:
        key = normalize_name(row.get("drug"))
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(row)
    for i, row in enumerate(out, 1):
        row["rank"] = i
    return out, {
        "enabled": True,
        "area": normalize_area_name(area),
        "policy": "stable_deduplication_no_area_score",
        "area_used_for_scoring": False,
        "top10": [c.get("drug") for c in out[:10]],
    }



def _candidate_branch_names(candidate: Dict[str, Any]) -> List[str]:
    """Collect normalized candidate-source labels without using evaluation labels."""
    values: List[Any] = []
    for field in (
        "candidate_sources",
        "candidate_source",
        "supporting_anchor_branches",
        "supporting_branches",
    ):
        value = candidate.get(field)
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            values.extend(value)
        else:
            values.append(value)

    names: List[str] = []
    seen = set()
    for value in values:
        text = str(value or "").strip().lower()
        if not text:
            continue
        # Legacy combined source strings may use +, |, or comma separators.
        for part in re.split(r"[+|,]", text):
            part = part.strip()
            if part and part not in seen:
                seen.add(part)
                names.append(part)
    if bool(candidate.get("source_disease_context_fallback_graph")) and "disease_context_fallback_graph" not in seen:
        names.append("disease_context_fallback_graph")
        seen.add("disease_context_fallback_graph")
    if bool(candidate.get("source_vector_neighbor_anchor_graph")) and "vector_neighbor_anchor_graph" not in seen:
        names.append("vector_neighbor_anchor_graph")
        seen.add("vector_neighbor_anchor_graph")
    return names



def _candidate_structural_branch_flags(
    candidate: Dict[str, Any],
    prior_threshold: float = 1e-12,
) -> Dict[str, bool]:
    """Return gold-independent T/B/Cp/D provenance flags."""
    branches = set(_candidate_branch_names(candidate))
    anchor_support = bool(
        candidate.get("source_disease_context_fallback_graph")
        or branches.intersection({
            "disease_context_fallback_graph",
            "disease_context_anchor_graph",
            "target_context_anchor_graph",
        })
    )
    vector_support = bool(
        candidate.get("source_vector_neighbor_anchor_graph")
        or branches.intersection({"vector_neighbor_anchor_graph", "vector_anchor_graph"})
    )
    prior_support = safe_float(candidate.get("embedding_prior_score"), 0.0) > float(prior_threshold)
    direct_support = bool(
        candidate.get("source_common_direct_graph")
        or branches.intersection({"common_direct_graph", "target_disease_direct_gene_action_graph"})
    )
    return {
        "anchor": anchor_support,
        "vector": vector_support,
        "prior": prior_support,
        "direct": direct_support,
    }




def apply_structural_evidence_ranking(
    candidates: List[Dict[str, Any]],
    prior_threshold: float = 1e-12,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Apply the final stable core/reserve ordering used by GraphDRx.

    Exact T-only and exact B-only candidates are moved to a reserve pool. All
    other candidates remain in the core pool. Numeric scores are unchanged and
    existing order is preserved within each pool.
    """
    if prior_threshold < 0:
        raise ValueError("structural reserve prior threshold must be non-negative")

    rows = copy.deepcopy(candidates or [])
    group_counts: Counter = Counter()
    reason_counts: Counter = Counter()

    for original_index, row in enumerate(rows, start=1):
        flags = _candidate_structural_branch_flags(row, prior_threshold=prior_threshold)
        target_only = flags["anchor"] and not flags["vector"] and not flags["prior"] and not flags["direct"]
        vector_only = flags["vector"] and not flags["anchor"] and not flags["prior"] and not flags["direct"]
        is_reserve = bool(target_only or vector_only)
        tier = 1 if is_reserve else 0
        group = "reserve" if is_reserve else "core"
        if target_only:
            reason = "single_target_context_branch_only"
        elif vector_only:
            reason = "single_vector_branch_only"
        else:
            reason = "prior_direct_or_multibranch_support"

        signature_parts = []
        if flags["anchor"]:
            signature_parts.append("T")
        if flags["vector"]:
            signature_parts.append("B")
        if flags["prior"]:
            signature_parts.append("Cp")
        if flags["direct"]:
            signature_parts.append("D")

        row["rank_before_structural_evidence_ranking"] = int(row.get("rank") or original_index)
        row["structural_evidence_ranking_enabled"] = True
        row["structural_evidence_ranking_mode"] = "single_indirect_only_reserve"
        row["structural_evidence_tier"] = int(tier)
        row["structural_evidence_group"] = group
        row["structural_evidence_reason"] = reason
        row["structural_evidence_signature"] = "+".join(signature_parts) if signature_parts else "NONE"
        row["source_anchor_branch"] = bool(flags["anchor"])
        row["source_vector_branch"] = bool(flags["vector"])
        row["source_semantic_prior_support"] = bool(flags["prior"])
        row["source_direct_gene_action_branch"] = bool(flags["direct"])
        row["source_prior_graph_verified"] = bool(row.get("source_prior_graph_verified"))
        row["prior_verification_status"] = (
            "explicitly_graph_verified"
            if row.get("source_prior_graph_verified")
            else "skipped_existing_graph_path"
            if flags["prior"] and (flags["anchor"] or flags["vector"] or flags["direct"])
            else "not_applicable"
        )
        row["_structural_sort_tier"] = int(tier)
        row["_structural_original_index"] = int(original_index)
        group_counts[group] += 1
        reason_counts[reason] += 1

    rows.sort(key=lambda r: (int(r.get("_structural_sort_tier", 0)), int(r.get("_structural_original_index", 0))))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        row["rank_shift_structural_evidence_ranking"] = int(row.get("rank_before_structural_evidence_ranking") or rank) - rank
        row.pop("_structural_sort_tier", None)
        row.pop("_structural_original_index", None)

    report = {
        "enabled": True,
        "mode": "single_indirect_only_reserve",
        "policy": "stable_core_then_single_indirect_reserve",
        "stage": "after_sparse_rescue_and_stable_dedup_before_eval1",
        "prior_definition": "embedding_prior_score > prior_threshold",
        "prior_threshold": float(prior_threshold),
        "candidate_count": len(rows),
        "group_counts": dict(sorted(group_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "deletes_candidates": False,
        "changes_numeric_scores": False,
        "preserves_order_within_each_pool": True,
        "uses_gold_labels": False,
        "issues_additional_neo4j_queries": False,
    }
    return rows, report


def summarize_drug_embedding_rerank_status(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize whether drug-embedding rerank was effectively active.

    Some lower-level embedding failures are recorded per candidate and do not
    raise out of apply_drug_embedding_rerank_for_case. This helper prevents
    such runs from being reported as "ok" when every candidate failed to
    obtain an active drug embedding signal.
    """
    rows = list(candidates or [])
    total = len(rows)
    errors = [str(r.get("drug_embedding_error")) for r in rows if r.get("drug_embedding_error")]
    n_error = len(errors)
    n_success_flag = sum(1 for r in rows if bool(r.get("source_drug_embedding")) and not r.get("drug_embedding_error"))
    n_nonzero_similarity = 0
    n_similarity_present = 0
    for r in rows:
        if "drug_embedding_similarity" in r:
            n_similarity_present += 1
        try:
            if abs(float(r.get("drug_embedding_similarity") or 0.0)) > 1e-12:
                n_nonzero_similarity += 1
        except Exception:
            pass

    unique_errors = sorted(set(errors))[:5]
    if total == 0:
        status = "ok_empty_candidate_list"
    elif n_success_flag > 0 and n_error == 0:
        status = "ok"
    elif n_success_flag > 0 and n_error > 0:
        status = "partial"
    elif n_error > 0:
        status = "failed_no_active_embeddings"
    elif n_similarity_present > 0 and n_nonzero_similarity == 0:
        status = "ok_zero_similarity"
    else:
        status = "no_embedding_attempt_recorded"

    return {
        "status": status,
        "candidate_count": total,
        "success_count": n_success_flag,
        "error_count": n_error,
        "similarity_present_count": n_similarity_present,
        "nonzero_similarity_count": n_nonzero_similarity,
        "effectively_active": bool(n_success_flag > 0 or n_nonzero_similarity > 0),
        "unique_errors": unique_errors,
    }


def run_pilot(
    ablation_mode: str = ABLATION_MODE,
    neo4j_password: str = NEO4J_PASSWORD,
    test_cases: Optional[List[Dict[str, Any]]] = None,
    drug_rag_csv: str = DRUG_RAG_CSV,
    disease_features_csv: str = DISEASE_RAG_CSV,
    use_drug_embedding_rerank: bool = USE_DRUG_EMBEDDING_RERANK,
    drug_embedding_weight: float = DRUG_EMBEDDING_WEIGHT,
    use_general_offline_rerank: bool = USE_GENERAL_OFFLINE_RERANK,
    offline_rerank_drug_weight: float = OFFLINE_RERANK_DRUG_SIM_WEIGHT,
    output_tag_suffix: str = "",
    use_mechanism_lookup: bool = USE_MECHANISM_LOOKUP,
    use_disease_embedding_prior: bool = USE_DISEASE_EMBEDDING_PRIOR,
    use_target_context_graph: bool = True,
    use_vector_anchor_graph: bool = True,
    use_prior_drug_graph: bool = True,
    use_semantic_prior_branch: bool = True,
    use_common_direct_graph: bool = True,
    structural_reserve_prior_threshold: float = 1e-12,
    eval_diseases_for_prior_mask: Optional[List[str]] = None,
    mask_eval_diseases_from_prior: bool = False,
    max_similarity_for_prior: Optional[float] = None,
    disease_prior_mode: str = "vector_anchor_verify",
    prior_verify_top_n: int = 120,
    prior_verify_limit_per_pattern: int = 5,
    keep_unverified_prior_candidates: bool = False,
    vector_anchor_top_diseases: int = 30,
    vector_anchor_max_genes: int = 100,
    vector_anchor_max_terms: int = 140,
    enable_sparse_area_rescue: bool = False,
    sparse_rescue_areas: Optional[List[str]] = None,
    sparse_rescue_protect_top_n: Optional[int] = None,
    resume_existing: bool = False,
):
    global CURRENT_CASE_AREA
    driver = make_driver(neo4j_password=neo4j_password)
    run_tag = f"graphdrx_top{CANDIDATE_K}_{ablation_mode}{output_tag_suffix}"
    out_dir = OUT_BASE_DIR / run_tag
    _graphdrx_forced_output_dir = os.getenv("GRAPHDRX_FORCE_OUTPUT_DIR")
    if _graphdrx_forced_output_dir:
        out_dir = Path(_graphdrx_forced_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_tag = f"graphdrx_{ablation_mode}{output_tag_suffix}"
    all_results_path = out_dir / f"{file_tag}_all_results.json"
    results_jsonl_path = out_dir / f"{file_tag}_results.jsonl"
    rag_context_path = out_dir / f"{file_tag}_rag_context.json"
    compact_candidates_path = out_dir / f"{file_tag}_top{OUTPUT_COMPACT_TOP_K}_candidates.csv"
    summary_path = out_dir / f"{file_tag}_summary.csv"
    metrics_path = out_dir / f"{file_tag}_metrics.json"

    all_results = []
    case_metrics = []
    rag_context_by_disease = {}
    
    disease_df = pd.read_csv(disease_features_csv)
    disease_name_idf = build_disease_name_idf_from_df(disease_df)
    drug_df = load_drug_rag_df(drug_rag_csv)
    cases_to_run = list(test_cases or [])

    if not cases_to_run:
        raise ValueError("No test cases to run. Check --test-cases-csv and --area-filter.")

    if resume_existing and (results_jsonl_path.exists() or all_results_path.exists()):
        try:
            if results_jsonl_path.exists():
                all_results = load_jsonl_records(results_jsonl_path)
                resume_source = results_jsonl_path
                if rag_context_path.exists():
                    rag_context_by_disease = json.loads(rag_context_path.read_text(encoding="utf-8"))
            else:
                previous_payload = json.loads(all_results_path.read_text(encoding="utf-8"))
                all_results = [r for r in (previous_payload.get("results") or []) if isinstance(r, dict)]
                rag_context_by_disease = dict(previous_payload.get("rag_context_by_disease") or {})
                resume_source = all_results_path
            case_metrics = [r.get("metrics") for r in all_results if isinstance(r.get("metrics"), dict)]
            completed = {
                re.sub(r"\s+", " ", str(r.get("disease") or "").strip().lower())
                for r in all_results
                if str(r.get("disease") or "").strip()
            }
            before = len(cases_to_run)
            cases_to_run = [
                c for c in cases_to_run
                if re.sub(r"\s+", " ", str(c.get("disease") or "").strip().lower()) not in completed
            ]
            print(
                f"[RESUME] loaded {len(all_results)} completed cases from {resume_source}; "
                f"remaining={len(cases_to_run)}/{before}"
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to resume GraphDRx output: {exc}") from exc
    elif not resume_existing:
        for stale in (results_jsonl_path, rag_context_path, compact_candidates_path):
            stale.unlink(missing_ok=True)

    try:
        for idx, case in enumerate(cases_to_run, start=1):
            input_disease = str(case["disease"] or "").strip()
            resolved_disease = resolve_exact_disease_node(driver, input_disease)
            disease = str(resolved_disease["disease_name"])
            case = dict(case)
            case["disease"] = disease
            case["_disease_element_id"] = resolved_disease.get("element_id")
            case["_disease_exact_element_ids"] = resolved_disease.get("exact_match_element_ids") or [resolved_disease.get("element_id")]
            case_id = case.get("case_id", disease.replace(" ", "_"))
            CURRENT_CASE_AREA = normalize_area_name(case.get("area") or "")
            print(
                f"[DISEASE RESOLUTION] input={input_disease!r} -> exact={disease!r} "
                f"node_index={resolved_disease.get('node_index')} "
                f"node_id={resolved_disease.get('node_id')} "
                f"element_id={resolved_disease.get('element_id')} "
                f"exact_duplicate_count={resolved_disease.get('duplicate_exact_name_count', 1)}"
            )
            if int(resolved_disease.get("duplicate_exact_name_count") or 1) > 1:
                print(
                    "[DISEASE RESOLUTION] duplicate exact-name nodes merged as one virtual bundle; "
                    f"primary_cache_node={resolved_disease.get('element_id')} "
                    f"bundle={resolved_disease.get('exact_match_element_ids')}"
                )
            # Retrieval and reranking parameters are identical across areas.
            eff_vector_anchor_top_diseases = vector_anchor_top_diseases
            eff_vector_anchor_max_genes = vector_anchor_max_genes
            eff_vector_anchor_max_terms = vector_anchor_max_terms

            # Direct disease-drug names are loaded only as text-redaction terms.
            # Evaluation labels are not constructed until ranking has finished.
            redaction_drugs = get_target_disease_redaction_drugs(
                driver,
                disease,
                disease_element_id=resolved_disease.get("element_id"),
                disease_element_ids=resolved_disease.get("exact_match_element_ids"),
            )
            case_for_prior = {
                "disease": disease,
                "area": CURRENT_CASE_AREA,
                "case_id": case_id,
                "_redaction_drug_terms": redaction_drugs,
                "_disease_element_id": resolved_disease.get("element_id"),
                "_disease_exact_element_ids": resolved_disease.get("exact_match_element_ids") or [resolved_disease.get("element_id")],
            }

            if use_disease_embedding_prior:
                embedding_prior = prepare_embedding_prior_for_case(
                    driver=driver,
                    case=case_for_prior,
                    disease_df=disease_df,
                    disease_features_csv=disease_features_csv,
                    top_similar_diseases=50,
                    min_similarity=0.85,
                    force_rebuild_embedding=False,
                    eval_diseases=eval_diseases_for_prior_mask or [],
                    mask_eval_diseases_from_prior=mask_eval_diseases_from_prior,
                    max_similarity_for_prior=max_similarity_for_prior,
                )
            else:
                embedding_prior = {
                    "similar_diseases": [],
                    "candidate_priors": [],
                }

            embedding_prior["neighbor_policy"] = {
                "mode": "embedding_similarity_only",
                "uses_disease_name_rules": False,
                "uses_area_keyword_rules": False,
                "target_disease_drugs_used_for_redaction_only": True,
                "redaction_term_count": len(redaction_drugs),
            }

            print("\n[EMBEDDING PRIOR - SIMILAR DISEASES]")
            for r in embedding_prior.get("similar_diseases", [])[:10]:
                print(f"  {r.get('similarity_score'):.4f} | {r.get('disease')}")
            
            print("\n[EMBEDDING PRIOR - TOP PRIOR DRUGS]")
            for r in embedding_prior.get("candidate_priors", [])[:20]:
                print(
                    f"  {r.get('drug')} | "
                    f"emb_prior={r.get('embedding_prior_score')} | "
                    f"rels={r.get('prior_rels')} | "
                    f"near={r.get('prior_diseases', [])[:3]}"
                )

            disease_prior_mode_norm = str(disease_prior_mode or "vector_anchor_verify").lower()

            t0 = time.time()

            disease_context = get_disease_context(
                driver,
                disease,
                disease_element_id=resolved_disease.get("element_id"),
                disease_element_ids=resolved_disease.get("exact_match_element_ids"),
            )
            
            disease_rag_row = get_disease_rag_row(
                disease=disease,
                disease_df=disease_df,
            )
            
            safe_disease_rag_text = build_safe_disease_rag_text(
                disease_row=disease_rag_row,
                gold_drugs=redaction_drugs,
                max_chars_per_field=1200,
            )
            
            disease_context["safe_disease_rag_text"] = safe_disease_rag_text
            disease_context["disease_rag_match"] = {
                "mondo_name": disease_rag_row.get("mondo_name") if disease_rag_row else None,
                "group_name_bert": disease_rag_row.get("group_name_bert") if disease_rag_row else None,
            }

            rag_context_by_disease[disease] = {
                "case_index": idx,
                "disease": disease,
                "input_disease": input_disease,
                "disease_resolution": resolved_disease,
                "matched_disease_names": disease_context.get("matched_disease_names", []),
                "disease_rag_match": disease_context.get("disease_rag_match", {}),
                "disease_context": disease_context,
            }

            # 1) Deterministic target-disease biological-context anchors (T).
            print("\n[TARGET CONTEXT] building deterministic disease-context anchors...")
            anchor_obj = build_target_context_anchors(
                disease=disease,
                disease_context=disease_context,
                case_area=case.get("area") or "",
                disease_name_idf=disease_name_idf,
            )
            anchor_obj = resolve_mechanism_axes_against_kg(
                driver=driver,
                mechanism_anchor_obj=anchor_obj,
                timeout_sec=20,
            )
            resolution_report = anchor_obj.get("anchor_kg_resolution", {})
            if (
                resolution_report.get("status") == "ok"
                and int(resolution_report.get("n_axes_with_any_resolved_anchor") or 0) == 0
            ):
                anchor_obj = build_target_context_anchors(
                    disease=disease,
                    disease_context=disease_context,
                    case_area=case.get("area") or "",
                    disease_name_idf=disease_name_idf,
                )
                anchor_obj["anchor_kg_resolution"] = {
                    **resolution_report,
                    "status": "no_resolved_axis_original_context_retained",
                }
                resolution_report = anchor_obj["anchor_kg_resolution"]
            print(
                "[TARGET CONTEXT KG RESOLUTION] "
                f"status={resolution_report.get('status')} | "
                f"resolved_axes={resolution_report.get('n_axes_with_any_resolved_anchor')} | "
                f"genes={resolution_report.get('n_resolved_genes')} | "
                f"bp={resolution_report.get('n_resolved_bp_terms')} | "
                f"path={resolution_report.get('n_resolved_path_terms')}",
                flush=True,
            )

            vector_mode_requested = disease_prior_mode_norm in {"vector_anchor", "vector_anchor_verify"}
            prior_drug_mode_requested = disease_prior_mode_norm in {"verify", "vector_anchor_verify"}
            effective_target_context_graph = bool(use_target_context_graph)
            vector_anchor_context_available = bool(
                use_disease_embedding_prior and vector_mode_requested
            )
            effective_vector_anchor_graph = bool(
                vector_anchor_context_available and use_vector_anchor_graph
            )
            effective_semantic_prior_branch = bool(
                use_disease_embedding_prior and use_semantic_prior_branch
            )
            effective_prior_drug_graph = bool(
                effective_semantic_prior_branch and use_prior_drug_graph and prior_drug_mode_requested
            )
            effective_common_direct_graph = bool(use_common_direct_graph)

            if vector_anchor_context_available:
                print("\n[VECTOR-NEIGHBOR CONTEXT] building ordered KG anchor pool...")
                anchor_obj = augment_mechanism_anchors_with_vector_context(
                    driver=driver,
                    disease=disease,
                    mechanism_anchor_obj=anchor_obj,
                    embedding_prior=embedding_prior,
                    top_n=eff_vector_anchor_top_diseases,
                    max_genes=eff_vector_anchor_max_genes,
                    max_terms=eff_vector_anchor_max_terms,
                    case_area=case.get("area") or "",
                )
                vg = anchor_obj.get("vector_guided_graphrag", {})
                print(
                    f"[VECTOR-NEIGHBOR ANCHORS] near_diseases={len(vg.get('near_diseases', []))} | "
                    f"genes={len(vg.get('vector_genes', []))} | "
                    f"bp_terms={len(vg.get('vector_bio_terms', []))} | "
                    f"pathways={len(vg.get('vector_pathways', []))} | "
                    f"phenotypes={len(vg.get('vector_phenotypes', []))}",
                    flush=True,
                )
            else:
                anchor_obj["vector_anchor_pool"] = {}
                anchor_obj["vector_guided_graphrag"] = {
                    "enabled": False,
                    "mode": "disabled",
                    "reason": "disease_embedding_context_disabled",
                }

            print("[TARGET CONTEXT AXES]")
            for i, axis in enumerate(anchor_obj.get("mechanism_axes", []), start=1):
                print(
                    f"  axis#{i} {axis.get('axis_name')} | "
                    f"priority={axis.get('priority')} | "
                    f"genes={axis.get('anchor_genes', [])[:8]} | "
                    f"terms={axis.get('anchor_biology_terms', [])[:5]}"
                )
            selected_axes_preview = select_mechanism_axes(
                anchor_obj.get("mechanism_axes", []) or [], max_axes=3
            )
            print("[SELECTED TARGET CONTEXT AXES]")
            for axis in selected_axes_preview:
                print(
                    f"  selected#{axis.get('axis_selection_rank')} {axis.get('axis_name')} | "
                    f"priority={axis.get('priority')} | "
                    f"overlap={axis.get('max_overlap_with_selected_axes')}"
                )

            # Branch 1-1 and 1-2: explicit anchor-to-drug graph retrieval.
            print("\n[ANCHOR-GUIDED GRAPHRAG] retrieving target-context and vector-neighbor branches...")
            retrieval_obj = retrieve_anchor_guided_drug_candidates(
                driver=driver,
                disease=disease,
                mechanism_anchor_obj=anchor_obj,
                candidate_k=CANDIDATE_K,
                limit_per_axis_pattern=PATH_LIMIT_PER_PATTERN,
                ablation_mode=ablation_mode,
                use_mechanism_lookup=use_mechanism_lookup,
                case_area=case.get("area") or "",
                use_target_context_graph=effective_target_context_graph,
                use_vector_anchor_graph=effective_vector_anchor_graph,
                use_common_direct_graph=effective_common_direct_graph,
            )

            for c in retrieval_obj.get("candidate_drugs", [])[:100]:
                print(
                    f"  #{c['rank']} {c['drug']} | score={c.get('candidate_score')} | "
                    f"branches={c.get('candidate_sources') or c.get('supporting_anchor_branches') or []} | "
                    f"axis={c.get('mechanism_axis')}"
                )

            # Branch 2-1: similar-disease clinical drug seeds, independently verified
            # against the target disease using bounded target-disease graph patterns.
            retrieval_obj["disease_prior_mode"] = disease_prior_mode_norm
            retrieval_obj["vector_guided_graphrag"] = anchor_obj.get(
                "vector_guided_graphrag", {"enabled": False}
            )
            retrieval_obj["retrieval_branches"] = {
                "target_context_graph": effective_target_context_graph,
                "vector_neighbor_context_available": vector_anchor_context_available,
                "vector_neighbor_anchor_graph": effective_vector_anchor_graph,
                "common_direct_disease_gene_graph": effective_common_direct_graph,
                "semantic_prior_support": effective_semantic_prior_branch,
                "prior_drug_target_graph_verification": effective_prior_drug_graph,
                "contraindication_only_prior_seed_allowed": False,
            }

            if effective_prior_drug_graph:
                print("\n[PRIOR-DRUG GRAPH VERIFY] verifying similar-disease clinical drug seeds...")
                verified_prior_candidates, summary_general = verify_embedding_prior_candidates_as_graphrag_seeds(
                    driver=driver,
                    disease=disease,
                    embedding_prior=embedding_prior,
                    mechanism_obj=retrieval_obj,
                    disease_context=disease_context,
                    top_n=prior_verify_top_n,
                    limit_per_pattern=prior_verify_limit_per_pattern,
                    skip_existing_graph_drugs=True,
                    require_vector_anchor_evidence=False,
                    verification_source_label="prior_drug_target_graph_verified",
                )
                retrieval_obj["prior_graph_verification"] = {
                    "enabled": True,
                    "mode": "independent_prior_drug_target_graph_verification",
                    "base_patterns": list(DEFAULT_PRIOR_VERIFY_PATTERNS),
                    "area_patterns": enabled_prior_verify_patterns_for_current_case(disease),
                    "all_tested_indication_phases_seed_eligible": True,
                    "contraindication_only_seed_eligible": False,
                    "summary": summary_general,
                }
                retrieval_obj["candidate_drugs_prior_graph_verified"] = verified_prior_candidates
                graph_plus_verified = deduplicate_candidates_keep_best(
                    retrieval_obj.get("candidate_drugs", []) + verified_prior_candidates
                )
                ranked_candidates = merge_graph_candidates_with_embedding_priors(
                    graph_rows=graph_plus_verified,
                    embedding_prior=embedding_prior,
                    graph_score_field="candidate_score",
                    max_prior_only=0,
                )
            elif effective_semantic_prior_branch and disease_prior_mode_norm == "union":
                # Diagnostic legacy lane only. Main runs never use unverified prior-only candidates.
                retrieval_obj["prior_graph_verification"] = {
                    "enabled": False,
                    "mode": "legacy_union_unverified_prior_candidates",
                    "warning": "Diagnostic only; not part of the GraphDRx main method.",
                }
                ranked_candidates = merge_graph_candidates_with_embedding_priors(
                    graph_rows=retrieval_obj.get("candidate_drugs", []),
                    embedding_prior=embedding_prior,
                    graph_score_field="candidate_score",
                    max_prior_only=200,
                )
            elif effective_semantic_prior_branch:
                retrieval_obj["prior_graph_verification"] = {
                    "enabled": False,
                    "mode": "prior_drug_branch_disabled",
                }
                # Attach disease-neighborhood prior scores only to graph-generated drugs;
                # never add path-free drugs in this branch.
                ranked_candidates = merge_graph_candidates_with_embedding_priors(
                    graph_rows=retrieval_obj.get("candidate_drugs", []),
                    embedding_prior=embedding_prior,
                    graph_score_field="candidate_score",
                    max_prior_only=0,
                )
            else:
                retrieval_obj["prior_graph_verification"] = {
                    "enabled": False,
                    "mode": (
                        "semantic_prior_branch_disabled"
                        if use_disease_embedding_prior and not use_semantic_prior_branch
                        else "disease_embedding_disabled"
                    ),
                }
                ranked_candidates = retrieval_obj.get("candidate_drugs", [])

            # Preserve C_p support provenance for both explicitly verified prior-only
            # candidates and graph-existing candidates whose C_v query was skipped.
            prior_rows_by_drug = {
                normalize_name(row.get("drug")): row
                for row in (embedding_prior.get("candidate_priors") or [])
                if effective_semantic_prior_branch and row.get("drug")
            }
            for c in ranked_candidates:
                prior_row = prior_rows_by_drug.get(normalize_name(c.get("drug")))
                if prior_row and safe_float(c.get("embedding_prior_score"), 0.0) > 0:
                    c["source_semantic_prior_support"] = True
                    c["prior_primary_support_diseases"] = primary_prior_support_diseases(prior_row)
                    c.setdefault("prior_diseases", prior_row.get("prior_diseases", []))
                    c.setdefault("prior_rels", prior_row.get("prior_rels", []))
                else:
                    c["source_semantic_prior_support"] = False

            for rank, c in enumerate(ranked_candidates, start=1):
                c["rank"] = rank
                c["candidate_score_original"] = c.get("candidate_score", 0.0)
                c["candidate_score"] = c.get(
                    "final_score_with_embedding_prior",
                    c.get("candidate_score", 0.0),
                )
                c["confidence"] = c.get(
                    "confidence",
                    "embedding_prior" if not c.get("source_graph_candidate") else "medium",
                )
                c["mechanism_axis"] = c.get(
                    "mechanism_axis",
                    "embedding_prior_similar_disease",
                )
            
            retrieval_obj["candidate_drugs_graphrag_only"] = retrieval_obj.get("candidate_drugs", [])
            retrieval_obj["candidate_drugs_with_embedding_prior_base"] = copy.deepcopy(ranked_candidates)

            retrieval_obj["candidate_drugs_with_embedding_prior"] = copy.deepcopy(ranked_candidates)
            retrieval_obj["candidate_drugs"] = ranked_candidates
            retrieval_obj["embedding_prior"] = {
                "similar_diseases": embedding_prior.get("similar_diseases", []),
                "top_prior_drugs": embedding_prior.get("candidate_priors", [])[:50],
            }

            # 2.6) Drug representation embedding rerank
            # Compare disease-context anchors with treatment-restricted drug embeddings.
            # This does NOT use drug indication text or target disease direct drug edges.
            if use_drug_embedding_rerank:
                retrieval_obj["candidate_drugs_before_drug_embedding"] = retrieval_obj.get("candidate_drugs", [])
            
                try:
                    ranked_with_drug_embedding = apply_drug_embedding_rerank_for_case(
                        driver=driver,
                        disease=disease,
                        candidates=retrieval_obj.get("candidate_drugs", []),
                        drug_df=drug_df,
                        mechanism_anchor_obj=anchor_obj,
                        disease_context=disease_context,
                        force_rebuild=False,
                        drug_embedding_weight=drug_embedding_weight,
                        max_candidates=DRUG_EMBEDDING_MAX_CANDIDATES,
                    )
            
                    retrieval_obj["candidate_drugs_with_drug_embedding"] = ranked_with_drug_embedding
                    retrieval_obj["candidate_drugs"] = ranked_with_drug_embedding
                    embedding_status = summarize_drug_embedding_rerank_status(ranked_with_drug_embedding)
                    retrieval_obj["drug_embedding_rerank"] = {
                        "enabled": True,
                        **embedding_status,
                        "weight": drug_embedding_weight,
                        "max_candidates": DRUG_EMBEDDING_MAX_CANDIDATES,
                        "drug_rag_csv": drug_rag_csv,
                    }
                    if not embedding_status.get("effectively_active"):
                        print(
                            f"[DRUG EMBEDDING RERANK WARNING] disease={disease} | "
                            f"status={embedding_status.get('status')} | "
                            f"errors={embedding_status.get('error_count')} | "
                            f"examples={embedding_status.get('unique_errors')}",
                            flush=True,
                        )
            
                except Exception as e:
                    print(
                        f"[DRUG EMBEDDING RERANK ERROR] "
                        f"disease={disease} | {type(e).__name__}: {e}",
                        flush=True,
                    )
            
                    # fallback: keep pre-drug-embedding ranking
                    fallback_candidates = retrieval_obj.get(
                        "candidate_drugs_before_drug_embedding",
                        retrieval_obj.get("candidate_drugs", []),
                    )
            
                    retrieval_obj["candidate_drugs_with_drug_embedding"] = fallback_candidates
                    retrieval_obj["candidate_drugs"] = fallback_candidates
                    retrieval_obj["drug_embedding_rerank"] = {
                        "enabled": True,
                        "status": "failed_fallback_to_pre_drug_embedding",
                        "error_type": type(e).__name__,
                        "error": str(e),
                        "weight": drug_embedding_weight,
                        "max_candidates": DRUG_EMBEDDING_MAX_CANDIDATES,
                        "drug_rag_csv": drug_rag_csv,
                    }
            
            else:
                retrieval_obj["drug_embedding_rerank"] = {"enabled": False}
            
            # 2.65) Final disease-agnostic general rerank.
            if use_general_offline_rerank:
                retrieval_obj["candidate_drugs_before_general_rerank"] = retrieval_obj.get("candidate_drugs", [])
                ranked_general = apply_general_offline_rerank(
                    candidates=retrieval_obj.get("candidate_drugs", []),
                    drug_sim_weight=offline_rerank_drug_weight,
                )
                retrieval_obj["candidate_drugs"] = ranked_general
                retrieval_obj["general_offline_rerank"] = {
                    "enabled": True,
                    "mode": "standard_full_reorder",
                    "offline_rerank_drug_weight": offline_rerank_drug_weight,
                }
            else:
                retrieval_obj["general_offline_rerank"] = {"enabled": False}

            # 2.75) Topology-triggered rescue. Area controls motifs/budgets only.
            # Translational properties are deliberately applied after rescue so every
            # candidate, including sparse/family fallback candidates, is calibrated once.
            area_name_for_rescue = normalize_area_name(case.get("area") or "")
            requested_sparse_areas = {normalize_area_name(x) for x in (sparse_rescue_areas or []) if str(x).strip()}
            before_sparse = retrieval_obj.get("candidate_drugs", [])
            candidate_floor_triggered = len(before_sparse) < int(GLOBAL_CANDIDATE_FLOOR)
            area_sparse_requested = area_name_for_rescue in requested_sparse_areas
            if enable_sparse_area_rescue and (area_sparse_requested or candidate_floor_triggered):
                retrieval_obj["candidate_drugs_before_sparse_area_rescue"] = before_sparse
                try:
                    after_sparse, sparse_report = apply_area_sparse_tail_rescue(
                        driver=driver,
                        disease_name=disease,
                        area=area_name_for_rescue,
                        candidates=before_sparse,
                        protect_top_n=sparse_rescue_protect_top_n,
                        top_k=100,
                        timeout_sec=25,
                        anchor_genes=_sparse_extract_anchor_genes_from_retrieval(retrieval_obj),
                        force_candidate_floor=candidate_floor_triggered,
                    )
                    retrieval_obj["candidate_drugs"] = after_sparse
                    retrieval_obj["candidate_drugs_sparse_area_rescue_final"] = after_sparse
                    retrieval_obj["sparse_area_rescue"] = sparse_report
                except Exception as e:
                    retrieval_obj["sparse_area_rescue"] = {"enabled": True, "status": "failed", "area": area_name_for_rescue, "error": str(e)}
                    retrieval_obj["candidate_drugs"] = before_sparse
            else:
                retrieval_obj["sparse_area_rescue"] = {
                    "enabled": bool(enable_sparse_area_rescue),
                    "area": area_name_for_rescue,
                    "status": "disabled_or_not_triggered",
                    "candidate_floor": int(GLOBAL_CANDIDATE_FLOOR),
                    "candidate_count_before_rescue": len(before_sparse),
                }

            # 2.76) Stable deduplication only; no area-specific score.
            retrieval_obj["candidate_drugs_before_area_weighted_final"] = copy.deepcopy(
                retrieval_obj.get("candidate_drugs", [])
            )
            final_rows, final_report = apply_graphdrx_area_final_ranking(
                retrieval_obj, area_name_for_rescue
            )
            retrieval_obj["candidate_drugs_before_structural_evidence_ranking"] = copy.deepcopy(final_rows)
            structurally_ranked, structural_ranking_report = apply_structural_evidence_ranking(
                final_rows,
                prior_threshold=structural_reserve_prior_threshold,
            )
            retrieval_obj["candidate_drugs_after_structural_evidence_ranking"] = copy.deepcopy(structurally_ranked)
            retrieval_obj["structural_evidence_ranking"] = structural_ranking_report
            retrieval_final = annotate_candidate_scientific_diagnostics(structurally_ranked)

            # Direct target-disease contraindications are safety counter-evidence.
            # They are annotated after retrieval and never remove or reorder the
            # Eval1 retrieval list.
            contraindication_map: Dict[str, List[Dict[str, Any]]] = {}
            contraindication_annotation_report: Dict[str, Any] = {
                "enabled": True,
                "status": "ok",
                "n_candidates": len(retrieval_final),
                "n_direct_contraindications": 0,
            }
            try:
                contraindication_map = get_target_disease_contraindication_map(
                    driver=driver,
                    disease=disease,
                    candidate_names=[c.get("drug") for c in retrieval_final],
                    disease_element_id=case.get("_disease_element_id"),
                    disease_element_ids=case.get("_disease_exact_element_ids") or [],
                    timeout_sec=30,
                )
            except Exception as exc:
                contraindication_annotation_report = {
                    "enabled": True,
                    "status": "failed_annotation_not_used_for_retrieval",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "n_candidates": len(retrieval_final),
                    "n_direct_contraindications": 0,
                }
                contraindication_map = {}

            retrieval_final = annotate_target_disease_contraindications(
                retrieval_final,
                contraindication_map,
                translational_partition=False,
            )
            contraindication_annotation_report["n_direct_contraindications"] = sum(
                1 for c in retrieval_final if c.get("target_disease_contraindication")
            )
            contraindication_annotation_report["drugs"] = [
                c.get("drug") for c in retrieval_final if c.get("target_disease_contraindication")
            ]

            retrieval_obj["candidate_drugs_area_weighted_final"] = copy.deepcopy(retrieval_final)
            retrieval_obj["area_weighted_final"] = final_report
            retrieval_obj["target_disease_contraindication_annotation"] = contraindication_annotation_report

            # 2.77) Maintain two explicit rank views.
            # Eval1 and candidate_drugs use retrieval_final. Translational
            # calibration operates on an independent copy and can partition
            # weak sparse evidence or known contraindications into a tail.
            retrieval_obj["candidate_drugs_before_translational_calibration"] = copy.deepcopy(retrieval_final)
            retrieval_obj["candidate_drugs_before_area_property_calibration"] = copy.deepcopy(retrieval_final)
            translational_seed = annotate_target_disease_contraindications(
                retrieval_final,
                contraindication_map,
                translational_partition=True,
            )
            try:
                translational_ranked, prop_report = apply_area_property_postrank_calibration(
                    translational_seed, area=area_name_for_rescue
                )
                retrieval_obj["candidate_drugs_area_property_calibrated_final"] = copy.deepcopy(translational_ranked)
                retrieval_obj["area_property_calibration"] = prop_report
            except Exception as e:
                retrieval_obj["area_property_calibration"] = {
                    "enabled": True,
                    "status": "failed",
                    "error": str(e),
                }
                translational_ranked = copy.deepcopy(translational_seed)
                for i, row in enumerate(translational_ranked, start=1):
                    row["rank"] = i

            retrieval_obj["candidate_drugs_retrieval_final"] = copy.deepcopy(retrieval_final)
            retrieval_obj["candidate_drugs_translational_priority"] = copy.deepcopy(translational_ranked)
            retrieval_obj["candidate_drugs_for_downstream"] = copy.deepcopy(translational_ranked)
            retrieval_obj["candidate_drugs"] = copy.deepcopy(retrieval_final)
            retrieval_obj["rank_views"] = {
                "eval1": "candidate_drugs_retrieval_final",
                "downstream": "candidate_drugs_translational_priority",
                "backward_compatible_candidate_drugs": "candidate_drugs_retrieval_final",
            }
            retrieval_obj["default_downstream_rank_view"] = "translational_priority"
            retrieval_obj["eval1_rank_view"] = "retrieval_final"

            pattern_stats = retrieval_obj.get("anchor_pattern_query_stats") or {}
            retrieval_obj["candidate_flow_counts"] = {
                "raw_anchor_rows": sum(int((s or {}).get("rows") or 0) for s in pattern_stats.values()),
                "deduplicated_graph_candidates": len(retrieval_obj.get("candidate_drugs_graphrag_only", []) or []),
                "prior_verified_candidates": sum(
                    1 for c in retrieval_final if c.get("source_prior_drug_target_graph") or c.get("source_prior_graph_verified")
                ),
                "structural_core_candidates": int(
                    ((retrieval_obj.get("structural_evidence_ranking") or {}).get("group_counts") or {}).get("core", 0)
                ),
                "structural_reserve_candidates": int(
                    ((retrieval_obj.get("structural_evidence_ranking") or {}).get("group_counts") or {}).get("reserve", 0)
                ),
                "before_sparse_rescue": len(retrieval_obj.get("candidate_drugs_before_sparse_area_rescue", []) or []),
                "after_sparse_rescue": len(retrieval_obj.get("candidate_drugs_sparse_area_rescue_final", retrieval_final) or []),
                "retrieval_final": len(retrieval_final),
                "translational_priority": len(translational_ranked),
            }
            retrieval_obj["abstention_reason"] = infer_abstention_reason(
                retrieval_obj, retrieval_final
            )

            print("\n[RETRIEVAL FINAL CANDIDATES -- EVAL1 VIEW]")
            for c in retrieval_obj.get("candidate_drugs_retrieval_final", [])[:100]:
                print(
                    f"  #{c['rank']} {c['drug']} | "
                    f"retrieval={c.get('candidate_score')} | "
                    f"general={c.get('general_rerank_score')} | "
                    f"graph={c.get('base_graph_score')} | "
                    f"disease_emb={c.get('embedding_prior_score')} | "
                    f"structural={c.get('structural_evidence_group')}:{c.get('structural_evidence_signature')} | "
                    f"drug_emb={c.get('drug_embedding_similarity')} | "
                    f"direction={c.get('direction_confidence')} | "
                    f"contra={c.get('target_disease_contraindication')} | "
                    f"partition={c.get('rank_partition')}"
                )

            print("\n[TRANSLATIONAL PRIORITY CANDIDATES -- DOWNSTREAM VIEW]")
            for c in retrieval_obj.get("candidate_drugs_translational_priority", [])[:20]:
                print(
                    f"  #{c['rank']} {c['drug']} | "
                    f"translational={c.get('candidate_score')} | "
                    f"adjustment={c.get('translational_adjustment')} | "
                    f"risk={c.get('translational_risk_tier')} | "
                    f"partition={c.get('rank_partition')}"
                )

            # 2.8) Stage-wise evaluation candidate sets
            stage_candidate_sets = {
                "graph_only": retrieval_obj.get("candidate_drugs_graphrag_only", []),
                "disease_embedding": retrieval_obj.get("candidate_drugs_with_embedding_prior_base", []),
                "drug_embedding": retrieval_obj.get(
                    "candidate_drugs_with_drug_embedding",
                    retrieval_obj.get("candidate_drugs_with_embedding_prior", []),
                ),
                "general_rerank": retrieval_obj.get(
                    "candidate_drugs_before_sparse_area_rescue",
                    retrieval_obj.get("candidate_drugs_before_translational_calibration", retrieval_obj.get("candidate_drugs", [])),
                ),
                "sparse_rescue": retrieval_obj.get(
                    "candidate_drugs_retrieval_final",
                    retrieval_obj.get("candidate_drugs_before_translational_calibration", retrieval_obj.get("candidate_drugs", [])),
                ),
                "translational_calibrated": retrieval_obj.get(
                    "candidate_drugs_translational_priority",
                    retrieval_obj.get("candidate_drugs_area_property_calibrated_final", []),
                ),
                "translational_priority": retrieval_obj.get(
                    "candidate_drugs_translational_priority", []
                ),
                "final": retrieval_obj.get(
                    "candidate_drugs_retrieval_final", retrieval_obj.get("candidate_drugs", [])
                ),
            }
            
            # Retrieval is complete. Evaluation labels are constructed only now
            # and are never passed back into candidate generation or scoring.
            manual_gold = []
            gold_drugs, gold_records = build_gold_drugs(driver=driver, case=case)

            expanded_case = dict(case)
            expanded_case["positive_rels"] = OFFLABEL_GOLD_RELS
            expanded_gold_drugs, expanded_gold_records = build_gold_drugs(
                driver=driver, case=expanded_case
            )

            supportive_case = dict(case)
            supportive_case["positive_rels"] = SUPPORTIVE_POSITIVE_RELS_DISCOVERY
            supportive_gold_drugs, supportive_gold_records = build_gold_drugs(
                driver=driver, case=supportive_case
            )

            contraindication_case = dict(case)
            contraindication_case["positive_rels"] = NEGATIVE_DD_RELS
            contraindication_gold_drugs, contraindication_gold_records = build_gold_drugs(
                driver=driver, case=contraindication_case
            )

            stage_metrics = {}
            for stage_name, candidates in stage_candidate_sets.items():
                tmp_obj = {"candidate_drugs": candidates}
                stage_metrics[stage_name] = evaluate_case(
                    disease=disease,
                    gold_drugs=gold_drugs,
                    mechanism_obj=tmp_obj
                )

            # ------------------------------------------------------------
            # Evaluation gold split
            #   strict: manual_positive + indication
            #   expanded/offlabel: manual_positive + indication + off_label_use
            #   supportive: optional supplementary only, includes tested/studied
            # ------------------------------------------------------------
            strict_gold_drugs = sorted({
                str(r.get("drug") or "").strip()
                for r in gold_records
                if str(r.get("drug") or "").strip()
                and (
                    r.get("source") == "manual_curated"
                    or r.get("relation") == "indication"
                )
            })
            
            # Do NOT fallback strict indication gold to expanded/off-label gold.
            # A zero-gold disease remains in case-level outputs but is excluded from
            # aggregate strict-metric denominators. Expanded metrics are separate.
            strict_gold_is_fallback = False
            
            metrics_strict = evaluate_case(
                disease=disease,
                gold_drugs=strict_gold_drugs,
                mechanism_obj=retrieval_obj
            )
            
            metrics_expanded = evaluate_case(
                disease=disease,
                gold_drugs=expanded_gold_drugs,
                mechanism_obj=retrieval_obj
            )
 
            metrics_supportive = evaluate_case(
                disease=disease,
                gold_drugs=supportive_gold_drugs,
                mechanism_obj=retrieval_obj
            )

            metrics_contraindication = evaluate_case(
                disease=disease,
                gold_drugs=contraindication_gold_drugs,
                mechanism_obj=retrieval_obj
            )
           
            metrics = {}
            
            for key, value in metrics_strict.items():
                metrics[f"strict_{key}"] = value
            
            for key, value in metrics_expanded.items():
                metrics[f"expanded_{key}"] = value

            for key, value in metrics_supportive.items():
                metrics[f"supportive_{key}"] = value

            for key, value in metrics_contraindication.items():
                metrics[f"contraindication_{key}"] = value


            # Keep old unprefixed metric names as indication-only metrics for TxGNN-aligned evaluation.
            metrics.update(metrics_strict)
            
            metrics["n_strict_gold"] = len(strict_gold_drugs)
            metrics["n_expanded_gold"] = len(expanded_gold_drugs)
            metrics["strict_gold_is_fallback"] = strict_gold_is_fallback
            metrics["strict_gold_drugs"] = strict_gold_drugs
            metrics["expanded_gold_drugs"] = expanded_gold_drugs
            metrics["n_supportive_gold"] = len(supportive_gold_drugs)
            metrics["supportive_gold_drugs"] = supportive_gold_drugs
            metrics["n_contraindication_gold"] = len(contraindication_gold_drugs)
            metrics["contraindication_gold_drugs"] = contraindication_gold_drugs
            
            metrics["case_id"] = case.get("case_id")
            metrics["area"] = case.get("area")

            runtime_sec = round(time.time() - t0, 2)

            result = {
                "disease": disease,
                "area": case.get("area"),
                "manual_gold_drugs": manual_gold,
                "gold_drugs": gold_drugs,
                "gold_records": gold_records,
                "expanded_gold_records": expanded_gold_records,
                "supportive_gold_records": supportive_gold_records,
                "contraindication_gold_drugs": contraindication_gold_drugs,
                "contraindication_gold_records": contraindication_gold_records,
                "retrieval": retrieval_obj,
                "metrics": metrics,
                "stage_metrics": stage_metrics,
                "runtime_sec": runtime_sec,
                "general_offline_rerank": retrieval_obj.get("general_offline_rerank", {}),
                "structural_evidence_ranking": retrieval_obj.get("structural_evidence_ranking", {}),
            }

            all_results.append(result)
            case_metrics.append(metrics)

            print("\n[CASE METRICS]")
            print(json.dumps(metrics, ensure_ascii=False, indent=2))
            print(f"runtime_sec={runtime_sec}")


            # ------------------------------------------------------------
            # Incremental save (폴더 통합 및 파일 분리 최적화)
            # ------------------------------------------------------------
            
            area_str = str(case.get("area") or "").strip()
            area_tag = re.sub(r"[^A-Za-z0-9]+", "-", area_str.lower()).strip("-")
            
            folder_suffix = output_tag_suffix
            
            if area_tag:
                parts = folder_suffix.strip("_").split("_")
                parts = [p for p in parts if p != area_tag]
                folder_suffix = "_" + "_".join(parts) if parts else ""
            
            run_tag = f"graphdrx_top{CANDIDATE_K}_{ablation_mode}{folder_suffix}"
            out_dir = OUT_BASE_DIR / run_tag
            _graphdrx_forced_output_dir = os.getenv("GRAPHDRX_FORCE_OUTPUT_DIR")
            if _graphdrx_forced_output_dir:
                out_dir = Path(_graphdrx_forced_output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            
            file_tag = f"graphdrx_{ablation_mode}{output_tag_suffix}"
            
            # ------------------------------------------------------------
            # 아래 저장 로직은 기존 코드와 동일합니다.
            # ------------------------------------------------------------
            all_results_path = out_dir / f"{file_tag}_all_results.json"
            results_jsonl_path = out_dir / f"{file_tag}_results.jsonl"
            rag_context_path = out_dir / f"{file_tag}_rag_context.json"
            compact_candidates_path = out_dir / f"{file_tag}_top{OUTPUT_COMPACT_TOP_K}_candidates.csv"
            summary_path = out_dir / f"{file_tag}_summary.csv"
            metrics_path = out_dir / f"{file_tag}_metrics.json"
            
            # Durable incremental persistence: append exactly one complete disease
            # record instead of rewriting a multi-GB JSON file after every case.
            append_jsonl_record(results_jsonl_path, result)
            atomic_write_json(rag_context_path, rag_context_by_disease, indent=None)
            write_compact_top_candidates_csv(
                all_results,
                compact_candidates_path,
                top_k=OUTPUT_COMPACT_TOP_K,
            )
            
            write_summary_csv(case_metrics, summary_path)

            # Also save relation-specific per-case summaries in the same run.
            # Default summary_path remains backward-compatible expanded/offlabel metrics.
            write_summary_csv(
                extract_prefixed_metric_rows(case_metrics, "strict"),
                out_dir / f"{file_tag}_summary_indication_only.csv",
            )
            write_summary_csv(
                extract_prefixed_metric_rows(case_metrics, "expanded"),
                out_dir / f"{file_tag}_summary_indication_offlabel.csv",
            )
            write_summary_csv(
                extract_prefixed_metric_rows(case_metrics, "supportive"),
                out_dir / f"{file_tag}_summary_supportive.csv",
            )
            write_summary_csv(
                extract_prefixed_metric_rows(case_metrics, "contraindication"),
                out_dir / f"{file_tag}_summary_contraindication.csv",
            )

            # Stage-wise and pattern-wise diagnostics for strict indication-only gold.
            # These are cheap post-hoc summaries from the same all_results payload.
            stage_rows = flatten_stage_metric_rows(all_results, key="stage_metrics")
            stage_aggregate = aggregate_stage_metrics(all_results, key="stage_metrics")
            write_rows_csv(
                stage_rows,
                out_dir / f"{file_tag}_stage_disease_metrics_indication_only.csv",
                preferred=["stage", "case_id", "area", "disease", "runtime_sec"],
            )
            write_rows_csv(
                aggregate_stage_metrics_by_area(all_results, key="stage_metrics"),
                out_dir / f"{file_tag}_stage_area_metrics_indication_only.csv",
                preferred=["stage", "area", "n_cases", "total_gold_labels", "retrieval_hits_at_10_rate", "retrieval_hits_at_100_rate", "retrieval_mean_gold_recall_at_100", "retrieval_micro_gold_recall_at_100", "retrieval_mean_mrr"],
            )

            pattern_rows = flatten_pattern_metric_rows(all_results)
            pattern_aggregate = aggregate_pattern_metric_rows(pattern_rows)
            write_rows_csv(
                pattern_rows,
                out_dir / f"{file_tag}_pattern_disease_metrics_indication_only.csv",
                preferred=["pattern", "case_id", "area", "disease", "runtime_sec", "n_pattern_candidates", "n_best_path_candidates"],
            )
            write_rows_csv(
                aggregate_pattern_metrics_by_area(pattern_rows),
                out_dir / f"{file_tag}_pattern_area_metrics_indication_only.csv",
                preferred=["pattern", "area", "n_cases", "total_gold_labels", "total_pattern_candidates", "total_best_path_candidates", "retrieval_hits_at_10_rate", "retrieval_hits_at_100_rate", "retrieval_mean_gold_recall_at_100", "retrieval_micro_gold_recall_at_100", "retrieval_mean_mrr"],
            )

        # ------------------------------------------------------------
        # Aggregate metrics
        # case_metrics contains flat metric rows.
        # all_results contains nested result objects, so do NOT aggregate all_results directly.
        # ------------------------------------------------------------
        aggregate = aggregate_metrics(case_metrics)
        
        strict_metric_rows = []
        expanded_metric_rows = []
        supportive_metric_rows = []
        contraindication_metric_rows = []

        for r in case_metrics:
            strict_row = {}
            expanded_row = {}
            supportive_row = {}
            contraindication_row = {}

            for key, value in r.items():
                if key.startswith("strict_"):
                    strict_row[key.replace("strict_", "", 1)] = value
                elif key.startswith("expanded_"):
                    expanded_row[key.replace("expanded_", "", 1)] = value
                elif key.startswith("supportive_"):
                    supportive_row[key.replace("supportive_", "", 1)] = value
                elif key.startswith("contraindication_"):
                    contraindication_row[key.replace("contraindication_", "", 1)] = value

            if strict_row:
                strict_metric_rows.append(strict_row)
            if expanded_row:
                expanded_metric_rows.append(expanded_row)
            if supportive_row:
                supportive_metric_rows.append(supportive_row)
            if contraindication_row:
                contraindication_metric_rows.append(contraindication_row)

        aggregate["strict_metrics"] = aggregate_metrics(strict_metric_rows)
        aggregate["expanded_metrics"] = aggregate_metrics(expanded_metric_rows)
        aggregate["supportive_metrics"] = aggregate_metrics(supportive_metric_rows)
        aggregate["contraindication_metrics"] = aggregate_metrics(contraindication_metric_rows)
        aggregate["structural_evidence_ranking_enabled"] = True
        aggregate["structural_evidence_ranking_mode"] = "single_indirect_only_reserve"
        aggregate["ablation_mode"] = ablation_mode
        aggregate["pipeline"] = "graphdrx_graphrag"
        aggregate["candidate_k"] = CANDIDATE_K
        aggregate["use_drug_embedding_rerank"] = use_drug_embedding_rerank
        aggregate["drug_embedding_weight"] = drug_embedding_weight
        aggregate["drug_embedding_max_candidates"] = DRUG_EMBEDDING_MAX_CANDIDATES
        drug_embedding_statuses = []
        for r in all_results:
            retrieval_for_status = r.get("retrieval") or {}
            status = retrieval_for_status.get("drug_embedding_rerank")
            if isinstance(status, dict):
                drug_embedding_statuses.append(status)
        aggregate["drug_embedding_effectively_active"] = any(
            bool(s.get("effectively_active")) for s in drug_embedding_statuses
        )
        aggregate["drug_embedding_success_total"] = sum(int(s.get("success_count") or 0) for s in drug_embedding_statuses)
        aggregate["drug_embedding_error_total"] = sum(int(s.get("error_count") or 0) for s in drug_embedding_statuses)
        aggregate["drug_embedding_nonzero_similarity_total"] = sum(int(s.get("nonzero_similarity_count") or 0) for s in drug_embedding_statuses)
        aggregate["drug_embedding_status_counts"] = dict(Counter(str(s.get("status")) for s in drug_embedding_statuses))
        aggregate["use_general_offline_rerank"] = use_general_offline_rerank
        aggregate["offline_rerank_drug_weight"] = offline_rerank_drug_weight
        aggregate["output_tag_suffix"] = output_tag_suffix
        aggregate["use_disease_embedding_prior"] = use_disease_embedding_prior
        aggregate["use_semantic_prior_branch"] = use_semantic_prior_branch
        aggregate["use_common_direct_graph"] = use_common_direct_graph
        aggregate["use_mechanism_lookup"] = use_mechanism_lookup
        aggregate["hardcoded_answer_rules"] = {
            "uses_disease_name_rules": False,
            "uses_drug_name_rules": False,
            "uses_gene_allowlists": False,
            "uses_gold_or_outcome_rules": False,
        }
        aggregate["area_property_calibration_enabled"] = bool(ENABLE_TRANSLATIONAL_CALIBRATION)
        aggregate["translational_calibration_enabled"] = bool(ENABLE_TRANSLATIONAL_CALIBRATION)
        aggregate["stage_metrics"] = aggregate_stage_metrics(all_results, key="stage_metrics")
        _pattern_rows_final = flatten_pattern_metric_rows(all_results)
        aggregate["pattern_metrics"] = aggregate_pattern_metric_rows(_pattern_rows_final)
        aggregate["disabled_anchor_patterns"] = sorted(DISABLED_ANCHOR_PATTERNS)
        aggregate["method_note"] = (
            "GraphDRx retrieval with target-context, vector-neighbor, semantic-prior, and direct gene-action channels, one-pass evidence integration, "
            "topology-derived area policy, sparse rescue, and one final global translational calibration. "
            "See run_manifest.json and graphdrx_method_config_used.json for exact ablation flags."
        )

        metrics_path = out_dir / f"{file_tag}_metrics.json"
        atomic_write_json(metrics_path, aggregate, indent=2)

        final_payload = {
            "results": all_results,
            "rag_context_by_disease": rag_context_by_disease,
            "aggregate_metrics": aggregate,
        }
        if WRITE_FULL_ALL_RESULTS_JSON:
            # Written once, compactly, and atomically. The JSONL and Top-K CSV remain
            # complete recovery artifacts even if this optional large export fails.
            atomic_write_json(all_results_path, final_payload, indent=None)

        print("\n" + "=" * 120)
        print("[FINAL AGGREGATE METRICS]")
        print("=" * 120)
        print(json.dumps(aggregate, ensure_ascii=False, indent=2))

        print("\nSaved:")
        if WRITE_FULL_ALL_RESULTS_JSON:
            print(f"- {all_results_path}")
        print(f"- {results_jsonl_path}")
        print(f"- {compact_candidates_path}")
        print(f"- {summary_path}")
        print(f"- {out_dir / f'{file_tag}_summary_indication_only.csv'}")
        print(f"- {out_dir / f'{file_tag}_summary_indication_offlabel.csv'}")
        print(f"- {out_dir / f'{file_tag}_summary_supportive.csv'}")
        print(f"- {out_dir / f'{file_tag}_stage_disease_metrics_indication_only.csv'}")
        print(f"- {out_dir / f'{file_tag}_stage_area_metrics_indication_only.csv'}")
        print(f"- {out_dir / f'{file_tag}_pattern_disease_metrics_indication_only.csv'}")
        print(f"- {out_dir / f'{file_tag}_pattern_area_metrics_indication_only.csv'}")
        print(f"- {metrics_path}")

    finally:
        driver.close()

