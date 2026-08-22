# ============================================================
# Disease Context Embedding + Semantic Therapeutic Prior
#
# Purpose:
# 1. Build disease-only safe context without treatment leakage
# 2. Summarize it into compact embedding text
# 3. Store embedding on DISEASE nodes in Neo4j
# 4. Use vector-similar diseases to derive semantic therapeutic priors
# 5. Return embedding_prior_score for GraphRAG reranking
#
# Important:
# - Does NOT use orphanet_management_and_treatment
# - Does NOT use direct drug-disease edges for the queried disease
# - Does NOT use drug free-text MoA
# - side_effect is NOT a disease relation; it belongs to DRUG-PHENO risk module
# ============================================================

import os
import re
import json
import math
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from neo4j import GraphDatabase, Query


# ============================================================
# 0. Config
# ============================================================

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

# embedding model examples:
# - nomic-embed-text
# - mxbai-embed-large
# - bge-m3 if installed in Ollama
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

# summarization model
SUMMARY_MODEL = os.getenv("GRAPHDRX_SUMMARY_MODEL", "gemma3:12b")

DISEASE_VECTOR_INDEX_NAME = "disease_safe_context_embedding_idx"

# Direct drug-disease relations.
# These are used for gold construction / masking,
# NOT side_effect.
DISCOVERY_POSITIVE_DD_RELS = [
    "indication",
    "off_label_use",
    "tested_indication",
    "studied_for_treatment_of",
]


# This is DRUG-PHENO, not direct disease relation.
DRUG_PHENOTYPE_RISK_RELS = [
    "side_effect",
]

# Disease text fields safe for discovery input.
# These describe disease biology / phenotype / causes,
# but should not directly say "treat with drug X".
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

# Context only. Useful but weak mechanistic signal.
DISEASE_TEXT_FIELDS_CONTEXT_ONLY = [
    "orphanet_prevalence",
    "orphanet_epidemiology",
]

# Excluded to avoid treatment / clinical workflow leakage.
DISEASE_TEXT_FIELDS_EXCLUDE = [
    "orphanet_management_and_treatment",
    "mayo_prevention",
    "mayo_see_doc",
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
    "nucleus",
    "nucleoplasm",
    "metal ion binding",
    "protein kinase binding",
    "kinase binding",
    "protein serine/threonine kinase activity",
    "intracellular signal transduction",
    "cell-cell signaling",
    "g protein-coupled receptor signaling pathway",
    "regulation of transcription by rna polymerase ii",
    "positive regulation of transcription by rna polymerase ii",
    "negative regulation of transcription by rna polymerase ii",
    "positive regulation of gene expression",
    "negative regulation of gene expression",
    "positive regulation of protein phosphorylation",
    "negative regulation of protein phosphorylation",
    "positive regulation of kinase activity",
    "negative regulation of kinase activity",
    "positive regulation of cell population proliferation",
    "negative regulation of cell population proliferation",
    "regulation of cell cycle",
    "adaptive immune response",
    "innate immune response",
    "cytokine-mediated signaling pathway",
    "intracellular protein transport",
    "signaling",
    "regulation of signaling",
    "cell communication",
    "response to stimulus",
}


# ============================================================
# 1. Basic helpers
# ============================================================

def make_driver(
    neo4j_uri: str = NEO4J_URI,
    neo4j_user: str = NEO4J_USER,
    neo4j_password: Optional[str] = None,
):
    password = neo4j_password or NEO4J_PASSWORD
    if not password:
        raise RuntimeError(
            "Neo4j password is empty. Set NEO4J_PASSWORD or pass neo4j_password."
        )
    return GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, password))


def run_query(
    driver,
    query: str,
    params: Optional[Dict[str, Any]] = None,
    timeout_sec: int = 60,
) -> List[Dict[str, Any]]:
    with driver.session() as session:
        result = session.run(Query(query, timeout=timeout_sec), params or {})
        return [record.data() for record in result]


def normalize_name(x: Any) -> str:
    return str(x or "").strip().lower()


def clean_nan_text(x: Any) -> str:
    s = str(x or "").strip()
    if not s or s.lower() in {"nan", "none", "null", "na"}:
        return ""
    return s


def unique_keep_order(items: List[Any]) -> List[str]:
    out = []
    seen = set()
    for x in items or []:
        s = str(x or "").strip()
        if not s:
            continue
        key = normalize_name(s)
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def truncate_text(text: str, max_chars: int) -> str:
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def mask_terms_in_text(text: str, terms: List[str], mask_token: str = "[MASKED]") -> str:
    masked = str(text or "")
    for term in sorted(terms or [], key=len, reverse=True):
        term = str(term or "").strip()
        if not term:
            continue
        masked = re.sub(
            re.escape(term),
            mask_token,
            masked,
            flags=re.IGNORECASE,
        )
    return masked


def flatten_values(xs):
    if xs is None:
        return []
    if isinstance(xs, (list, tuple, set)):
        out = []
        for x in xs:
            out.extend(flatten_values(x))
        return out
    return [xs]


def max_numeric_value(x: Any, default: float = 0.0) -> float:
    vals = []
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


def minmax_clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))
# ============================================================
# 2. Ollama helpers
# ============================================================

def ollama_chat(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    timeout_sec: int = 240,
) -> str:
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama chat request failed: {e}")

    return obj.get("message", {}).get("content", "")


def ollama_embed(
    text: str,
    model: str = EMBED_MODEL,
    timeout_sec: int = 180,
) -> List[float]:
    """
    Ollama embedding endpoint.
    Works with embedding models such as nomic-embed-text or mxbai-embed-large.
    """
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/embeddings"
    payload = {
        "model": model,
        "prompt": str(text or ""),
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama embedding request failed: {e}")

    emb = obj.get("embedding")
    if not emb:
        raise RuntimeError(f"No embedding returned by Ollama model={model}")

    return [float(x) for x in emb]


# ============================================================
# 3. Disease RAG row retrieval
# ============================================================

def norm_text(x: Any) -> str:
    return str(x or "").strip().lower()


def load_disease_features(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find disease feature file: {path}")
    return pd.read_csv(path)

def get_disease_rag_row(
    disease: str,
    disease_df: pd.DataFrame,
) -> Optional[Dict[str, Any]]:
    """Find a disease-text row by exact full-name equality only."""
    q = norm_text(disease)
    df = disease_df.copy()
    for col in ["mondo_name", "group_name_bert"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    if "mondo_name" in df.columns:
        hit = df[df["mondo_name"].map(norm_text) == q]
        if len(hit) > 0:
            return pick_best_disease_row(hit)

    if "group_name_bert" in df.columns:
        hit = df[df["group_name_bert"].map(norm_text) == q]
        if len(hit) > 0:
            return hit.iloc[0].to_dict()

    return None


# ============================================================
# 4. KG disease neighborhood
# ============================================================

def get_kg_disease_neighborhood(
    driver,
    disease: str,
    disease_element_id: Optional[str] = None,
    disease_element_ids: Optional[List[str]] = None,
    max_genes: int = 100,
    max_phenotypes: int = 80,
    max_family: int = 50,
) -> Dict[str, Any]:
    """Return the unioned disease-only neighborhood for an exact-name bundle.

    Direct drug-disease relations are never queried. Parent/child edges between
    bundle members are excluded so duplicate records are not treated as family
    evidence.
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

    OPTIONAL MATCH (d)-[:phenotype_present|phenotype_absent]-(ph:PHENO)
    WITH d, disease_genes, collect(DISTINCT ph.name) AS phenotypes

    OPTIONAL MATCH (d)-[:parent_child]-(fam:DISEASE)
    WHERE NOT elementId(fam) IN $disease_element_ids
      AND toLower(trim(coalesce(fam.name, ''))) <> toLower(trim($disease))
    RETURN
      d.name AS disease_name,
      elementId(d) AS element_id,
      disease_genes,
      phenotypes,
      collect(DISTINCT fam.name) AS family_names
    ORDER BY elementId(d)
    """

    rows = run_query(
        driver,
        q,
        {
            "disease": disease,
            "disease_element_ids": exact_ids,
        },
        timeout_sec=60,
    )
    if not rows:
        raise ValueError(
            f"Exact DISEASE node not found while building embedding context: {disease!r}"
        )

    return {
        "query_disease": disease,
        "matched_disease_names": unique_keep_order([
            r.get("disease_name") or disease for r in rows
        ]),
        "bundle_element_ids": unique_keep_order([
            r.get("element_id") for r in rows if r.get("element_id")
        ]),
        "disease_genes": unique_keep_order([
            x for r in rows for x in (r.get("disease_genes") or []) if x
        ])[:max_genes],
        "phenotypes": unique_keep_order([
            x for r in rows for x in (r.get("phenotypes") or []) if x
        ])[:max_phenotypes],
        "family_names": unique_keep_order([
            x for r in rows for x in (r.get("family_names") or []) if x
        ])[:max_family],
    }


# ============================================================
# 5. Safe disease context builder
# ============================================================

def build_safe_disease_context(
    disease: str,
    disease_row: Optional[Dict[str, Any]],
    kg_context: Dict[str, Any],
    mask_terms: Optional[List[str]] = None,
    max_chars_per_field: int = 1400,
) -> Dict[str, Any]:
    """
    Build treatment-restricted disease context for disease representation and embedding.

    mask_terms:
      - use heldout/gold drug names only if you are worried they appear in disease text
      - normally disease text should not contain drug names, but masking is safe
    """
    mask_terms = mask_terms or []

    safe_text = {}
    context_only = {}

    if disease_row:
        for field in DISEASE_TEXT_FIELDS_SAFE:
            raw = clean_nan_text(disease_row.get(field))
            if not raw:
                continue
            raw = mask_terms_in_text(raw, mask_terms, mask_token="[MASKED_DRUG]")
            safe_text[field] = truncate_text(raw, max_chars_per_field)

        for field in DISEASE_TEXT_FIELDS_CONTEXT_ONLY:
            raw = clean_nan_text(disease_row.get(field))
            if not raw:
                continue
            raw = mask_terms_in_text(raw, mask_terms, mask_token="[MASKED_DRUG]")
            context_only[field] = truncate_text(raw, 600)

    canonical_names = {
        "mondo_name": clean_nan_text(disease_row.get("mondo_name")) if disease_row else "",
        "group_name_bert": clean_nan_text(disease_row.get("group_name_bert")) if disease_row else "",
        "matched_disease_names": kg_context.get("matched_disease_names") or [disease],
    }

    return {
        "query_disease": disease,
        "canonical_names": canonical_names,
        "safe_disease_text": safe_text,
        "context_only": context_only,
        "kg_neighborhood": {
            "disease_genes": kg_context.get("disease_genes", []),
            "phenotypes": kg_context.get("phenotypes", []),
            "disease_family_names": kg_context.get("family_names", []),
        },
        "excluded_fields_note": [
            "orphanet_management_and_treatment",
            "mayo_prevention",
            "mayo_see_doc",
            "direct drug-disease relations",
            "drug indication text",
            "drug free-text mechanism of action",
            "drug pharmacodynamics free text",
        ],
    }


def flatten_context_for_summary(context: Dict[str, Any]) -> str:
    """
    Convert safe context dict to deterministic text before summarization.
    """
    parts = []

    parts.append(f"Disease: {context.get('query_disease', '')}")

    names = context.get("canonical_names") or {}
    if names.get("mondo_name"):
        parts.append(f"MONDO canonical name: {names.get('mondo_name')}")
    if names.get("group_name_bert"):
        parts.append(f"Disease group name: {names.get('group_name_bert')}")

    safe_text = context.get("safe_disease_text") or {}
    for field in DISEASE_TEXT_FIELDS_SAFE:
        if field in safe_text:
            parts.append(f"\n[{field}]\n{safe_text[field]}")

    context_only = context.get("context_only") or {}
    for field in DISEASE_TEXT_FIELDS_CONTEXT_ONLY:
        if field in context_only:
            parts.append(f"\n[{field} - context only]\n{context_only[field]}")

    kg = context.get("kg_neighborhood") or {}

    genes = kg.get("disease_genes") or []
    if genes:
        parts.append("\n[KG disease-associated genes]\n" + ", ".join(genes[:100]))

    phenotypes = kg.get("phenotypes") or []
    if phenotypes:
        parts.append("\n[KG disease phenotypes]\n" + ", ".join(phenotypes[:80]))

    family = kg.get("disease_family_names") or []
    if family:
        parts.append("\n[KG disease family/subtype names]\n" + ", ".join(family[:50]))

    parts.append(
        "\n[Leakage control]\n"
        "Treatment, management, prevention, direct drug-disease edges, drug indication text, "
        "and drug mechanism-of-action free text were excluded."
    )

    return "\n".join(parts).strip()


# ============================================================
# 6. Summary for embedding
# ============================================================

DISEASE_SUMMARY_SYSTEM_PROMPT = """
You are preparing disease-only context for a drug repurposing discovery benchmark.

Rules:
1. Do NOT mention any drug names.
2. Do NOT infer treatments.
3. Do NOT include treatment, management, prevention, clinical guideline, indication, or drug mechanism-of-action text.
4. Summarize only disease biology, symptoms, causes, affected systems, phenotypes, genes, pathways, and disease family context.
5. Prefer mechanistically useful anchors but avoid recommending drugs.
6. Use concise biomedical English.
7. Output plain text only.

Required structure:
Disease overview:
Key pathophysiology:
Clinical/phenotypic manifestations:
Genetic or molecular anchors:
Disease-family or subtype context:
Useful KG retrieval anchors:
"""

def summarize_disease_context_for_embedding(
    context: Dict[str, Any],
    summary_model: str = SUMMARY_MODEL,
    max_input_chars: int = 9000,
    fallback_to_raw: bool = True,
) -> str:
    raw_context = flatten_context_for_summary(context)
    raw_context = truncate_text(raw_context, max_input_chars)

    user_prompt = f"""
Summarize the following disease-only context into a compact embedding document.

Disease-only safe context:
{raw_context}

Remember:
- Do not mention drug names.
- Do not include treatment or management.
- Do not recommend therapies.
- Focus on disease biology, phenotype, genes, and KG retrieval anchors.
"""

    try:
        summary = ollama_chat(
            model=summary_model,
            system_prompt=DISEASE_SUMMARY_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.0,
            timeout_sec=240,
        )
        summary = clean_nan_text(summary)
        if summary:
            return summary
    except Exception as e:
        print(f"[WARN] disease summary failed: {e}", flush=True)

    if fallback_to_raw:
        return raw_context

    raise RuntimeError("Disease summary failed and fallback_to_raw=False")


# ============================================================
# 7. Neo4j embedding storage
# ============================================================

def resolve_embedding_dim(embed_model: str = EMBED_MODEL) -> int:
    emb = ollama_embed("dimension probe", model=embed_model)
    return len(emb)


def create_disease_vector_index(
    driver,
    embedding_dim: int,
    index_name: str = DISEASE_VECTOR_INDEX_NAME,
    property_name: str = "safe_context_embedding",
) -> None:
    """
    Create Neo4j vector index.

    Neo4j 5.x syntax.
    If your Neo4j version does not support vector indexes,
    keep embeddings as properties and do cosine scoring in Python instead.
    """
    q = f"""
    CREATE VECTOR INDEX {index_name} IF NOT EXISTS
    FOR (d:DISEASE) ON (d.{property_name})
    OPTIONS {{
      indexConfig: {{
        `vector.dimensions`: {int(embedding_dim)},
        `vector.similarity_function`: 'cosine'
      }}
    }}
    """
    run_query(driver, q, timeout_sec=60)
    
def pick_best_disease_row(hit_df: pd.DataFrame) -> dict:
    score_cols = [
        "mondo_definition",
        "umls_description",
        "orphanet_definition",
        "orphanet_clinical_description",
        "mayo_symptoms",
        "mayo_causes",
        "mayo_risk_factors",
        "mayo_complications",
    ]

    df = hit_df.copy()

    def row_score(row):
        score = 0
        for col in score_cols:
            val = clean_nan_text(row.get(col))
            if val:
                score += min(len(val), 2000)
        return score

    df["_context_score"] = df.apply(row_score, axis=1)
    df = df.sort_values("_context_score", ascending=False)
    return df.iloc[0].drop(labels=["_context_score"]).to_dict()

def upsert_disease_safe_embedding(
    driver,
    disease: str,
    safe_context: dict,
    summary_text: str,
    embedding: list,
    disease_element_id: Optional[str] = None,
    disease_element_ids: Optional[List[str]] = None,
    embed_model: str = EMBED_MODEL,
    summary_model: str = SUMMARY_MODEL,
    mask_terms: Optional[List[str]] = None,
) -> dict:
    """Store the bundle embedding on one deterministic primary member only."""
    q = """
    MATCH (d:DISEASE)
    WHERE ($disease_element_id IS NOT NULL AND elementId(d) = $disease_element_id)
       OR ($disease_element_id IS NULL
           AND toLower(trim(d.name)) = toLower(trim($disease)))
    WITH d ORDER BY elementId(d) LIMIT 1
    SET
      d.safe_context_text = $summary_text,
      d.safe_context_embedding = $embedding,
      d.safe_context_embedding_model = $embed_model,
      d.safe_context_summary_model = $summary_model,
      d.safe_context_mask_terms_count = $mask_terms_count,
      d.safe_context_mask_terms_signature = $mask_terms_signature,
      d.safe_context_bundle_signature = $bundle_signature,
      d.safe_context_bundle_size = $bundle_size,
      d.safe_context_updated_at = datetime(),
      d.safe_context_source = "disease_text_plus_kg_bundle_neighborhood_no_treatment"
    RETURN d.name AS disease_name, elementId(d) AS element_id
    """

    rows = run_query(
        driver,
        q,
        {
            "disease": disease,
            "disease_element_id": disease_element_id,
            "summary_text": summary_text,
            "embedding": embedding,
            "embed_model": embed_model,
            "summary_model": summary_model,
            "mask_terms_count": len(mask_terms or []),
            "mask_terms_signature": "||".join(sorted({str(x).strip().lower() for x in (mask_terms or []) if str(x).strip()})),
            "bundle_signature": "||".join(sorted({str(x) for x in (disease_element_ids or []) if x})),
            "bundle_size": len({str(x) for x in (disease_element_ids or []) if x}) or 1,
        },
        timeout_sec=60,
    )

    if not rows:
        raise RuntimeError(f"No exact DISEASE node found for disease={disease}")

    return {
        "disease": disease,
        "updated_nodes": rows,
        "embedding_dim": len(embedding),
    }


def build_and_store_disease_embedding(
    driver,
    disease: str,
    disease_element_id: Optional[str] = None,
    disease_element_ids: Optional[List[str]] = None,
    disease_df: Optional[pd.DataFrame] = None,
    disease_features_csv: Optional[str | Path] = None,
    mask_terms: Optional[List[str]] = None,
    embed_model: str = EMBED_MODEL,
    summary_model: str = SUMMARY_MODEL,
    create_index: bool = True,
) -> Dict[str, Any]:
    """
    End-to-end:
    disease row + KG neighborhood -> safe context -> summary -> embedding -> Neo4j.
    """
    if disease_df is None:
        if disease_features_csv is None:
            disease_row = None
        else:
            disease_df = load_disease_features(disease_features_csv)
            disease_row = get_disease_rag_row(disease, disease_df)
    else:
        disease_row = get_disease_rag_row(disease, disease_df)

    kg_context = get_kg_disease_neighborhood(
        driver,
        disease,
        disease_element_id=disease_element_id,
        disease_element_ids=disease_element_ids,
    )

    safe_context = build_safe_disease_context(
        disease=disease,
        disease_row=disease_row,
        kg_context=kg_context,
        mask_terms=mask_terms or [],
    )

    summary_text = summarize_disease_context_for_embedding(
        safe_context,
        summary_model=summary_model,
    )

    embedding = ollama_embed(summary_text, model=embed_model)

    if create_index:
        create_disease_vector_index(driver, embedding_dim=len(embedding))

    result = upsert_disease_safe_embedding(
        driver=driver,
        disease=disease,
        safe_context=safe_context,
        disease_element_id=disease_element_id,
        disease_element_ids=disease_element_ids,
        summary_text=summary_text,
        embedding=embedding,
        embed_model=embed_model,
        summary_model=summary_model,
        mask_terms=mask_terms or [],
    )

    result["safe_context"] = safe_context
    result["summary_text"] = summary_text
    result["embedding"] = embedding

    return result


# ============================================================
# 8. Similar disease retrieval using KG vector index
# ============================================================

def query_similar_diseases_by_embedding(
    driver,
    disease: str,
    embedding: Optional[List[float]] = None,
    summary_text: Optional[str] = None,
    top_k: int = 50,
    min_score: float = 0.85,
    index_name: str = DISEASE_VECTOR_INDEX_NAME,
    exclude_names: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Find vector-similar diseases, deduplicated by normalized disease name."""
    if embedding is None:
        if not summary_text:
            raise ValueError("Either embedding or summary_text must be provided.")
        embedding = ollama_embed(summary_text, model=EMBED_MODEL)

    exclude_names = unique_keep_order((exclude_names or []) + [disease])
    query_k = max(int(top_k) * 3, int(top_k))

    q = f"""
    CALL db.index.vector.queryNodes($index_name, $query_k, $embedding)
    YIELD node, score

    WHERE node:DISEASE
      AND node.name IS NOT NULL
      AND NOT toLower(trim(node.name)) IN [x IN $exclude_names | toLower(trim(x))]
      AND score >= $min_score

    WITH toLower(trim(node.name)) AS normalized_name,
         max(score) AS similarity_score,
         collect(node.name)[0] AS disease,
         collect(node.safe_context_text)[0] AS safe_context_text
    RETURN disease, similarity_score, safe_context_text
    ORDER BY similarity_score DESC
    LIMIT $top_k
    """

    return run_query(
        driver,
        q,
        {
            "index_name": index_name,
            "query_k": query_k,
            "top_k": int(top_k),
            "embedding": embedding,
            "exclude_names": exclude_names,
            "min_score": float(min_score),
        },
        timeout_sec=60,
    )


# Similar-disease filtering uses only explicit evaluation masking and an optional
# numeric similarity ceiling. Disease-name keyword families are not used.

def filter_similar_disease_priors(
    similar_diseases,
    eval_diseases=None,
    max_similarity_for_prior=None,
):
    eval_norm = {normalize_name(x) for x in (eval_diseases or [])}
    out = []
    removed = []

    for r in similar_diseases:
        sd = r.get("disease", "")
        sim = float(r.get("similarity_score") or 0)

        reason = None
        if normalize_name(sd) in eval_norm:
            reason = "in_eval_disease_list"
        elif max_similarity_for_prior is not None and sim >= max_similarity_for_prior:
            reason = f"too_high_similarity_{sim:.3f}"

        if reason:
            rr = dict(r)
            rr["mask_reason"] = reason
            removed.append(rr)
        else:
            out.append(r)

    return out, removed
    

def get_stored_disease_embedding(
    driver,
    disease: str,
    disease_element_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    # Use properties(d) to avoid Neo4j warnings when optional cache metadata
    # keys have not been created yet in a fresh KG. Directly referencing a
    # non-existing property key can emit Neo.ClientNotification noise even
    # though the query is otherwise valid.
    q = """
    MATCH (d:DISEASE)
    WHERE ($disease_element_id IS NOT NULL AND elementId(d) = $disease_element_id)
       OR ($disease_element_id IS NULL
           AND toLower(trim(d.name)) = toLower(trim($disease)))
    RETURN
      d.name AS disease,
      properties(d) AS props
    LIMIT 1
    """
    rows = run_query(
        driver,
        q,
        {"disease": disease, "disease_element_id": disease_element_id},
        timeout_sec=30,
    )
    if not rows:
        return None
    row = dict(rows[0])
    props = row.pop("props", {}) or {}
    return {
        "disease": row.get("disease"),
        "safe_context_text": props.get("safe_context_text"),
        "embedding": props.get("safe_context_embedding"),
        "embedding_model": props.get("safe_context_embedding_model"),
        "mask_terms_count": props.get("safe_context_mask_terms_count"),
        "mask_terms_signature": props.get("safe_context_mask_terms_signature"),
        "bundle_signature": props.get("safe_context_bundle_signature"),
        "bundle_size": props.get("safe_context_bundle_size"),
    }


# ============================================================
# 9. Similar-disease therapeutic prior
# ============================================================

def relation_weight(rel: str, max_phase: Any = None) -> float:
    rel = str(rel or "").lower()
    phase = safe_float(max_phase, default=0.0)

    if rel == "indication":
        if phase >= 4:
            return 1.00
        if phase >= 3:
            return 0.90
        if phase >= 2:
            return 0.75
        return 0.70

    if rel == "off_label_use":
        return 0.65

    if rel == "tested_indication":
        if phase >= 4:
            return 0.85
        if phase >= 3:
            return 0.75
        if phase >= 2:
            return 0.50
        if phase > 0:
            return 0.35
        return 0.30

    if rel == "studied_for_treatment_of":
        # CTD therapeutic-style literature support. It is weaker than indication
        # or phase-coded tested_indication, but it is still discovery-supportive.
        return 0.35

    if rel == "studied_for_marker_mechanism_of":
        # Marker/mechanism studies are retained for provenance and masking only;
        # they are not therapeutic candidate evidence.
        return 0.0

    if rel == "contraindication":
        return -0.80

    return 0.0


def get_candidate_prior_from_similar_diseases(
    driver,
    target_disease: str,
    similar_diseases: List[Dict[str, Any]],
    heldout_diseases: Optional[List[str]] = None,
    positive_rels: Optional[List[str]] = None,
    include_contraindication_as_risk: bool = True,
    limit: int = 500,
) -> List[Dict[str, Any]]:
    """
    Use vector-similar diseases as treatment prior.

    Important:
    - Excludes heldout diseases.
    - Uses similar diseases' drug-disease edges.
    - Does not use target disease direct drug-disease edges.
    """
    heldout_diseases = unique_keep_order((heldout_diseases or []) + [target_disease])
    positive_rels = positive_rels or DISCOVERY_POSITIVE_DD_RELS

    rels = list(positive_rels)
    if include_contraindication_as_risk:
        rels += ["contraindication"]

    similar_names = [
        r["disease"]
        for r in similar_diseases
        if r.get("disease")
    ]

    sim_score_by_name = {
        normalize_name(r["disease"]): safe_float(r.get("similarity_score"))
        for r in similar_diseases
        if r.get("disease")
    }

    if not similar_names:
        return []

    rel_union = "|".join(rels)

    q = f"""
    MATCH (near:DISEASE)-[r:{rel_union}]->(dr:DRUG)
    WHERE near.name IN $similar_diseases
      AND NOT toLower(near.name) IN [x IN $heldout_diseases | toLower(x)]
      AND dr.name IS NOT NULL

    RETURN
      dr.name AS drug,
      collect(DISTINCT near.name)[0..20] AS prior_diseases,
      collect(DISTINCT type(r)) AS prior_rels,
      collect(DISTINCT r.Max_Phase)[0..10] AS max_phases,
      collect(DISTINCT {{rel:type(r), phase:r.Max_Phase, disease:near.name}})[0..50] AS prior_relation_edges,
      count(DISTINCT r) AS prior_edge_count,
      dr.Group AS `Group`,
      dr.Max_Phase AS drug_max_phase,
      dr.Black_Box AS BlackBox,
      dr.Withdrawn_Flag AS Withdrawn,
      dr.Inorganic_Flag AS Inorganic,
      dr.Molecular_Weight AS MW,
      dr.Polar_Surface_Area AS TPSA,
      coalesce(dr.CX_LogP, dr.XLogP, dr.CLogP, dr.AlogP) AS LogP,
      dr.QED_Weighted AS QED
    """

    rows = run_query(
        driver,
        q,
        {
            "similar_diseases": similar_names,
            "heldout_diseases": heldout_diseases,
        },
        timeout_sec=90,
    )

    out = []
    for row in rows:
        prior_diseases = row.get("prior_diseases") or []
        prior_rels = row.get("prior_rels") or []
        max_phases = flatten_values(row.get("max_phases") or [])
        prior_relation_edges = row.get("prior_relation_edges") or []

        # Similarity component: strongest similar disease carrying this prior.
        sim_component = 0.0
        for d in prior_diseases:
            sim_component = max(sim_component, sim_score_by_name.get(normalize_name(d), 0.0))

        # Relation strength component.
        #
        # Important: negative-only priors must remain negative.  The older
        # implementation initialized rel_component=0 and used max(...), which
        # converted a pure contraindication prior into 0.0.  That made
        # contraindication-only disease-neighborhood transfer look like a
        # positive similarity prior.  Here, positive clinical support dominates
        # when present; otherwise a negative-only relation keeps its negative
        # sign for downstream risk-aware scoring.
        rel_weights = []
        relation_weight_details = []
        if prior_relation_edges:
            for edge in prior_relation_edges:
                rel = edge.get("rel") if isinstance(edge, dict) else None
                phase = edge.get("phase") if isinstance(edge, dict) else None
                w = relation_weight(rel, phase)
                rel_weights.append(w)
                relation_weight_details.append({"rel": rel, "phase": phase, "weight": round(float(w), 4)})
        else:
            # Backward-compatible fallback for older cached result rows.  Do not combine
            # unrelated relation types with unrelated drug-level phases; use relation-only
            # weights when edge-level phase provenance is absent.
            for rel in prior_rels:
                w = relation_weight(rel, None)
                rel_weights.append(w)
                relation_weight_details.append({"rel": rel, "phase": None, "weight": round(float(w), 4)})

        positive_rel_weights = [float(w) for w in rel_weights if float(w) > 0.0]
        negative_rel_weights = [float(w) for w in rel_weights if float(w) < 0.0]
        if positive_rel_weights:
            rel_component = max(positive_rel_weights)
            relation_component_mode = "positive_support"
        elif negative_rel_weights:
            rel_component = min(negative_rel_weights)
            relation_component_mode = "negative_only"
        else:
            rel_component = 0.0
            relation_component_mode = "neutral_or_unknown"

        # Edge count saturation.
        count_component = min(math.log1p(safe_float(row.get("prior_edge_count"))) / math.log1p(10), 1.0)

        # Contra risk signal separated from positive prior.
        contraindication_hit = any(str(r).lower() == "contraindication" for r in prior_rels)

        if contraindication_hit and not positive_rel_weights and rel_component < 0:
            embedding_prior_score = -0.5 * abs(rel_component) * max(sim_component, 0.1)
        else:
            embedding_prior_score = (
                0.60 * sim_component
                + 0.30 * max(rel_component, 0.0)
                + 0.10 * count_component
            )

        row["similarity_component"] = sim_component
        row["relation_component"] = rel_component
        row["relation_component_mode"] = relation_component_mode
        row["relation_component_weights"] = [round(float(w), 4) for w in rel_weights]
        row["relation_weight_details"] = relation_weight_details
        row["count_component"] = count_component
        row["embedding_prior_score"] = round(float(embedding_prior_score), 6)
        row["contraindication_prior_hit"] = bool(contraindication_hit)
        row["has_positive_therapeutic_prior"] = bool(positive_rel_weights)
        row["prior_seed_eligible"] = bool(positive_rel_weights)

        out.append(row)

    out.sort(
        key=lambda x: (
            -safe_float(x.get("embedding_prior_score"), 0.0),
            -int(x.get("prior_edge_count") or 0),
            normalize_name(x.get("drug")),
        )
    )
    return out[: max(int(limit or 0), 0)]


# ============================================================
# 10. Optional drug side-effect risk overlap
# ============================================================

def get_drug_phenotype_risk_overlap(
    driver,
    disease: str,
    candidate_drugs: List[str],
    limit_per_drug: int = 20,
) -> Dict[str, Dict[str, Any]]:
    """
    side_effect is DRUG-PHENO.
    Use it only as risk/phenotype overlap, not direct disease masking.
    """
    if not candidate_drugs:
        return {}

    q = """
    MATCH (dis:DISEASE)
    WHERE toLower(trim(dis.name)) = toLower(trim($disease))

    MATCH (dis)-[:phenotype_present]-(ph:PHENO)
    MATCH (dr:DRUG)-[:side_effect]-(ph)

    WHERE dr.name IN $candidate_drugs

    RETURN
      dr.name AS drug,
      collect(DISTINCT ph.name)[0..$limit_per_drug] AS overlapping_side_effect_phenotypes,
      count(DISTINCT ph) AS overlap_count
    """

    rows = run_query(
        driver,
        q,
        {
            "disease": disease,
            "candidate_drugs": candidate_drugs,
            "limit_per_drug": int(limit_per_drug),
        },
        timeout_sec=60,
    )

    return {
        r["drug"]: {
            "overlapping_side_effect_phenotypes": r.get("overlapping_side_effect_phenotypes") or [],
            "overlap_count": r.get("overlap_count") or 0,
            "side_effect_risk_penalty": min(safe_float(r.get("overlap_count")) * 0.01, 0.10),
        }
        for r in rows
        if r.get("drug")
    }


# ============================================================
# 11. Main high-level API
# ============================================================

def expand_heldout_disease_family(
    driver,
    disease: str,
    disease_element_id: Optional[str] = None,
    disease_element_ids: Optional[List[str]] = None,
    max_nodes: int = 200,
) -> list[str]:
    """Expand held-out masking from the exact-name bundle through ontology edges."""
    exact_ids = unique_keep_order(
        [x for x in (disease_element_ids or []) if x]
        + ([disease_element_id] if disease_element_id else [])
    )
    q = """
    MATCH (d:DISEASE)
    WHERE (size($disease_element_ids) > 0 AND elementId(d) IN $disease_element_ids)
       OR (size($disease_element_ids) = 0
           AND toLower(trim(d.name)) = toLower(trim($disease)))
    OPTIONAL MATCH (d)-[:parent_child*1..2]-(pc:DISEASE)
    WHERE NOT elementId(pc) IN $disease_element_ids
      AND toLower(trim(coalesce(pc.name, ''))) <> toLower(trim($disease))
    WITH collect(DISTINCT pc.name) AS related_names
    WITH [$disease] + [name IN related_names WHERE name IS NOT NULL] AS all_names
    UNWIND all_names AS name
    WITH DISTINCT name
    WHERE name IS NOT NULL
    RETURN name
    LIMIT $max_nodes
    """
    rows = run_query(
        driver,
        q,
        {
            "disease": disease,
            "disease_element_ids": exact_ids,
            "max_nodes": max_nodes,
        },
        timeout_sec=60,
    )
    return unique_keep_order([r["name"] for r in rows if r.get("name")])

def ensure_disease_embedding(
    driver,
    disease: str,
    disease_element_id: Optional[str] = None,
    disease_element_ids: Optional[List[str]] = None,
    disease_df: Optional[pd.DataFrame] = None,
    disease_features_csv: Optional[str | Path] = None,
    force_rebuild_embedding: bool = False,
    mask_terms: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    EMBEDDING CREATION ONLY.

    This function does NOT:
    - expand heldout disease family
    - query similar diseases
    - use treatment priors
    - use positive_rels / negative_rels

    It only creates or loads a unique embedding for the exact disease node.
    """
    mask_terms = mask_terms or []
    mask_signature = "||".join(sorted({str(x).strip().lower() for x in mask_terms if str(x).strip()}))
    stored = get_stored_disease_embedding(
        driver,
        disease,
        disease_element_id=disease_element_id,
    )
    stored_signature = str((stored or {}).get("mask_terms_signature") or "")
    needs_mask_rebuild = bool(mask_signature) and stored_signature != mask_signature
    bundle_signature = "||".join(sorted({str(x) for x in (disease_element_ids or []) if x}))
    stored_bundle_signature = str((stored or {}).get("bundle_signature") or "")
    needs_bundle_rebuild = len(set(disease_element_ids or [])) > 1 and stored_bundle_signature != bundle_signature

    if force_rebuild_embedding or needs_mask_rebuild or needs_bundle_rebuild or not stored or not stored.get("embedding"):
        built = build_and_store_disease_embedding(
            driver=driver,
            disease=disease,
            disease_element_id=disease_element_id,
            disease_element_ids=disease_element_ids,
            disease_df=disease_df,
            disease_features_csv=disease_features_csv,
            mask_terms=mask_terms,
            create_index=True,
        )

        return {
            "disease": disease,
            "summary_text": built.get("summary_text", ""),
            "embedding": built.get("embedding"),
            "embedding_dim": len(built.get("embedding") or []),
            "created_or_rebuilt": True,
            "updated_nodes": built.get("updated_nodes", []),
            "mask_terms_applied_count": len(mask_terms),
            "mask_terms_signature": mask_signature,
        }

    return {
        "disease": disease,
        "summary_text": stored.get("safe_context_text") or "",
        "embedding": stored.get("embedding"),
        "embedding_dim": len(stored.get("embedding") or []),
        "created_or_rebuilt": False,
        "updated_nodes": [],
        "mask_terms_applied_count": int(stored.get("mask_terms_count") or 0),
        "mask_terms_signature": stored_signature,
    }
    
def prepare_embedding_prior_for_case(
    driver,
    case: Dict[str, Any],
    disease_df: Optional[pd.DataFrame] = None,
    disease_features_csv: Optional[str | Path] = None,
    top_similar_diseases: int = 50,
    min_similarity: float = 0.85,
    prior_limit: int = 500,
    force_rebuild_embedding: bool = False,
    eval_diseases: Optional[List[str]] = None,
    mask_eval_diseases_from_prior: bool = False,
    max_similarity_for_prior: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Call this once per test case before GraphRAG candidate ranking.

    Returns:
    {
      "disease": ...,
      "summary_text": ...,
      "similar_diseases": [...],
      "candidate_priors": [...],
      "prior_score_by_drug": {drug_name: score},
      "risk_by_drug": {...}
    }
    """
    disease = case["disease"]
    disease_element_id = case.get("_disease_element_id")
    disease_element_ids = case.get("_disease_exact_element_ids") or []
    heldout_diseases = case.get("heldout_diseases")
    if not heldout_diseases:
        heldout_diseases = expand_heldout_disease_family(
            driver,
            disease,
            disease_element_id=disease_element_id,
            disease_element_ids=disease_element_ids,
            max_nodes=100,
        )

    # Direct target-disease drug names may be supplied only as redaction terms
    # to remove treatment mentions from disease text. They are never exposed to
    # prior construction, ranking, or candidate scoring.
    mask_terms = case.get("_redaction_drug_terms") or []

    embedding_obj = ensure_disease_embedding(
        driver=driver,
        disease=disease,
        disease_element_id=disease_element_id,
        disease_element_ids=disease_element_ids,
        disease_df=disease_df,
        disease_features_csv=disease_features_csv,
        force_rebuild_embedding=force_rebuild_embedding,
        mask_terms=mask_terms,
    )
    
    summary_text = embedding_obj["summary_text"]
    embedding = embedding_obj["embedding"]

    if not embedding:
        raise RuntimeError(f"No embedding available for disease={disease}")
        
    similar_raw = query_similar_diseases_by_embedding(
        driver=driver,
        disease=disease,
        embedding=embedding,
        top_k=top_similar_diseases,
        min_score=min_similarity,
        exclude_names=heldout_diseases,
    )

    similar = similar_raw
    masked_similar_diseases = []
    if mask_eval_diseases_from_prior or max_similarity_for_prior is not None:
        similar, masked_similar_diseases = filter_similar_disease_priors(
            similar_diseases=similar_raw,
            eval_diseases=(eval_diseases or []) if mask_eval_diseases_from_prior else [],
            max_similarity_for_prior=max_similarity_for_prior,
        )

    # Retrieval priors are deliberately separated from strict Eval1 gold labels.
    # Eval1 gold can remain indication-only, while the discovery prior should retain
    # weak but therapeutic-supportive ChEMBL/CTD evidence such as off_label_use,
    # tested_indication, and studied_for_treatment_of. Marker/mechanism-study
    # relations are context-only and are excluded from candidate generation. Otherwise, the
    # candidate generator silently becomes an indication-recovery engine rather than
    # a drug-repurposing hypothesis generator.
    # Relation scope is global and identical for every disease. It cannot be
    # overridden through benchmark rows or disease-specific case metadata.
    retrieval_prior_rels = list(DISCOVERY_POSITIVE_DD_RELS)
    strict_eval_positive_rels = []

    all_prior_rows = get_candidate_prior_from_similar_diseases(
        driver=driver,
        target_disease=disease,
        similar_diseases=similar,
        heldout_diseases=heldout_diseases,
        positive_rels=retrieval_prior_rels,
        include_contraindication_as_risk=True,
        limit=prior_limit,
    )
    # Contraindication-only neighbors are retained as counter-evidence metadata but
    # can never become prior-drug seeds. Mixed positive+contraindication rows remain
    # eligible and carry the risk flag downstream.
    priors = [r for r in all_prior_rows if bool(r.get("prior_seed_eligible"))]
    contraindication_only_priors = [
        r for r in all_prior_rows
        if bool(r.get("contraindication_prior_hit")) and not bool(r.get("prior_seed_eligible"))
    ]

    prior_score_by_drug = {
        row["drug"]: row.get("embedding_prior_score", 0.0)
        for row in priors
        if row.get("drug")
    }

    # Optional side-effect phenotype overlap risk.
    candidate_drugs = [row["drug"] for row in priors if row.get("drug")]
    risk_by_drug = get_drug_phenotype_risk_overlap(
        driver=driver,
        disease=disease,
        candidate_drugs=candidate_drugs[:300],
    )

    return {
        "disease": disease,
        "heldout_diseases": heldout_diseases,
        "summary_text": summary_text,
        "similar_diseases": similar,
        "similar_diseases_raw": similar_raw,
        "masked_similar_diseases": masked_similar_diseases,
        "candidate_priors": priors,
        "contraindication_only_priors": contraindication_only_priors,
        "prior_score_by_drug": prior_score_by_drug,
        "risk_by_drug": risk_by_drug,
        "retrieval_prior_rels": retrieval_prior_rels,
        "strict_eval_positive_rels": strict_eval_positive_rels,
        "disease_embedding_mask_terms_count": embedding_obj.get("mask_terms_applied_count", 0),
    }


def attach_embedding_prior_scores(
    candidate_rows: List[Dict[str, Any]],
    embedding_prior: Dict[str, Any],
    prior_weight: float = 2.0,
    risk_weight: float = 0.2,
    score_field: str = "score",
) -> List[Dict[str, Any]]:
    """
    Add embedding prior score to existing GraphRAG candidate/path rows.

    Key change:
    - embedding prior is upweighted because graph_score is usually 6-11
      while embedding_prior_score is usually 0.6-0.9.
    - side_effect risk is kept weak at retrieval stage.
    """
    prior_score_by_drug = embedding_prior.get("prior_score_by_drug") or {}
    risk_by_drug = embedding_prior.get("risk_by_drug") or {}

    out = []

    for row in candidate_rows or []:
        drug = row.get("drug") or row.get("drug_name")
        base_score = safe_float(row.get(score_field), 0.0)

        emb_score = safe_float(prior_score_by_drug.get(drug), 0.0)

        # If this is a prior-only row, it may already carry embedding_prior_score.
        if emb_score <= 0:
            emb_score = safe_float(row.get("embedding_prior_score"), 0.0)

        risk_info = risk_by_drug.get(drug) or {}
        risk_penalty = safe_float(risk_info.get("side_effect_risk_penalty"), 0.0)

        final_score = (
            base_score
            + prior_weight * emb_score
            - risk_weight * risk_penalty
        )

        new_row = dict(row)
        new_row["base_graph_score"] = base_score
        new_row["embedding_prior_score"] = emb_score
        new_row["side_effect_risk_penalty"] = risk_penalty
        new_row["final_score_with_embedding_prior"] = round(final_score, 6)

        if risk_info:
            new_row["overlapping_side_effect_phenotypes"] = risk_info.get(
                "overlapping_side_effect_phenotypes", []
            )

        out.append(new_row)

    out.sort(
        key=lambda x: x.get("final_score_with_embedding_prior", 0.0),
        reverse=True,
    )
    return out

def diversify_candidate_ranking(
    rows: List[Dict[str, Any]],
    graph_head_n: int = 15,
    prior_only_head_n: int = 5,
) -> List[Dict[str, Any]]:
    """
    Force a small number of embedding-prior-only candidates into the early list.

    Why:
    - GraphRAG scores are large, so prior-only candidates with score ~0.7
      otherwise fall to rank 100+.
    - This preserves the original purpose of embedding prior:
      rescuing candidates missed by anchor-guided GraphRAG.
    """
    rows = rows or []

    def final_score(x):
        return safe_float(x.get("final_score_with_embedding_prior"), 0.0)

    graph_rows = [
        r for r in rows
        if r.get("source_graph_candidate")
    ]

    prior_only_rows = [
        r for r in rows
        if r.get("source_embedding_prior") and not r.get("source_graph_candidate")
    ]

    graph_rows = sorted(graph_rows, key=final_score, reverse=True)
    prior_only_rows = sorted(
        prior_only_rows,
        key=lambda x: safe_float(x.get("embedding_prior_score"), 0.0),
        reverse=True,
    )

    selected = []
    seen = set()

    def add(row):
        drug = row.get("drug") or row.get("drug_name")
        key = normalize_name(drug)
        if not key or key in seen:
            return
        seen.add(key)
        selected.append(row)

    # Main high-confidence GraphRAG candidates
    for r in graph_rows[:graph_head_n]:
        add(r)

    # Explicit rescue lane for embedding-prior-only candidates
    for r in prior_only_rows[:prior_only_head_n]:
        add(r)

    # Remaining candidates by final score
    for r in sorted(rows, key=final_score, reverse=True):
        add(r)

    for rank, r in enumerate(selected, start=1):
        r["rank"] = rank

    return selected
    
def merge_graph_candidates_with_embedding_priors(
    graph_rows: List[Dict[str, Any]],
    embedding_prior: Dict[str, Any],
    graph_score_field: str = "score",
    max_prior_only: int = 200,
) -> List[Dict[str, Any]]:
    """
    Union:
    - existing GraphRAG candidates
    - embedding prior candidates from similar diseases

    This helps when Cypher pattern misses a gold drug.
    """
    by_drug: Dict[str, Dict[str, Any]] = {}

    for row in graph_rows or []:
        drug = row.get("drug") or row.get("drug_name")
        if not drug:
            continue

        key = normalize_name(drug)
        base_score = safe_float(row.get(graph_score_field), 0.0)

        cur = by_drug.get(key)
        if cur is None or base_score > safe_float(cur.get(graph_score_field), 0.0):
            new_row = dict(row)
            new_row["drug"] = drug
            new_row["source_graph_candidate"] = True
            new_row["source_embedding_prior"] = bool(new_row.get("source_embedding_prior", False))
            by_drug[key] = new_row

    # Add prior-only candidates.
    priors = embedding_prior.get("candidate_priors") or []
    for row in priors[:max_prior_only]:
        drug = row.get("drug")
        if not drug:
            continue

        key = normalize_name(drug)
        if key not in by_drug:
            by_drug[key] = {
                "drug": drug,
                graph_score_field: 0.0,
                "source_graph_candidate": False,
                "source_embedding_prior": True,
                "prior_diseases": row.get("prior_diseases", []),
                "prior_rels": row.get("prior_rels", []),
                "embedding_prior_score": row.get("embedding_prior_score", 0.0),
                "relation_component": row.get("relation_component"),
                "relation_component_mode": row.get("relation_component_mode"),
                "relation_component_weights": row.get("relation_component_weights", []),
                "similarity_component": row.get("similarity_component"),
                "count_component": row.get("count_component"),
                "contraindication_prior_hit": row.get("contraindication_prior_hit", False),
                "Group": row.get("Group"),
                "MaxPhase": row.get("drug_max_phase"),
                "BlackBox": row.get("BlackBox"),
                "Withdrawn": row.get("Withdrawn"),
                "Inorganic": row.get("Inorganic"),
                "MW": row.get("MW"),
                "TPSA": row.get("TPSA"),
                "LogP": row.get("LogP"),
                "QED": row.get("QED"),
            }
        else:
            by_drug[key]["source_embedding_prior"] = True
            by_drug[key]["prior_diseases"] = row.get("prior_diseases", [])
            by_drug[key]["prior_rels"] = row.get("prior_rels", [])
            by_drug[key]["embedding_prior_score"] = row.get("embedding_prior_score", 0.0)
            by_drug[key]["relation_component"] = row.get("relation_component")
            by_drug[key]["relation_component_mode"] = row.get("relation_component_mode")
            by_drug[key]["relation_component_weights"] = row.get("relation_component_weights", [])
            by_drug[key]["similarity_component"] = row.get("similarity_component")
            by_drug[key]["count_component"] = row.get("count_component")
            by_drug[key]["contraindication_prior_hit"] = row.get("contraindication_prior_hit", False)

    merged = list(by_drug.values())

    # Attach final scores.
    merged = attach_embedding_prior_scores(
        candidate_rows=merged,
        embedding_prior=embedding_prior,
        prior_weight=1.0,
        risk_weight=0.1,
        score_field=graph_score_field,
    )
    
    for row in merged:
        if safe_float(row.get("embedding_prior_score"), 0.0) > 0:
            row["source_embedding_prior"] = True

    # # Force a small rescue lane for prior-only candidates.
    # merged = diversify_candidate_ranking(
    #     merged,
    #     graph_head_n=15,
    #     prior_only_head_n=5,
    # )
    
    return merged

def get_embedding_candidate_diseases_from_csv(
    disease_features_csv: str | Path,
    limit: Optional[int] = None,
) -> List[str]:
    """
    Get unique disease names from disease_features.csv.

    Preference:
    - group_name_bert when available
    - otherwise mondo_name
    """
    df = load_disease_features(disease_features_csv)

    names = []

    if "group_name_bert" in df.columns:
        names.extend([
            clean_nan_text(x)
            for x in df["group_name_bert"].tolist()
            if clean_nan_text(x)
        ])

    if "mondo_name" in df.columns:
        names.extend([
            clean_nan_text(x)
            for x in df["mondo_name"].tolist()
            if clean_nan_text(x)
        ])

    names = unique_keep_order(names)

    if limit is not None:
        names = names[: int(limit)]

    return names


def build_disease_embeddings_batch(
    driver,
    disease_features_csv: str | Path,
    diseases: Optional[List[str]] = None,
    limit: Optional[int] = None,
    force_rebuild_embedding: bool = False,
    sleep_sec: float = 0.0,
    out_jsonl: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    OFFLINE BATCH STEP.

    For each disease:
    - create unique safe disease summary
    - create embedding
    - store on exact DISEASE node

    This does NOT query similar diseases and does NOT create candidate priors.
    """
    disease_df = load_disease_features(disease_features_csv)

    if diseases is None:
        diseases = get_embedding_candidate_diseases_from_csv(
            disease_features_csv=disease_features_csv,
            limit=limit,
        )
    else:
        diseases = unique_keep_order(diseases)
        if limit is not None:
            diseases = diseases[: int(limit)]

    out_path = Path(out_jsonl) if out_jsonl else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "total": len(diseases),
        "success": 0,
        "failed": 0,
        "skipped_or_loaded": 0,
        "failures": [],
    }

    for i, disease in enumerate(diseases, start=1):
        print(f"\n[BATCH EMBEDDING {i}/{len(diseases)}] {disease}", flush=True)

        try:
            obj = ensure_disease_embedding(
                driver=driver,
                disease=disease,
                disease_df=disease_df,
                force_rebuild_embedding=force_rebuild_embedding,
            )

            stats["success"] += 1
            if not obj.get("created_or_rebuilt"):
                stats["skipped_or_loaded"] += 1

            rec = {
                "disease": disease,
                "embedding_dim": obj.get("embedding_dim"),
                "created_or_rebuilt": obj.get("created_or_rebuilt"),
                "summary_preview": (obj.get("summary_text") or "")[:500],
            }

            if out_path:
                with out_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            print(
                f"  OK | dim={obj.get('embedding_dim')} | "
                f"rebuilt={obj.get('created_or_rebuilt')}",
                flush=True,
            )

        except Exception as e:
            stats["failed"] += 1
            stats["failures"].append({"disease": disease, "error": str(e)})
            print(f"  FAIL | {e}", flush=True)

        if sleep_sec > 0:
            time.sleep(sleep_sec)

    return stats
# ============================================================
# 12. CLI test
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["build_one", "query_prior", "build_batch"],
        default="query_prior",
        help=(
            "build_one: only create/store embedding for one disease. "
            "query_prior: create/load embedding, vector-search similar diseases, and get priors. "
            "build_batch: create/store embeddings for many diseases."
        ),
    )

    parser.add_argument("--disease", default=None)
    parser.add_argument("--disease_features_csv", default="data/rag_corpus/disease_features.csv")
    parser.add_argument("--neo4j_password", default=None)

    parser.add_argument("--top_similar", type=int, default=50)
    parser.add_argument("--min_similarity", type=float, default=0.85)
    parser.add_argument("--mask_eval_diseases_from_prior", action="store_true")
    parser.add_argument("--max_similarity_for_prior", type=float, default=None)
    parser.add_argument("--eval_disease_list", default=None, help="Optional text/CSV file of eval diseases to mask from similar-disease priors.")
    parser.add_argument("--force_rebuild", action="store_true")
    parser.add_argument("--out", default=None)

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--disease_list", default=None)
    parser.add_argument("--sleep_sec", type=float, default=0.0)

    args = parser.parse_args()

    driver = make_driver(neo4j_password=args.neo4j_password)

    eval_diseases_for_mask = []
    if args.eval_disease_list:
        p = Path(args.eval_disease_list)
        if p.suffix.lower() == ".csv":
            _df = pd.read_csv(p)
            if "disease" not in _df.columns:
                raise ValueError("--eval_disease_list CSV must contain a disease column")
            eval_diseases_for_mask = [clean_nan_text(x) for x in _df["disease"].tolist() if clean_nan_text(x)]
        else:
            eval_diseases_for_mask = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

    try:
        if args.mode in {"build_one", "query_prior"} and not args.disease:
            raise ValueError("--disease is required for build_one/query_prior mode")

        if args.mode == "build_one":
            disease_df = load_disease_features(args.disease_features_csv)

            obj = ensure_disease_embedding(
                driver=driver,
                disease=args.disease,
                disease_df=disease_df,
                force_rebuild_embedding=args.force_rebuild,
            )

            payload = {
                "mode": "build_one",
                "disease": args.disease,
                "embedding_dim": obj.get("embedding_dim"),
                "created_or_rebuilt": obj.get("created_or_rebuilt"),
                "summary_text": obj.get("summary_text"),
            }

            print("\n================ Disease embedding created/loaded ================")
            print(json.dumps(
                {k: v for k, v in payload.items() if k != "summary_text"},
                ensure_ascii=False,
                indent=2,
            ))
            print("\n================ Summary preview ================")
            print((obj.get("summary_text") or "")[:3000])

            if args.out:
                out_path = Path(args.out)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                print(f"\nSaved: {out_path}")

        elif args.mode == "query_prior":
            case = {
                "case_id": f"DISC-{args.disease}",
                "disease": args.disease,
            }

            prior = prepare_embedding_prior_for_case(
                driver=driver,
                case=case,
                disease_features_csv=args.disease_features_csv,
                top_similar_diseases=args.top_similar,
                min_similarity=args.min_similarity,
                force_rebuild_embedding=args.force_rebuild,
                eval_diseases=eval_diseases_for_mask,
                mask_eval_diseases_from_prior=args.mask_eval_diseases_from_prior,
                max_similarity_for_prior=args.max_similarity_for_prior,
            )

            print("\n================ Disease embedding summary ================")
            print(prior["summary_text"][:3000])

            print("\n================ Heldout diseases ================")
            for d in prior.get("heldout_diseases", [])[:80]:
                print(" ", d)

            print("\n================ Similar diseases after heldout exclusion ================")
            for r in prior["similar_diseases"][:20]:
                print(round(r["similarity_score"], 4), r["disease"])

            print("\n================ Candidate priors ================")
            for r in prior["candidate_priors"][:50]:
                print(
                    r["drug"],
                    "score=", r["embedding_prior_score"],
                    "rels=", r.get("prior_rels"),
                    "near=", r.get("prior_diseases", [])[:3],
                )

            if args.out:
                out_path = Path(args.out)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(prior, f, ensure_ascii=False, indent=2)
                print(f"\nSaved: {out_path}")

        elif args.mode == "build_batch":
            diseases = None

            if args.disease_list:
                p = Path(args.disease_list)
                diseases = [
                    line.strip()
                    for line in p.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]

            stats = build_disease_embeddings_batch(
                driver=driver,
                disease_features_csv=args.disease_features_csv,
                diseases=diseases,
                limit=args.limit,
                force_rebuild_embedding=args.force_rebuild,
                sleep_sec=args.sleep_sec,
                out_jsonl=args.out,
            )

            print("\n================ Batch embedding stats ================")
            print(json.dumps(stats, ensure_ascii=False, indent=2))

    finally:
        driver.close()


if __name__ == "__main__":
    main()