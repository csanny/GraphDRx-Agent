# ============================================================
# Drug embedding prior / MoA-aware reranking utilities
# For GraphRAG drug repurposing pipeline
#
# Purpose:
#   - Build SAFE drug text from Drug_data_All_RAG.csv + KG drug-gene edges
#   - Exclude indication / disease-treatment text from embeddings
#   - Store embeddings on DRUG nodes in Neo4j
#   - Compute drug representation similarity to disease-context anchor text
#   - Merge this score with GraphRAG candidates
#
# Safe fields used:
#   mechanism_of_action, pharmacodynamics, pathway, category, ATC codes,
#   indication class, protein binding, half-life, KG target/action summary.
#   Legacy C_/K_/P_-prefixed column names are accepted as fallbacks only.
#
# Excluded from embedding:
#   indication, description, USAN definition, SMILES, synonyms,
#   direct drug-disease relations.
# ============================================================

import os
import re
import csv
import json
import time
import math
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from neo4j import GraphDatabase, Query


# ------------------------------------------------------------
# Defaults
# ------------------------------------------------------------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
EMBED_MODEL = os.getenv("DRUG_EMBED_MODEL", "nomic-embed-text")

DEFAULT_DRUG_RAG_CSV = "data/rag_corpus/Drug_data_All_RAG.csv"

DRUG_GENE_RELS = [
    "target", "inhibition", "activation", "binding", "modulation",
    "enzyme", "transporter", "carrier",
]

DIRECT_DD_RELS = [
    "indication",
    "off_label_use",
    "tested_indication",
    "studied_for_treatment_of",
    "studied_for_marker_mechanism_of",
    "contraindication",
]

DRUG_RAG_FIELD_ALIASES = {
    "mechanism_of_action": ["mechanism_of_action", "Mechanism_of_Action", "mechanism", "Mechanism", "K_mechanism_of_action"],
    "pharmacodynamics": ["pharmacodynamics", "Pharmacodynamics", "K_pharmacodynamics"],
    "pathway": ["pathway", "Pathway", "K_pathway"],
    "category": ["category", "Category", "drug_category", "Drug_Category", "K_category"],
    "atc_codes": ["ATC Codes", "ATC_Codes", "ATC", "C_ATC Codes"],
    "indication_class": ["Indication Class", "Indication_Class", "indication_class", "C_Indication Class"],
    "protein_binding": ["protein_binding", "Protein_Binding", "protein_binding_pct", "Protein_Binding_pct", "K_protein_binding"],
    "half_life": ["half_life", "Half_Life", "Half_Life_h", "K_half_life"],
    "indication": ["indication", "Indication", "K_indication"],
    "description": ["description", "Description", "K_description"],
    "usan_definition": ["USAN Definition", "USAN_Definition", "C_USAN Definition"],
    "synonyms": ["Merged_Synonyms", "synonyms", "Synonyms"],
}

SAFE_TEXT_FIELDS = [
    "mechanism_of_action",
    "pharmacodynamics",
    "pathway",
    "category",
    "atc_codes",
    "indication_class",
    "protein_binding",
    "half_life",
]

# Do NOT use these in embedding text.
EXCLUDED_TEXT_FIELDS = [
    "indication",
    "description",
    "usan_definition",
    "SMILES",
    "synonyms",
]

DRUG_EMBEDDING_PROP = "safe_drug_embedding"
DRUG_TEXT_PROP = "safe_drug_text"
DRUG_EMBEDDING_MODEL_PROP = "safe_drug_embedding_model"
DRUG_EMBEDDING_UPDATED_AT_PROP = "safe_drug_embedding_updated_at"
DRUG_EMBEDDING_SOURCE_PROP = "safe_drug_embedding_source"

# Ollama embedding models have finite context windows. Disease-context anchors can
# become verbose enough to trigger HTTP 500 input-length errors, so drug-side
# embedding text is compacted and hard-capped before every embedding call.
DRUG_EMBED_QUERY_MAX_CHARS = int(os.getenv("DRUG_EMBED_QUERY_MAX_CHARS", "3500"))
DRUG_SAFE_TEXT_MAX_CHARS = int(os.getenv("DRUG_SAFE_TEXT_MAX_CHARS", "3500"))


# ------------------------------------------------------------
# Basic helpers
# ------------------------------------------------------------
def normalize_name(x: Any) -> str:
    return str(x or "").strip().lower()


def clean_text(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null", "[]"}:
        return ""
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def truncate_text(s: str, max_chars: int = 1200) -> str:
    s = clean_text(s)
    if len(s) <= max_chars:
        return s
    return s[:max_chars].rsplit(" ", 1)[0]


def compact_embedding_text(text: str, max_chars: int) -> str:
    """Normalize whitespace and apply a word-boundary hard cap for embeddings."""
    s = clean_text(text)
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    cut = s[:max_chars]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.strip()


def get_rag_field(row: Optional[Dict[str, Any]], canonical_field: str) -> Any:
    """Return a RAG CSV field by canonical name, accepting prefix-free and legacy names."""
    if not row:
        return None
    aliases = DRUG_RAG_FIELD_ALIASES.get(canonical_field, [canonical_field])
    for key in aliases:
        if key in row and clean_text(row.get(key)):
            return row.get(key)
    return None


def mask_terms_in_text(text: str, terms: Optional[List[str]] = None, mask: str = "[MASKED]") -> str:
    out = str(text or "")
    for term in sorted(terms or [], key=lambda x: len(str(x)), reverse=True):
        t = str(term or "").strip()
        if not t:
            continue
        out = re.sub(re.escape(t), mask, out, flags=re.IGNORECASE)
    return out


def parse_list_like_text(x: Any, max_items: int = 40) -> List[str]:
    """
    Handles strings like:
      "['a', 'b']"
      "a | b | c"
      "a; b; c"
    """
    s = clean_text(x)
    if not s:
        return []

    # Try literal list from CSV
    if s.startswith("[") and s.endswith("]"):
        try:
            import ast
            obj = ast.literal_eval(s)
            if isinstance(obj, (list, tuple)):
                return [clean_text(v) for v in obj if clean_text(v)][:max_items]
        except Exception:
            pass

    # ATC / pathway style separators
    parts = re.split(r"\s*\|\s*|\s*;\s*", s)
    parts = [clean_text(p) for p in parts if clean_text(p)]
    if len(parts) > 1:
        return parts[:max_items]

    return [s[:2000]]


def make_driver(neo4j_password: Optional[str] = None):
    password = neo4j_password or NEO4J_PASSWORD
    if not password:
        raise RuntimeError(
            "Neo4j password is empty. Set NEO4J_PASSWORD or pass --neo4j-password."
        )
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, password))


def run_query(driver, query: str, params: Optional[Dict[str, Any]] = None, timeout_sec: int = 60) -> List[Dict[str, Any]]:
    with driver.session() as session:
        result = session.run(Query(query, timeout=timeout_sec), params or {})
        return [r.data() for r in result]


# ------------------------------------------------------------
# Ollama embedding
# ------------------------------------------------------------
def ollama_embed(text: str, model: str = EMBED_MODEL, timeout_sec: int = 120) -> List[float]:
    """
    Uses Ollama /api/embeddings.
    Default model: nomic-embed-text.
    """
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/embeddings"
    prompt_text = compact_embedding_text(text, DRUG_SAFE_TEXT_MAX_CHARS)
    payload = {
        "model": model,
        "prompt": prompt_text,
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
    except urllib.error.HTTPError as e:
        # Preserve the Ollama response body.  Without this, all embedding
        # failures collapse to a generic HTTP 500, which is not actionable.
        body = e.read().decode("utf-8", errors="replace")
        body = body.replace("\n", " ").strip()[:1000]
        raise RuntimeError(f"Ollama embedding request failed: HTTP {e.code}: {body}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama embedding request failed: {e}")

    emb = obj.get("embedding")
    if not isinstance(emb, list) or not emb:
        raise RuntimeError(f"Invalid embedding response: {obj}")
    return [float(x) for x in emb]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ------------------------------------------------------------
# Drug RAG row matching
# ------------------------------------------------------------
def load_drug_rag_df(path: str = DEFAULT_DRUG_RAG_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["node_id", "selected_ChID", "ref_ChID"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    return df


def get_drug_rag_row_by_ids(
    drug_df: pd.DataFrame,
    node_id: Optional[str] = None,
    selected_chid: Optional[str] = None,
    selected_cid: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    df = drug_df

    if node_id and "node_id" in df.columns:
        hit = df[df["node_id"].astype(str).str.lower() == str(node_id).lower()]
        if len(hit) > 0:
            return hit.iloc[0].to_dict()

    if selected_chid and "selected_ChID" in df.columns:
        hit = df[df["selected_ChID"].astype(str).str.lower() == str(selected_chid).lower()]
        if len(hit) > 0:
            return hit.iloc[0].to_dict()

    if selected_cid is not None and "selected_CID" in df.columns:
        try:
            cid = float(selected_cid)
            hit = df[df["selected_CID"].astype(float) == cid]
            if len(hit) > 0:
                return hit.iloc[0].to_dict()
        except Exception:
            pass

    return None


def get_drug_node(driver, drug_name: str) -> Optional[Dict[str, Any]]:
    q = """
    MATCH (dr:DRUG)
    WHERE toLower(dr.name) = toLower($drug)
    RETURN
      elementId(dr) AS eid,
      dr.name AS name,
      dr.node_id AS node_id,
      dr.id AS id,
      dr.selected_ChID AS selected_ChID,
      dr.selected_CID AS selected_CID,
      dr.Group AS `Group`,
      dr.Drug_Type AS Drug_Type,
      dr.Max_Phase AS Max_Phase,
      dr.Black_Box AS Black_Box,
      dr.Withdrawn_Flag AS Withdrawn_Flag,
      dr.Inorganic_Flag AS Inorganic_Flag,
      dr.Molecular_Weight AS Molecular_Weight,
      dr.Polar_Surface_Area AS Polar_Surface_Area,
      coalesce(dr.CX_LogP, dr.XLogP, dr.CLogP, dr.AlogP) AS LogP,
      dr.QED_Weighted AS QED_Weighted,
      dr.Oral AS Oral,
      dr.Parenteral AS Parenteral,
      dr.Topical AS Topical,
      dr.Prodrug AS Prodrug,
      dr.Warning_Class_List AS Warning_Class_List,
      dr.safe_drug_text AS safe_drug_text,
      dr.safe_drug_embedding AS safe_drug_embedding
    LIMIT 1
    """
    rows = run_query(driver, q, {"drug": drug_name}, timeout_sec=30)
    return rows[0] if rows else None


def get_drug_action_summary(driver, drug_name: str, limit: int = 80) -> List[Dict[str, Any]]:
    rel_union = "|".join(DRUG_GENE_RELS)
    q = f"""
    MATCH (dr:DRUG)
    WHERE toLower(dr.name) = toLower($drug)
    MATCH (dr)-[r:{rel_union}]-(g:GENE)
    WHERE g.name IS NOT NULL
    RETURN DISTINCT type(r) AS rel, g.name AS gene
    ORDER BY rel, gene
    LIMIT $limit
    """
    return run_query(driver, q, {"drug": drug_name, "limit": limit}, timeout_sec=30)


# ------------------------------------------------------------
# Build SAFE drug text
# ------------------------------------------------------------
def build_safe_drug_text(
    drug_name: str,
    drug_node: Optional[Dict[str, Any]] = None,
    drug_rag_row: Optional[Dict[str, Any]] = None,
    action_rows: Optional[List[Dict[str, Any]]] = None,
    mask_terms: Optional[List[str]] = None,
    max_chars_per_field: int = 800,
    max_total_chars: int = DRUG_SAFE_TEXT_MAX_CHARS,
) -> str:
    """
    Drug embedding text. No indication/description/disease-treatment fields.
    """
    parts = []
    parts.append(f"Drug: {drug_name}")

    if drug_node:
        basic = []
        for k in ["Group", "Drug_Type", "Max_Phase"]:
            v = clean_text(drug_node.get(k))
            if v:
                basic.append(f"{k}={v}")
        if basic:
            parts.append("KG drug metadata: " + "; ".join(basic))

        props = []
        for k in [
            "Molecular_Weight", "Polar_Surface_Area", "LogP", "QED_Weighted",
            "Black_Box", "Withdrawn_Flag", "Inorganic_Flag",
            "Oral", "Parenteral", "Topical", "Prodrug",
            "Warning_Class_List",
        ]:
            v = clean_text(drug_node.get(k))
            if v:
                props.append(f"{k}={v}")
        if props:
            parts.append("Properties and risk flags: " + "; ".join(props))

    if drug_rag_row:
        # Category / ATC first. Use prefix-free canonical labels; legacy column
        # names are resolved in get_rag_field().
        for field in ["category", "atc_codes", "indication_class"]:
            values = parse_list_like_text(get_rag_field(drug_rag_row, field), max_items=60)
            if values:
                text = "; ".join(values)
                text = mask_terms_in_text(text, mask_terms or [], mask="[MASKED_DISEASE]")
                parts.append(f"{field}: {truncate_text(text, max_chars_per_field)}")

        # Mechanistic text
        for field in ["mechanism_of_action", "pharmacodynamics", "pathway"]:
            text = clean_text(get_rag_field(drug_rag_row, field))
            if text:
                text = mask_terms_in_text(text, mask_terms or [], mask="[MASKED_DISEASE]")
                parts.append(f"{field}: {truncate_text(text, max_chars_per_field)}")

        # PK support text
        for field in ["protein_binding", "half_life"]:
            text = clean_text(get_rag_field(drug_rag_row, field))
            if text:
                text = mask_terms_in_text(text, mask_terms or [], mask="[MASKED_DISEASE]")
                parts.append(f"{field}: {truncate_text(text, 500)}")

    # KG action-target summary
    if action_rows:
        grouped: Dict[str, List[str]] = {}
        for r in action_rows:
            rel = clean_text(r.get("rel"))
            gene = clean_text(r.get("gene"))
            if not rel or not gene:
                continue
            grouped.setdefault(rel, [])
            if gene not in grouped[rel]:
                grouped[rel].append(gene)

        rel_parts = []
        for rel, genes in grouped.items():
            rel_parts.append(f"{rel}: {', '.join(genes[:30])}")
        if rel_parts:
            parts.append("KG drug-gene actions: " + "; ".join(rel_parts))

    text = "\n".join(p for p in parts if clean_text(p))
    return compact_embedding_text(text, max_total_chars)


# ------------------------------------------------------------
# Store / load embeddings on DRUG nodes
# ------------------------------------------------------------
def get_existing_drug_embedding(driver, drug_name: str) -> Optional[Dict[str, Any]]:
    q = f"""
    MATCH (dr:DRUG)
    WHERE toLower(dr.name) = toLower($drug)
      AND dr.{DRUG_EMBEDDING_PROP} IS NOT NULL
    RETURN
      dr.name AS drug,
      dr.{DRUG_TEXT_PROP} AS safe_drug_text,
      dr.{DRUG_EMBEDDING_PROP} AS embedding,
      dr.{DRUG_EMBEDDING_MODEL_PROP} AS embedding_model
    LIMIT 1
    """
    rows = run_query(driver, q, {"drug": drug_name}, timeout_sec=30)
    return rows[0] if rows else None


def store_drug_embedding(
    driver,
    drug_name: str,
    safe_text: str,
    embedding: List[float],
    embedding_model: str = EMBED_MODEL,
    source: str = DEFAULT_DRUG_RAG_CSV,
) -> None:
    q = f"""
    MATCH (dr:DRUG)
    WHERE toLower(dr.name) = toLower($drug)
    SET dr.{DRUG_TEXT_PROP} = $text,
        dr.{DRUG_EMBEDDING_PROP} = $embedding,
        dr.{DRUG_EMBEDDING_MODEL_PROP} = $model,
        dr.{DRUG_EMBEDDING_UPDATED_AT_PROP} = datetime(),
        dr.{DRUG_EMBEDDING_SOURCE_PROP} = $source
    RETURN dr.name AS drug
    """
    rows = run_query(
        driver,
        q,
        {
            "drug": drug_name,
            "text": safe_text,
            "embedding": embedding,
            "model": embedding_model,
            "source": source,
        },
        timeout_sec=60,
    )
    if not rows:
        raise RuntimeError(f"Drug not found in KG: {drug_name}")


def ensure_drug_embedding(
    driver,
    drug_name: str,
    drug_df: pd.DataFrame,
    force_rebuild: bool = False,
    embedding_model: str = EMBED_MODEL,
    mask_terms: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if not force_rebuild:
        existing = get_existing_drug_embedding(driver, drug_name)
        if existing and existing.get("embedding"):
            return {
                "drug": existing.get("drug") or drug_name,
                "safe_drug_text": existing.get("safe_drug_text") or "",
                "embedding": existing.get("embedding"),
                "embedding_model": existing.get("embedding_model"),
                "created": False,
            }

    drug_node = get_drug_node(driver, drug_name)
    if not drug_node:
        raise RuntimeError(f"Drug not found in KG: {drug_name}")

    drug_rag_row = get_drug_rag_row_by_ids(
        drug_df,
        node_id=drug_node.get("node_id") or drug_node.get("id"),
        selected_chid=drug_node.get("selected_ChID"),
        selected_cid=drug_node.get("selected_CID"),
    )
    action_rows = get_drug_action_summary(driver, drug_name)

    safe_text = build_safe_drug_text(
        drug_name=drug_node.get("name") or drug_name,
        drug_node=drug_node,
        drug_rag_row=drug_rag_row,
        action_rows=action_rows,
        mask_terms=mask_terms or [],
    )

    if not safe_text or len(safe_text) < 20:
        raise RuntimeError(f"No safe drug text generated for: {drug_name}")

    emb = ollama_embed(safe_text, model=embedding_model)
    store_drug_embedding(
        driver,
        drug_name=drug_name,
        safe_text=safe_text,
        embedding=emb,
        embedding_model=embedding_model,
    )

    return {
        "drug": drug_node.get("name") or drug_name,
        "safe_drug_text": safe_text,
        "embedding": emb,
        "embedding_model": embedding_model,
        "created": True,
    }


# ------------------------------------------------------------
# Build query text from disease-context retrieval anchors
# ------------------------------------------------------------
def build_drug_similarity_query_text(
    disease: str,
    mechanism_anchor_obj: Optional[Dict[str, Any]] = None,
    disease_context: Optional[Dict[str, Any]] = None,
    max_items: int = 40,
    max_chars: int = DRUG_EMBED_QUERY_MAX_CHARS,
) -> str:
    """
    Compact disease-context text to compare with drug representation embeddings.

    Retrieval-anchor context can become verbose, so this query keeps only
    the highest-signal anchors and applies a final hard cap.
    """
    parts = [f"Disease: {disease}"]

    if mechanism_anchor_obj:
        axes = mechanism_anchor_obj.get("mechanism_axes", []) or []
        # Keep the most important mechanism axes only.
        for axis in axes[:4]:
            axis_name = clean_text(axis.get("axis_name"))
            rationale = clean_text(axis.get("axis_rationale"))
            genes = axis.get("anchor_genes", []) or []
            terms = axis.get("anchor_biology_terms", []) or []
            paths = axis.get("anchor_pathways", []) or []
            tissues = axis.get("relevant_cell_types_or_tissues", []) or []
            actions = axis.get("expected_drug_actions_or_classes", []) or []

            block = []
            if axis_name:
                block.append(f"Axis: {axis_name}")
            if rationale:
                block.append(f"Rationale: {truncate_text(rationale, 220)}")
            if genes:
                block.append("Genes: " + ", ".join([str(g) for g in genes[:12]]))
            if terms:
                block.append("Biology: " + "; ".join([str(t) for t in terms[:12]]))
            if paths:
                block.append("Pathways: " + "; ".join([str(p) for p in paths[:8]]))
            if tissues:
                block.append("Tissues/cells: " + "; ".join([str(t) for t in tissues[:8]]))
            if actions:
                block.append("Expected drug actions/classes: " + "; ".join([str(a) for a in actions[:8]]))
            if block:
                parts.append(" | ".join(block))

        ga = mechanism_anchor_obj.get("global_anchors", {}) or {}
        if ga.get("target_genes"):
            parts.append("Global genes: " + ", ".join([str(x) for x in ga.get("target_genes", [])[:max_items]]))
        if ga.get("biology_terms"):
            parts.append("Global biology: " + "; ".join([str(x) for x in ga.get("biology_terms", [])[:max_items]]))
        if ga.get("pathways"):
            parts.append("Global pathways: " + "; ".join([str(x) for x in ga.get("pathways", [])[:20]]))

    if disease_context:
        if disease_context.get("disease_genes"):
            parts.append("Disease-associated genes: " + ", ".join(disease_context.get("disease_genes", [])[:30]))
        if disease_context.get("phenotypes"):
            parts.append("Disease phenotypes: " + "; ".join(disease_context.get("phenotypes", [])[:20]))

    return compact_embedding_text("\n".join(parts), max_chars)

def get_query_embedding(text: str, embedding_model: str = EMBED_MODEL) -> List[float]:
    text = compact_embedding_text(text, DRUG_EMBED_QUERY_MAX_CHARS)
    return ollama_embed(text, model=embedding_model)


# ------------------------------------------------------------
# Rerank candidate drugs with drug embedding similarity
# ------------------------------------------------------------
def add_drug_embedding_scores_to_candidates(
    driver,
    candidates: List[Dict[str, Any]],
    drug_df: pd.DataFrame,
    query_text: str,
    embedding_model: str = EMBED_MODEL,
    force_rebuild: bool = False,
    candidate_drug_field: str = "drug",
    score_field: str = "drug_embedding_similarity",
    max_candidates: int = 300,
    sleep_sec: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Adds drug MoA similarity to each candidate row.
    It only builds embeddings for candidate drugs, not all drugs.
    """
    if not candidates:
        return []

    try:
        q_emb = get_query_embedding(query_text, embedding_model=embedding_model)
    except Exception as e:
        print(
            f"[DRUG EMBEDDING QUERY ERROR] {type(e).__name__}: {e}. "
            f"Skip drug embedding scores for this case.",
            flush=True,
        )
    
        for c in candidates:
            c["drug_embedding_similarity"] = 0.0
            c["drug_embedding_error"] = f"{type(e).__name__}: {e}"
    
        return candidates

    out = []
    for i, row in enumerate(candidates[:max_candidates], start=1):
        drug = row.get(candidate_drug_field)
        if not drug:
            continue
        new_row = dict(row)
        try:
            info = ensure_drug_embedding(
                driver=driver,
                drug_name=str(drug),
                drug_df=drug_df,
                force_rebuild=force_rebuild,
                embedding_model=embedding_model,
                mask_terms=[],
            )
            sim = cosine_similarity(q_emb, info.get("embedding") or [])
            new_row[score_field] = round(float(sim), 6)
            new_row["source_drug_embedding"] = True
            new_row["safe_drug_text_preview"] = (info.get("safe_drug_text") or "")[:500]
        except Exception as e:
            new_row[score_field] = 0.0
            new_row["source_drug_embedding"] = False
            new_row["drug_embedding_error"] = f"{type(e).__name__}: {e}"

        out.append(new_row)
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    # Preserve rows beyond max_candidates unchanged
    for row in candidates[max_candidates:]:
        out.append(dict(row))

    return out


def merge_candidates_with_drug_embedding_scores(
    candidates: List[Dict[str, Any]],
    base_score_field: str = "candidate_score",
    drug_embedding_score_field: str = "drug_embedding_similarity",
    drug_embedding_weight: float = 1.0,
    final_score_field: str = "final_score_with_drug_embedding",
) -> List[Dict[str, Any]]:
    """
    Final score = base score + weight * cosine similarity.
    Recommended initial weight: 0.5~1.0.
    """
    out = []
    for row in candidates:
        r = dict(row)
        base = float(r.get(base_score_field) or 0.0)
        sim = float(r.get(drug_embedding_score_field) or 0.0)
        r["base_score_before_drug_embedding"] = base
        r[final_score_field] = round(base + drug_embedding_weight * sim, 6)
        r["drug_embedding_weight"] = drug_embedding_weight
        out.append(r)

    out.sort(
        key=lambda x: (
            float(x.get(final_score_field) or 0.0),
            float(x.get(base_score_field) or 0.0),
        ),
        reverse=True,
    )

    for i, r in enumerate(out, start=1):
        r["rank"] = i
        r["candidate_score_original_before_drug_embedding"] = r.get(base_score_field)
        r[base_score_field] = r.get(final_score_field)

    return out


# Convenience wrapper for candidate drug representation reranking
#   mechanism_obj["candidate_drugs_before_drug_embedding"] = mechanism_obj["candidate_drugs"]
#   mechanism_obj["candidate_drugs"] = apply_drug_embedding_rerank_for_case(...)
def apply_drug_embedding_rerank_for_case(
    driver,
    disease: str,
    candidates: List[Dict[str, Any]],
    drug_df: pd.DataFrame,
    mechanism_anchor_obj: Optional[Dict[str, Any]] = None,
    disease_context: Optional[Dict[str, Any]] = None,
    embedding_model: str = EMBED_MODEL,
    force_rebuild: bool = False,
    drug_embedding_weight: float = 0.8,
    max_candidates: int = 300,
) -> List[Dict[str, Any]]:
    query_text = build_drug_similarity_query_text(
        disease=disease,
        mechanism_anchor_obj=mechanism_anchor_obj,
        disease_context=disease_context,
    )
    print(f"[DRUG EMBEDDING QUERY] disease={disease} chars={len(query_text)} model={embedding_model}", flush=True)

    scored = add_drug_embedding_scores_to_candidates(
        driver=driver,
        candidates=candidates,
        drug_df=drug_df,
        query_text=query_text,
        embedding_model=embedding_model,
        force_rebuild=force_rebuild,
        max_candidates=max_candidates,
    )

    ranked = merge_candidates_with_drug_embedding_scores(
        scored,
        base_score_field="candidate_score",
        drug_embedding_score_field="drug_embedding_similarity",
        drug_embedding_weight=drug_embedding_weight,
    )
    return ranked


# ------------------------------------------------------------
# Batch build CLI
# ------------------------------------------------------------
def get_drug_names_from_kg(driver, limit: Optional[int] = None) -> List[str]:
    q = """
    MATCH (dr:DRUG)
    WHERE dr.name IS NOT NULL
    RETURN DISTINCT dr.name AS drug
    ORDER BY drug
    """
    if limit:
        q += "\nLIMIT $limit"
    rows = run_query(driver, q, {"limit": int(limit)} if limit else {}, timeout_sec=120)
    return [r["drug"] for r in rows if r.get("drug")]


def read_drug_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_batch(
    driver,
    drug_df: pd.DataFrame,
    drugs: List[str],
    out_path: str,
    force_rebuild: bool = False,
    embedding_model: str = EMBED_MODEL,
    sleep_sec: float = 0.0,
) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if os.path.exists(out_path) and not force_rebuild:
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if obj.get("drug") and obj.get("ok"):
                        done.add(normalize_name(obj["drug"]))
                except Exception:
                    pass

    with open(out_path, "a", encoding="utf-8") as f:
        for i, drug in enumerate(drugs, start=1):
            if normalize_name(drug) in done and not force_rebuild:
                print(f"[{i}/{len(drugs)}] SKIP {drug}", flush=True)
                continue
            try:
                info = ensure_drug_embedding(
                    driver=driver,
                    drug_name=drug,
                    drug_df=drug_df,
                    force_rebuild=force_rebuild,
                    embedding_model=embedding_model,
                )
                rec = {
                    "drug": info.get("drug") or drug,
                    "ok": True,
                    "created": info.get("created"),
                    "text_chars": len(info.get("safe_drug_text") or ""),
                    "embedding_dim": len(info.get("embedding") or []),
                    "embedding_model": embedding_model,
                }
                print(f"[{i}/{len(drugs)}] OK {drug} created={rec['created']}", flush=True)
            except Exception as e:
                rec = {
                    "drug": drug,
                    "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                    "embedding_model": embedding_model,
                }
                print(f"[{i}/{len(drugs)}] ERROR {drug}: {rec['error']}", flush=True)

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if sleep_sec > 0:
                time.sleep(sleep_sec)


def parse_args():
    p = argparse.ArgumentParser(description="Build and use safe drug embeddings for GraphRAG reranking.")
    p.add_argument("--mode", choices=["build_one", "build_batch", "print_text"], default="build_one")
    p.add_argument("--drug", default=None)
    p.add_argument("--drug_rag_csv", default=DEFAULT_DRUG_RAG_CSV)
    p.add_argument("--drug_list", default=None)
    p.add_argument("--out", default="output/drug_embedding_build.jsonl")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force_rebuild", action="store_true")
    p.add_argument("--embedding_model", default=EMBED_MODEL)
    p.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", ""))
    p.add_argument("--sleep_sec", type=float, default=0.0)
    return p.parse_args()


def main():
    args = parse_args()
    driver = make_driver(args.neo4j_password)
    drug_df = load_drug_rag_df(args.drug_rag_csv)

    try:
        if args.mode in {"build_one", "print_text"}:
            if not args.drug:
                raise ValueError("--drug is required for build_one/print_text")

            drug_node = get_drug_node(driver, args.drug)
            if not drug_node:
                raise RuntimeError(f"Drug not found: {args.drug}")
            rag_row = get_drug_rag_row_by_ids(
                drug_df,
                node_id=drug_node.get("node_id") or drug_node.get("id"),
                selected_chid=drug_node.get("selected_ChID"),
                selected_cid=drug_node.get("selected_CID"),
            )
            actions = get_drug_action_summary(driver, args.drug)
            text = build_safe_drug_text(args.drug, drug_node, rag_row, actions)

            if args.mode == "print_text":
                print(text)
            else:
                info = ensure_drug_embedding(
                    driver=driver,
                    drug_name=args.drug,
                    drug_df=drug_df,
                    force_rebuild=args.force_rebuild,
                    embedding_model=args.embedding_model,
                )
                print(json.dumps({
                    "drug": info.get("drug"),
                    "created": info.get("created"),
                    "text_chars": len(info.get("safe_drug_text") or ""),
                    "embedding_dim": len(info.get("embedding") or []),
                    "embedding_model": info.get("embedding_model"),
                }, ensure_ascii=False, indent=2))

        elif args.mode == "build_batch":
            if args.drug_list:
                drugs = read_drug_list(args.drug_list)
            else:
                drugs = get_drug_names_from_kg(driver, limit=args.limit)
            if args.limit:
                drugs = drugs[: args.limit]
            build_batch(
                driver=driver,
                drug_df=drug_df,
                drugs=drugs,
                out_path=args.out,
                force_rebuild=args.force_rebuild,
                embedding_model=args.embedding_model,
                sleep_sec=args.sleep_sec,
            )
    finally:
        driver.close()


if __name__ == "__main__":
    main()
