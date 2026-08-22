from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .common import sha256_obj, slugify


PACKET_SCHEMA_VERSION = "grounded-evidence-v1"

RELATION_SEMANTICS: dict[str, str] = {
    "target": "Reported drug-to-gene target relation; it does not by itself establish disease benefit.",
    "inhibition": "Directional inhibitory drug-to-gene action. Therapeutic relevance still requires a supported disease-mechanism link.",
    "activation": "Directional activating drug-to-gene action. Therapeutic relevance still requires a supported disease-mechanism link.",
    "binding": "Drug-protein binding relation without an automatically implied functional direction.",
    "modulation": "Drug-protein modulation relation whose direction may be incompletely specified.",
    "enzyme": "Drug-enzyme relation, which may concern metabolism or disposition rather than therapeutic mechanism.",
    "transporter": "Drug-transporter relation, often relevant to disposition or exposure.",
    "carrier": "Drug-carrier relation, generally relevant to transport or disposition.",
    "ppi": "Nondirectional protein-protein interaction; it does not establish causal signal propagation.",
    "interacts_with": "General biological interaction; direction and causality must not be assumed.",
    "associated_with": "Association relation; association does not establish causality or therapeutic benefit.",
    "linked_to": "General linkage relation without an automatically implied causal effect.",
    "parent_child": "Ontology hierarchy relation supporting conceptual proximity, not mechanistic equivalence.",
    "phenotype_present": "Phenotype-presence annotation rather than direct therapeutic evidence.",
    "phenotype_absent": "Phenotype-absence annotation rather than direct therapeutic evidence.",
    "indication": "Known drug-disease indication relation.",
    "off_label_use": "Reported off-label use; it is not equivalent to confirmed efficacy.",
    "tested_indication": "Clinical testing relation; testing does not imply efficacy or approval.",
    "studied_for_treatment_of": "Reported treatment-study relation; study presence does not establish benefit.",
    "studied_for_marker_mechanism_of": "Marker/mechanism study relation, not direct evidence of efficacy.",
    "contraindication": "Potential safety counter-evidence relevant to interpretation.",
    "side_effect": "Drug adverse-effect relation, not a therapeutic mechanism.",
    "synergistic_interaction": "Drug-drug synergy relation that does not establish single-agent efficacy.",
}

DISEASE_RAG_FIELDS = (
    "mondo_definition",
    "umls_description",
    "orphanet_definition",
    "orphanet_prevalence",
    "orphanet_epidemiology",
    "orphanet_clinical_description",
    "orphanet_management_and_treatment",
    "mayo_symptoms",
    "mayo_causes",
    "mayo_risk_factors",
    "mayo_complications",
    "mayo_prevention",
    "mayo_see_doc",
)

DRUG_RAG_FIELDS = (
    "C_ATC Codes",
    "C_Indication Class",
    "K_category",
    "C_USAN Definition",
    "K_indication",
    "K_mechanism_of_action",
    "K_pharmacodynamics",
    "K_pathway",
    "K_description",
    "K_protein_binding",
    "K_half_life",
    "SMILES",
)

PHYSICOCHEMICAL_PROPERTIES = (
    "CLogP", "XLogP", "AlogP", "CX_LogP", "CX_LogD", "CX_Acidic_pKa", "CX_Basic_pKa",
    "RO5_Violations_Lipinski", "Aromatic_Rings", "Molecular_Species", "Np_Likeness_Score",
    "QED_Weighted", "Complexity", "Charge", "Molecular_Weight", "Polar_Surface_Area",
    "Heavy_Atoms", "H_Bond_Acceptor", "H_Bond_Donor", "Rotatable_Bonds",
)

PK_EXPOSURE_PROPERTIES = (
    "Oral", "Parenteral", "Topical", "Half_Life_h", "Protein_Binding_pct",
)

SAFETY_PROPERTIES = (
    "Warning_Class_List", "Black_Box", "Withdrawn_Flag", "Inorganic_Flag",
)

DEVELOPMENT_PROPERTIES = (
    "Group", "Max_Phase", "First_Approval", "Drug_Type", "Orphan", "Prodrug",
    "First_In_Class", "Targets", "Bioactivities", "Linked_PubChem_Literature_Count",
    "Linked_PubChem_Patent_Family_Count",
)

IDENTIFIER_PROPERTIES = (
    "name", "node_id", "selected_CID", "selected_ChID", "InChIKey",
)

EXCLUDED_NODE_PROPERTY_PATTERNS = (
    "embedding", "vector", "signature", "mask_terms", "raw_prompt", "token_ids",
)


class EvidenceBuildError(RuntimeError):
    pass


class NodeResolutionError(EvidenceBuildError):
    pass


class NodeNotFoundError(NodeResolutionError):
    pass


class NodeAmbiguousError(NodeResolutionError):
    pass


def normalize_name(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip().lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null", "[]", "{}"}


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if hasattr(value, "iso_format"):
        try:
            return value.iso_format()
        except Exception:
            pass
    if hasattr(value, "to_native"):
        try:
            return jsonable(value.to_native())
        except Exception:
            pass
    return str(value)


def trim_text(value: Any, max_chars: int) -> str:
    text = str(value).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 32].rstrip() + " ... [TRUNCATED]"


def unique_values(series: Iterable[Any], max_values: int = 6, max_chars_each: int = 5000) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in series:
        if is_missing(value):
            continue
        text = trim_text(value, max_chars_each)
        key = normalize_name(text)
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= max_values:
            break
    return out


def parse_json_list(value: Any, field_name: str, row_no: int) -> list[Any]:
    if isinstance(value, list):
        return value
    if is_missing(value):
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise EvidenceBuildError(f"row {row_no}: invalid JSON in {field_name}: {exc}") from exc
    if not isinstance(parsed, list):
        raise EvidenceBuildError(f"row {row_no}: {field_name} must decode to a list")
    return parsed


def path_text(nodes: list[Any], rels: list[Any]) -> str:
    if not nodes:
        return ""
    chunks = [str(nodes[0])]
    for i, rel in enumerate(rels):
        nxt = nodes[i + 1] if i + 1 < len(nodes) else "?"
        chunks.append(f"-[{rel}]-> {nxt}")
    return " ".join(chunks)


@dataclass
class EvidenceCatalog:
    items: list[dict[str, Any]] = field(default_factory=list)
    _used: set[str] = field(default_factory=set)

    def add(self, evidence_id: str, category: str, source: str, text: str, **payload: Any) -> str:
        if evidence_id in self._used:
            raise EvidenceBuildError(f"Duplicate evidence_id generated: {evidence_id}")
        self._used.add(evidence_id)
        item = {
            "evidence_id": evidence_id,
            "category": category,
            "source": source,
            "text": trim_text(text, 8000),
        }
        item.update({k: jsonable(v) for k, v in payload.items() if v is not None})
        self.items.append(item)
        return evidence_id


class RagRepository:
    def __init__(self, rag_dir: str | Path):
        self.rag_dir = Path(rag_dir)
        required = {
            "disease_features": "disease_features.csv",
            "drug_indication": "Drug_Indication_All_RAG.csv",
            "drug_data": "Drug_data_All_RAG.csv",
            "drug_mechanism": "Drug_Mechanism_All_RAG.csv",
        }
        missing = [name for name in required.values() if not (self.rag_dir / name).exists()]
        if missing:
            raise EvidenceBuildError(
                f"RAG directory is missing required files: {missing}. rag_dir={self.rag_dir}"
            )
        self.disease_features = pd.read_csv(self.rag_dir / required["disease_features"], low_memory=False)
        self.drug_indication = pd.read_csv(self.rag_dir / required["drug_indication"], low_memory=False)
        self.drug_data = pd.read_csv(self.rag_dir / required["drug_data"], low_memory=False)
        self.drug_mechanism = pd.read_csv(self.rag_dir / required["drug_mechanism"], low_memory=False)

        self._validate_columns()
        for frame in (self.disease_features, self.drug_indication, self.drug_data, self.drug_mechanism):
            for col in frame.columns:
                if col.endswith("node_index") or col == "node_index":
                    frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("Int64")

    def _validate_columns(self) -> None:
        required_by_table = {
            "disease_features": {"node_index", "mondo_name", "group_name_bert"},
            "drug_data": {"node_index", "node_id", "Merged_Synonyms"},
            "drug_mechanism": {"drug_node_index", "gene_node_index", "edge_relation"},
            "drug_indication": {"disease_node_index", "drug_node_index", "edge_relation"},
        }
        for name, required in required_by_table.items():
            frame = getattr(self, name)
            missing = required - set(frame.columns)
            if missing:
                raise EvidenceBuildError(f"{name} missing required columns: {sorted(missing)}")

    def disease_rows(self, disease: str, internal_id: int | None) -> tuple[pd.DataFrame, str]:
        df = self.disease_features
        target = normalize_name(disease)
        if internal_id is not None:
            matched = df[df["node_index"] == int(internal_id)]
            if not matched.empty:
                return matched.copy(), "neo4j_internal_id"
        for col in ("group_name_bert", "mondo_name"):
            norm = df[col].fillna("").map(normalize_name)
            matched = df[norm == target]
            if not matched.empty:
                return matched.copy(), f"exact_{col}"
        return df.iloc[0:0].copy(), "unmatched"

    def drug_rows(self, drug: str, node: dict[str, Any]) -> tuple[pd.DataFrame, str]:
        df = self.drug_data
        internal_id = node.get("internal_id")
        props = node.get("properties") or {}
        if internal_id is not None:
            matched = df[df["node_index"] == int(internal_id)]
            if not matched.empty:
                return matched.copy(), "neo4j_internal_id"

        identifiers = (
            ("node_id", props.get("node_id")),
            ("selected_CID", props.get("selected_CID")),
            ("selected_ChID", props.get("selected_ChID")),
        )
        for col, value in identifiers:
            if value is None or col not in df.columns:
                continue
            matched = df[df[col].astype(str).str.strip() == str(value).strip()]
            if not matched.empty:
                return matched.copy(), f"neo4j_property_{col}"

        target = normalize_name(drug)
        exact_mask = df["Merged_Synonyms"].fillna("").map(
            lambda x: target in {normalize_name(token) for token in str(x).split("|") if token.strip()}
        )
        matched = df[exact_mask]
        if not matched.empty:
            return matched.copy(), "exact_merged_synonym"
        return df.iloc[0:0].copy(), "unmatched"

    def mechanism_rows(self, drug_indices: list[int], gene_indices: list[int], path_relations: list[str]) -> pd.DataFrame:
        df = self.drug_mechanism
        if not drug_indices:
            return df.iloc[0:0].copy()
        matched = df[df["drug_node_index"].isin(drug_indices)]
        if matched.empty:
            return matched.copy()
        if gene_indices:
            gene_specific = matched[matched["gene_node_index"].isin(gene_indices)]
            if not gene_specific.empty:
                matched = gene_specific
        action_rels = {str(x) for x in path_relations if str(x) in {"inhibition", "activation", "modulation", "binding"}}
        if action_rels:
            rel_specific = matched[matched["edge_relation"].astype(str).isin(action_rels)]
            if not rel_specific.empty:
                matched = rel_specific
        return matched.head(20).copy()

    def indication_rows(self, disease_indices: list[int], drug_indices: list[int]) -> pd.DataFrame:
        df = self.drug_indication
        if not disease_indices or not drug_indices:
            return df.iloc[0:0].copy()
        return df[
            df["disease_node_index"].isin(disease_indices)
            & df["drug_node_index"].isin(drug_indices)
        ].head(20).copy()


class Neo4jRepository:
    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ):
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise EvidenceBuildError("neo4j Python package is required. Run: pip install -r requirements.txt") from exc
        self.uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.environ.get("NEO4J_USER", "neo4j")
        self.password = password or os.environ.get("NEO4J_PASSWORD")
        self.database = database or os.environ.get("NEO4J_DATABASE", "neo4j")
        if not self.password:
            raise EvidenceBuildError("NEO4J_PASSWORD is not set")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self._node_cache: dict[tuple[str, str], dict[str, Any]] = {}

    def close(self) -> None:
        self.driver.close()

    def verify(self) -> None:
        self.driver.verify_connectivity()

    def _run(self, query: str, **params: Any) -> list[dict[str, Any]]:
        with self.driver.session(database=self.database) as session:
            return [dict(record) for record in session.run(query, **params)]

    def find_unique_node(self, name: str, label: str) -> dict[str, Any]:
        cache_key = (label, normalize_name(name))
        if cache_key in self._node_cache:
            return self._node_cache[cache_key]
        exact = self._run(
            """
            MATCH (n)
            WHERE $label IN labels(n)
              AND n.name IS NOT NULL
              AND toLower(trim(n.name)) = toLower(trim($name))
            RETURN n.name AS name, labels(n) AS labels, id(n) AS internal_id,
                   elementId(n) AS element_id, properties(n) AS properties
            ORDER BY id(n)
            """,
            label=label,
            name=name,
        )
        records = exact
        match_method = "case_insensitive_exact"
        if not records:
            candidates = self._run(
                """
                MATCH (n)
                WHERE $label IN labels(n)
                  AND n.name IS NOT NULL
                  AND toLower(n.name) CONTAINS toLower($name)
                RETURN n.name AS name, labels(n) AS labels, id(n) AS internal_id,
                       elementId(n) AS element_id, properties(n) AS properties
                ORDER BY id(n)
                LIMIT 50
                """,
                label=label,
                name=name,
            )
            records = [x for x in candidates if normalize_name(x.get("name")) == normalize_name(name)]
            match_method = "unique_normalized_exact"
        if not records:
            raise NodeNotFoundError(f"No {label} node found for '{name}'")
        if len(records) != 1:
            names = [x.get("name") for x in records]
            raise NodeAmbiguousError(
                f"Expected one {label} node for '{name}', found {len(records)}. matches={names}"
            )
        node = jsonable(records[0])
        node["match_method"] = match_method
        self._node_cache[cache_key] = node
        return node

    def path_segment_relations(self, left: dict[str, Any], right: dict[str, Any], relation: str) -> list[dict[str, Any]]:
        return [jsonable(x) for x in self._run(
            """
            MATCH (stored_start)-[r]->(stored_end)
            WHERE ((elementId(stored_start) = $left_id AND elementId(stored_end) = $right_id)
                OR (elementId(stored_start) = $right_id AND elementId(stored_end) = $left_id))
              AND type(r) = $relation
            RETURN type(r) AS relation,
                   stored_start.name AS stored_start,
                   stored_end.name AS stored_end,
                   properties(r) AS properties
            ORDER BY elementId(r)
            LIMIT 10
            """,
            left_id=left["element_id"],
            right_id=right["element_id"],
            relation=relation,
        )]

    def direct_drug_disease_relations(self, disease_node: dict[str, Any], drug_node: dict[str, Any]) -> list[dict[str, Any]]:
        return [jsonable(x) for x in self._run(
            """
            MATCH (stored_start)-[r]->(stored_end)
            WHERE ((elementId(stored_start) = $disease_id AND elementId(stored_end) = $drug_id)
                OR (elementId(stored_start) = $drug_id AND elementId(stored_end) = $disease_id))
            RETURN type(r) AS relation,
                   stored_start.name AS stored_start,
                   stored_end.name AS stored_end,
                   properties(r) AS properties
            ORDER BY type(r)
            """,
            disease_id=disease_node["element_id"],
            drug_id=drug_node["element_id"],
        )]

    def drug_side_effects(self, drug_node: dict[str, Any], limit: int = 20) -> list[dict[str, Any]]:
        return [jsonable(x) for x in self._run(
            """
            MATCH (dr)-[r:side_effect]-(p:PHENO)
            WHERE elementId(dr) = $drug_id
            RETURN p.name AS phenotype, properties(r) AS properties
            ORDER BY toLower(p.name)
            LIMIT $limit
            """,
            drug_id=drug_node["element_id"],
            limit=int(limit),
        )]


def cleaned_node_properties(properties: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in (properties or {}).items():
        key_lower = str(key).lower()
        if any(pattern in key_lower for pattern in EXCLUDED_NODE_PROPERTY_PATTERNS):
            continue
        if is_missing(value):
            continue
        value_json = jsonable(value)
        if isinstance(value_json, str):
            value_json = trim_text(value_json, 12000 if key == "safe_context_text" else 4000)
        out[str(key)] = value_json
    return out


def select_properties(properties: dict[str, Any], names: Iterable[str]) -> dict[str, Any]:
    return {name: properties[name] for name in names if name in properties and not is_missing(properties[name])}


def row_dict(row: pd.Series, columns: Iterable[str], max_chars: int = 5000) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in columns:
        if col not in row.index or is_missing(row[col]):
            continue
        value = jsonable(row[col])
        if isinstance(value, str):
            value = trim_text(value, max_chars)
        out[col] = value
    return out


class CandidateEvidenceBuilder:
    def __init__(self, rag: RagRepository, neo4j: Neo4jRepository):
        self.rag = rag
        self.neo4j = neo4j

    def build(self, row: pd.Series, row_no: int) -> dict[str, Any]:
        disease = str(row["disease"]).strip()
        drug = str(row["drug"]).strip()
        rank = int(row["rank"])
        nodes = parse_json_list(row.get("best_path_nodes"), "best_path_nodes", row_no)
        rels = [str(x) for x in parse_json_list(row.get("best_path_rels"), "best_path_rels", row_no)]
        if len(nodes) != len(rels) + 1:
            raise EvidenceBuildError(
                f"{disease} / {drug}: path nodes ({len(nodes)}) and relations ({len(rels)}) are inconsistent"
            )
        case_id = f"{slugify(disease)}__r{rank:02d}__{slugify(drug)}"
        catalog = EvidenceCatalog()
        sections: dict[str, list[dict[str, Any]]] = {
            "graphdrx": [],
            "relation_semantics": [],
            "disease_context": [],
            "drug_pharmacology": [],
            "physicochemical_properties": [],
            "pk_pd_exposure": [],
            "safety_developability": [],
            "path_relation_evidence": [],
            "drug_gene_mechanism": [],
            "drug_disease_development": [],
            "counter_evidence": [],
        }

        rank_text = f"GraphDRx rank={rank}; retrieval_score={float(row['retrieval_score']):.6f}."
        catalog.add("GRANK001", "graphdrx_rank", "GraphDRx fixed top-5 result", rank_text)
        sections["graphdrx"].append({
            "evidence_id": "GRANK001",
            "rank": rank,
            "retrieval_score": float(row["retrieval_score"]),
        })
        ptext = path_text(nodes, rels)
        catalog.add(
            "KGPATH001",
            "kg_path",
            "GraphDRx best path",
            ptext,
            pattern=str(row.get("best_path_pattern", "")),
            nodes=nodes,
            relations=rels,
        )
        sections["graphdrx"].append({
            "evidence_id": "KGPATH001",
            "pattern": str(row.get("best_path_pattern", "")).strip(),
            "nodes": nodes,
            "relations": rels,
            "plain_text": ptext,
        })
        for i, rel in enumerate(dict.fromkeys(rels), 1):
            eid = f"RELSEM{i:03d}"
            text = RELATION_SEMANTICS.get(
                rel,
                f"KG relation '{rel}'; its causal and therapeutic meaning must not be assumed beyond the supplied path.",
            )
            catalog.add(eid, "relation_semantics", "GraphDRx relation dictionary", text, relation=rel)
            sections["relation_semantics"].append({"evidence_id": eid, "relation": rel, "interpretation": text})

        disease_node = self.neo4j.find_unique_node(disease, "DISEASE")
        drug_node = self.neo4j.find_unique_node(drug, "DRUG")
        disease_props = cleaned_node_properties(disease_node.get("properties") or {})
        drug_props = cleaned_node_properties(drug_node.get("properties") or {})

        disease_rows, disease_match_method = self.rag.disease_rows(disease, disease_node.get("internal_id"))
        drug_rows, drug_match_method = self.rag.drug_rows(drug, drug_node)
        disease_indices = sorted({int(x) for x in disease_rows["node_index"].dropna().tolist()}) if not disease_rows.empty else []
        drug_indices = sorted({int(x) for x in drug_rows["node_index"].dropna().tolist()}) if not drug_rows.empty else []

        safe_context = disease_props.get("safe_context_text")
        if safe_context:
            eid = "DCTX001"
            catalog.add(
                eid,
                "disease_context",
                "Neo4j DISEASE.safe_context_text",
                str(safe_context),
                safe_context_source=disease_props.get("safe_context_source"),
                safe_context_model=disease_props.get("safe_context_model"),
            )
            sections["disease_context"].append({
                "evidence_id": eid,
                "field": "safe_context_text",
                "value": safe_context,
                "source": disease_props.get("safe_context_source"),
            })
        disease_counter = 1 + len(sections["disease_context"])
        if not disease_rows.empty:
            for field_name in DISEASE_RAG_FIELDS:
                if field_name not in disease_rows.columns:
                    continue
                values = unique_values(disease_rows[field_name], max_values=2, max_chars_each=3500)
                for value in values:
                    eid = f"DCTX{disease_counter:03d}"
                    disease_counter += 1
                    catalog.add(eid, "disease_context", f"disease_features.csv:{field_name}", value)
                    sections["disease_context"].append({
                        "evidence_id": eid,
                        "field": field_name,
                        "value": value,
                    })

        drug_counter = 1
        if not drug_rows.empty:
            for field_name in DRUG_RAG_FIELDS:
                if field_name not in drug_rows.columns:
                    continue
                values = unique_values(drug_rows[field_name], max_values=2, max_chars_each=5000)
                for value in values:
                    eid = f"DRUGRAG{drug_counter:03d}"
                    drug_counter += 1
                    catalog.add(eid, "drug_pharmacology", f"Drug_data_All_RAG.csv:{field_name}", value)
                    sections["drug_pharmacology"].append({
                        "evidence_id": eid,
                        "field": field_name,
                        "value": value,
                    })

        property_groups = (
            ("physicochemical_properties", "DRUGPROP_PC001", "physicochemical", PHYSICOCHEMICAL_PROPERTIES),
            ("pk_pd_exposure", "DRUGPROP_PK001", "pk_exposure", PK_EXPOSURE_PROPERTIES),
            ("safety_developability", "DRUGPROP_SAFE001", "safety", SAFETY_PROPERTIES),
            ("safety_developability", "DRUGPROP_DEV001", "development", DEVELOPMENT_PROPERTIES),
        )
        for section_name, eid, category, property_names in property_groups:
            selected = select_properties(drug_props, property_names)
            if selected:
                catalog.add(
                    eid,
                    category,
                    "Neo4j DRUG node properties",
                    json.dumps(selected, ensure_ascii=False, sort_keys=True),
                    properties=selected,
                )
                sections[section_name].append({"evidence_id": eid, "properties": selected})

        # Preserve identifiers for reproducible entity resolution, but do not present them as efficacy evidence.
        identifiers = select_properties(drug_props, IDENTIFIER_PROPERTIES)

        path_nodes_resolved: list[dict[str, Any]] = []
        gene_indices: list[int] = []
        for node_name in nodes:
            if normalize_name(node_name) == normalize_name(drug):
                node = drug_node
            elif normalize_name(node_name) == normalize_name(disease):
                node = disease_node
            else:
                # GraphDRx path may contain genes, biological processes, pathways, phenotypes, or diseases.
                node = None
                for label in ("GENE", "BP", "PATH", "MF", "CC", "PHENO", "DISEASE", "ANAT", "EXPO", "DRUG"):
                    try:
                        node = self.neo4j.find_unique_node(str(node_name), label)
                        break
                    except NodeNotFoundError:
                        continue
                if node is None:
                    raise EvidenceBuildError(f"Could not resolve path node '{node_name}' for {case_id}")
            path_nodes_resolved.append(node)
            if "GENE" in (node.get("labels") or []) and node.get("internal_id") is not None:
                gene_indices.append(int(node["internal_id"]))

        for i, rel in enumerate(rels, 1):
            left = path_nodes_resolved[i - 1]
            right = path_nodes_resolved[i]
            records = self.neo4j.path_segment_relations(left, right, rel)
            if not records:
                # The fixed GraphDRx output is still evidence, but a failed live-KG verification is material.
                continue
            for j, record in enumerate(records, 1):
                eid = f"KGEDGE{i:02d}_{j:02d}"
                text = (
                    f"Stored KG relation {record.get('stored_start')} -[{record.get('relation')}]-> "
                    f"{record.get('stored_end')}; properties={record.get('properties') or {}}"
                )
                catalog.add(eid, "kg_relation", "Neo4j live path verification", text, **record)
                sections["path_relation_evidence"].append({"evidence_id": eid, **record})

        mechanism_rows = self.rag.mechanism_rows(drug_indices, sorted(set(gene_indices)), rels)
        for i, (_, mechanism_row) in enumerate(mechanism_rows.iterrows(), 1):
            payload = row_dict(
                mechanism_row,
                (
                    "drug_node_index", "gene_node_index", "edge_relation", "Mechanism Comment",
                    "Selectivity Comment", "Binding Site Comment", "References",
                ),
                max_chars=5000,
            )
            eid = f"MECHRAG{i:03d}"
            catalog.add(
                eid,
                "drug_gene_mechanism",
                "Drug_Mechanism_All_RAG.csv",
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                data=payload,
            )
            sections["drug_gene_mechanism"].append({"evidence_id": eid, **payload})

        indication_rows = self.rag.indication_rows(disease_indices, drug_indices)
        for i, (_, indication_row) in enumerate(indication_rows.iterrows(), 1):
            payload = row_dict(
                indication_row,
                (
                    "disease_node_index", "drug_node_index", "edge_relation", "Max_Phase",
                    "MESH ID", "MESH Heading", "EFO IDs", "EFO Terms", "References",
                ),
                max_chars=5000,
            )
            eid = f"INDRAG{i:03d}"
            catalog.add(
                eid,
                "drug_disease_development",
                "Drug_Indication_All_RAG.csv",
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                data=payload,
            )
            sections["drug_disease_development"].append({"evidence_id": eid, **payload})

        direct_relations = self.neo4j.direct_drug_disease_relations(disease_node, drug_node)
        direct_counter = 1
        direct_dev_counter = len(sections["drug_disease_development"]) + 1
        for relation in direct_relations:
            rel_name = str(relation.get("relation"))
            if rel_name == "contraindication":
                eid = f"COUNTER{direct_counter:03d}"
                direct_counter += 1
                section_name = "counter_evidence"
                category = "counter_evidence"
            else:
                eid = f"KGDEV{direct_dev_counter:03d}"
                direct_dev_counter += 1
                section_name = "drug_disease_development"
                category = "drug_disease_relation"
            text = (
                f"Stored direct drug-disease relation {relation.get('stored_start')} -[{rel_name}]-> "
                f"{relation.get('stored_end')}; properties={relation.get('properties') or {}}"
            )
            catalog.add(eid, category, "Neo4j direct drug-disease relation", text, **relation)
            sections[section_name].append({"evidence_id": eid, **relation})

        for i, side_effect in enumerate(self.neo4j.drug_side_effects(drug_node), direct_counter):
            eid = f"COUNTER{i:03d}"
            text = f"Drug side-effect phenotype: {side_effect.get('phenotype')}; properties={side_effect.get('properties') or {}}"
            catalog.add(eid, "counter_evidence", "Neo4j DRUG-side_effect-PHENO", text, **side_effect)
            sections["counter_evidence"].append({"evidence_id": eid, **side_effect})

        verified_segments = len({item["evidence_id"].split("_")[0] for item in sections["path_relation_evidence"]})
        disease_context_available = bool(sections["disease_context"])
        drug_pharmacology_available = bool(sections["drug_pharmacology"] or sections["drug_gene_mechanism"])
        property_available = bool(
            sections["physicochemical_properties"]
            or sections["pk_pd_exposure"]
            or sections["safety_developability"]
        )
        core_ready = bool(
            disease_context_available
            and drug_pharmacology_available
            and sections["graphdrx"]
            and disease_node
            and drug_node
        )
        coverage = {
            "core_ready": core_ready,
            "disease_node_matched": True,
            "drug_node_matched": True,
            "disease_rag_matched": not disease_rows.empty,
            "drug_rag_matched": not drug_rows.empty,
            "disease_context_available": disease_context_available,
            "drug_pharmacology_available": drug_pharmacology_available,
            "physicochemical_available": bool(sections["physicochemical_properties"]),
            "pk_exposure_available": bool(sections["pk_pd_exposure"]),
            "safety_available": bool([x for x in sections["safety_developability"] if x["evidence_id"].startswith("DRUGPROP_SAFE")]),
            "development_properties_available": bool([x for x in sections["safety_developability"] if x["evidence_id"].startswith("DRUGPROP_DEV")]),
            "drug_gene_mechanism_rows": len(sections["drug_gene_mechanism"]),
            "drug_disease_development_rows": len(sections["drug_disease_development"]),
            "counter_evidence_rows": len(sections["counter_evidence"]),
            "path_segments_expected": len(rels),
            "path_segment_evidence_rows": len(sections["path_relation_evidence"]),
            "any_drug_property_available": property_available,
        }

        packet = {
            "packet_schema_version": PACKET_SCHEMA_VERSION,
            "case_id": case_id,
            "disease": disease,
            "source_disease": str(row.get("source_disease", disease)).strip(),
            "therapeutic_area": str(row["therapeutic_area"]).strip(),
            "drug": drug,
            "rank": rank,
            "retrieval_score": float(row["retrieval_score"]),
            "graphdrx": {
                "rank": rank,
                "retrieval_score": float(row["retrieval_score"]),
                "best_path_pattern": str(row.get("best_path_pattern", "")).strip(),
                "best_path_nodes": nodes,
                "best_path_relations": rels,
                "best_path_text": ptext,
            },
            "evidence_sections": sections,
            "evidence_catalog": catalog.items,
            "entity_resolution": {
                "disease": {
                    "input_name": disease,
                    "neo4j_name": disease_node.get("name"),
                    "neo4j_match_method": disease_node.get("match_method"),
                    "rag_match_method": disease_match_method,
                },
                "drug": {
                    "input_name": drug,
                    "neo4j_name": drug_node.get("name"),
                    "neo4j_match_method": drug_node.get("match_method"),
                    "rag_match_method": drug_match_method,
                    "identifiers": identifiers,
                },
            },
            "evidence_coverage": coverage,
            "metadata": {
                "input_source": "selected_top5.csv",
                "graphdrx_was_rerun": False,
                "fixed_configuration_selection_set": True,
                "external_model_knowledge_allowed": False,
            },
        }
        packet["packet_hash"] = sha256_obj({k: v for k, v in packet.items() if k != "packet_hash"})
        return packet
