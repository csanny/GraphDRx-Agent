from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .judge import CRITICAL_FLAGS, DIMENSIONS, DIRECT_DEVELOPMENT_RELATIONS, ROLE_SCORE_KEYS


SCORE_COLUMNS = [*DIMENSIONS, "primary_score", *ROLE_SCORE_KEYS]
DIRECT_RELATION_STRATUM_COLUMN = "direct_development_relation_stratum"


def flatten_judgments(judgments: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for batch in judgments:
        if not batch.get("judge_validation_pass", False):
            raise ValueError(f"Unvalidated judge batch found: {batch.get('case_id')}")
        config = batch["configuration"]
        evaluations = batch["judgment"].get("candidate_evaluations", [])
        for item in evaluations:
            row = {"configuration": config, "case_id": item.get("case_id")}
            scores = item.get("scores") or {}
            for dimension in DIMENSIONS:
                row[dimension] = scores.get(dimension)
            for key in ROLE_SCORE_KEYS:
                row[key] = item.get(key)
            row["critical_flags"] = "|".join(item.get("critical_flags") or [])
            row["strengths"] = " | ".join(item.get("strengths") or [])
            row["weaknesses"] = " | ".join(item.get("weaknesses") or [])
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    score_columns = [*DIMENSIONS, *ROLE_SCORE_KEYS]
    for column in score_columns:
        if column not in df.columns:
            raise ValueError(f"Missing judge score column: {column}")
        df[column] = pd.to_numeric(df[column], errors="raise")
        invalid = ~df[column].isin([1, 2, 3, 4, 5])
        if invalid.any():
            raise ValueError(
                f"Judge score outside 1-5 in {column}: "
                + repr(df.loc[invalid, ["configuration", "case_id", column]].to_dict("records"))
            )
    duplicates = df.duplicated(["configuration", "case_id"], keep=False)
    if duplicates.any():
        raise ValueError(
            "Duplicate configuration/case_id judge rows: "
            + repr(df.loc[duplicates, ["configuration", "case_id"]].to_dict("records"))
        )
    df["primary_score"] = df[list(DIMENSIONS)].mean(axis=1)
    return df


def _normalized_relation(record: dict[str, Any]) -> str:
    relation = record.get("relation") or record.get("edge_relation") or ""
    return str(relation).strip().lower().replace("-", "_").replace(" ", "_")


def build_case_metadata(packets: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Build prespecified candidate metadata for direct-development-evidence sensitivity analysis.

    A direct relation is present when the candidate packet's drug_disease_development
    section contains indication, off-label-use, tested-indication, or treatment-study
    evidence for that fixed drug-disease pair. Multiple KG/RAG records for the same
    relation are counted as duplicate provenance records, not independent facts.
    """
    rows: list[dict[str, Any]] = []
    allowed = set(DIRECT_DEVELOPMENT_RELATIONS)
    for packet in packets:
        sections = packet.get("evidence_sections") or {}
        development_records = [
            item
            for item in sections.get("drug_disease_development", [])
            if isinstance(item, dict)
        ]
        direct_records = [item for item in development_records if _normalized_relation(item) in allowed]
        relation_types = sorted({_normalized_relation(item) for item in direct_records})
        source_prefixes = sorted(
            {
                str(item.get("evidence_id", "")).split("0", 1)[0]
                for item in direct_records
                if item.get("evidence_id")
            }
        )
        n_evidence_records = sum(
            len(items) for items in sections.values() if isinstance(items, list)
        )
        rows.append(
            {
                "case_id": str(packet["case_id"]),
                "disease": str(packet.get("disease", "")),
                "drug": str(packet.get("drug", "")),
                "rank": int(packet.get("rank", 0)),
                "therapeutic_area": str(packet.get("therapeutic_area", "")),
                "direct_development_relation_present": bool(relation_types),
                DIRECT_RELATION_STRATUM_COLUMN: (
                    "direct_relation_present" if relation_types else "direct_relation_absent"
                ),
                "direct_development_relation_types": "|".join(relation_types),
                "n_direct_development_records": int(len(direct_records)),
                "n_unique_direct_relation_types": int(len(relation_types)),
                "n_duplicate_provenance_records": int(max(0, len(direct_records) - len(relation_types))),
                "direct_development_source_prefixes": "|".join(source_prefixes),
                "n_total_evidence_records": int(n_evidence_records),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if df["case_id"].duplicated().any():
        raise ValueError("Duplicate case_id in grounded input packets")
    return df.sort_values(["disease", "rank", "case_id"]).reset_index(drop=True)


def cluster_bootstrap_ci(
    df: pd.DataFrame,
    value_col: str,
    n_resamples: int = 20000,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    results: list[dict[str, Any]] = []
    for config, group in df.groupby("configuration"):
        disease_values = group.groupby("disease")[value_col].mean().dropna()
        values = disease_values.to_numpy(dtype=float)
        if len(values) == 0:
            continue
        sampled_indices = rng.integers(0, len(values), size=(n_resamples, len(values)))
        boot = values[sampled_indices].mean(axis=1)
        results.append(
            {
                "configuration": config,
                "mean": float(values.mean()),
                "ci_low": float(np.quantile(boot, 0.025)),
                "ci_high": float(np.quantile(boot, 0.975)),
                "n_diseases": int(len(values)),
            }
        )
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values("mean", ascending=False)


def cluster_bootstrap_ci_by_stratum(
    df: pd.DataFrame,
    value_cols: list[str],
    n_resamples: int = 20000,
    seed: int = 42,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for (config, stratum), group in df.groupby(["configuration", DIRECT_RELATION_STRATUM_COLUMN]):
        for value_col in value_cols:
            disease_values = group.groupby("disease")[value_col].mean().dropna()
            values = disease_values.to_numpy(dtype=float)
            if not len(values):
                continue
            sampled_indices = rng.integers(0, len(values), size=(n_resamples, len(values)))
            boot = values[sampled_indices].mean(axis=1)
            rows.append(
                {
                    "configuration": config,
                    DIRECT_RELATION_STRATUM_COLUMN: stratum,
                    "metric": value_col,
                    "mean": float(values.mean()),
                    "ci_low": float(np.quantile(boot, 0.025)),
                    "ci_high": float(np.quantile(boot, 0.975)),
                    "n_diseases": int(len(values)),
                    "n_unique_cases": int(group["case_id"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def direct_relation_cluster_bootstrap_association(
    df: pd.DataFrame,
    value_cols: list[str],
    n_resamples: int = 20000,
    seed: int = 42,
) -> pd.DataFrame:
    """Estimate the observational direct-present minus absent score association.

    Diseases are resampled as clusters. This is a sensitivity analysis, not a causal
    estimate, because direct-relation availability was not randomized and candidate
    maturity/evidence density may differ between strata.
    """
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    present_label = "direct_relation_present"
    absent_label = "direct_relation_absent"

    for config, group in df.groupby("configuration"):
        diseases = sorted(group["disease"].unique())
        disease_index = {disease: i for i, disease in enumerate(diseases)}
        sampled = rng.integers(0, len(diseases), size=(n_resamples, len(diseases)))
        for value_col in value_cols:
            p_sum = np.zeros(len(diseases), dtype=float)
            p_count = np.zeros(len(diseases), dtype=float)
            a_sum = np.zeros(len(diseases), dtype=float)
            a_count = np.zeros(len(diseases), dtype=float)
            for disease, disease_group in group.groupby("disease"):
                idx = disease_index[disease]
                present = disease_group[
                    disease_group[DIRECT_RELATION_STRATUM_COLUMN] == present_label
                ][value_col].dropna()
                absent = disease_group[
                    disease_group[DIRECT_RELATION_STRATUM_COLUMN] == absent_label
                ][value_col].dropna()
                p_sum[idx] = float(present.sum())
                p_count[idx] = float(len(present))
                a_sum[idx] = float(absent.sum())
                a_count[idx] = float(len(absent))

            observed_present = p_sum.sum() / p_count.sum() if p_count.sum() else np.nan
            observed_absent = a_sum.sum() / a_count.sum() if a_count.sum() else np.nan
            if not np.isfinite(observed_present) or not np.isfinite(observed_absent):
                continue

            boot_p_count = p_count[sampled].sum(axis=1)
            boot_a_count = a_count[sampled].sum(axis=1)
            valid = (boot_p_count > 0) & (boot_a_count > 0)
            boot_diff = (
                p_sum[sampled][valid].sum(axis=1) / boot_p_count[valid]
                - a_sum[sampled][valid].sum(axis=1) / boot_a_count[valid]
            )
            rows.append(
                {
                    "configuration": config,
                    "metric": value_col,
                    "mean_direct_present": float(observed_present),
                    "mean_direct_absent": float(observed_absent),
                    "mean_difference_present_minus_absent": float(observed_present - observed_absent),
                    "ci_low": float(np.quantile(boot_diff, 0.025)),
                    "ci_high": float(np.quantile(boot_diff, 0.975)),
                    "bootstrap_probability_difference_gt_0": float((boot_diff > 0).mean()),
                    "n_valid_bootstrap_resamples": int(len(boot_diff)),
                    "n_diseases_total": int(len(diseases)),
                    "n_diseases_with_direct_present": int((p_count > 0).sum()),
                    "n_diseases_with_direct_absent": int((a_count > 0).sum()),
                    "n_direct_present_rows": int(p_count.sum()),
                    "n_direct_absent_rows": int(a_count.sum()),
                    "interpretation": "observational association; not a causal leakage estimate",
                }
            )
    return pd.DataFrame(rows)


def paired_cluster_bootstrap_difference(
    df: pd.DataFrame,
    config_a: str,
    config_b: str,
    value_col: str = "primary_score",
    n_resamples: int = 20000,
    seed: int = 42,
) -> pd.DataFrame:
    disease_config = (
        df[df["configuration"].isin([config_a, config_b])]
        .groupby(["disease", "configuration"])[value_col]
        .mean()
        .unstack("configuration")
        .dropna(subset=[config_a, config_b])
    )
    if disease_config.empty:
        raise ValueError(f"No paired disease values for {config_a} versus {config_b}")
    differences = disease_config[config_a].to_numpy(dtype=float) - disease_config[config_b].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(0, len(differences), size=(n_resamples, len(differences)))
    boot = differences[sampled_indices].mean(axis=1)
    return pd.DataFrame(
        [
            {
                "configuration_a": config_a,
                "configuration_b": config_b,
                "mean_difference_a_minus_b": float(differences.mean()),
                "ci_low": float(np.quantile(boot, 0.025)),
                "ci_high": float(np.quantile(boot, 0.975)),
                "bootstrap_probability_a_gt_b": float((boot > 0).mean()),
                "n_paired_diseases": int(len(differences)),
            }
        ]
    )


def role_assignment_primary_marginals(candidate_scores: pd.DataFrame) -> pd.DataFrame:
    """Factorial marginal effect of assigning each model to each role on primary score."""
    rows: list[dict[str, Any]] = []
    disease_config = candidate_scores.groupby(["configuration", "disease"])["primary_score"].mean().reset_index()
    for role, position in (("Sponsor", 0), ("Expert", 1), ("Chair", 2)):
        work = disease_config.copy()
        work["model_code"] = work["configuration"].str[position]
        for model, group in work.groupby("model_code"):
            rows.append(
                {
                    "role": role,
                    "model_code": model,
                    "mean_primary_score": float(group["primary_score"].mean()),
                    "n_configuration_disease_units": int(len(group)),
                    "n_configurations": int(group["configuration"].nunique()),
                    "n_diseases": int(group["disease"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def role_diagnostic_marginals(candidate_scores: pd.DataFrame) -> pd.DataFrame:
    """Marginals for the role-specific diagnostic assigned to the corresponding role."""
    rows: list[dict[str, Any]] = []
    specs = [
        ("Sponsor", 0, "sponsor_quality"),
        ("Expert", 1, "expert_quality"),
        ("Chair", 2, "chair_quality"),
        ("Chair", 2, "chair_synthesis"),
    ]
    for role, position, metric in specs:
        disease_config = candidate_scores.groupby(["configuration", "disease"])[metric].mean().reset_index()
        disease_config["model_code"] = disease_config["configuration"].str[position]
        for model, group in disease_config.groupby("model_code"):
            rows.append(
                {
                    "role": role,
                    "diagnostic": metric,
                    "model_code": model,
                    "mean_score": float(group[metric].mean()),
                    "sd_across_configuration_disease_units": float(group[metric].std(ddof=1)),
                    "n_configuration_disease_units": int(len(group)),
                    "n_configurations": int(group["configuration"].nunique()),
                    "n_diseases": int(group["disease"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def role_diagnostic_marginals_by_direct_relation(candidate_scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stratum, group in candidate_scores.groupby(DIRECT_RELATION_STRATUM_COLUMN):
        result = role_diagnostic_marginals(group)
        if not result.empty:
            result.insert(0, DIRECT_RELATION_STRATUM_COLUMN, stratum)
            rows.extend(result.to_dict("records"))
    return pd.DataFrame(rows)


def critical_flag_summary(candidate_scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for config, group in candidate_scores.groupby("configuration"):
        flags_per_row = group["critical_flags"].fillna("").map(lambda s: {x for x in str(s).split("|") if x})
        for flag in CRITICAL_FLAGS:
            n = int(flags_per_row.map(lambda x: flag in x).sum())
            rows.append(
                {
                    "configuration": config,
                    "critical_flag": flag,
                    "n_flagged": n,
                    "n_reports": int(len(group)),
                    "flag_rate": float(n / len(group)),
                }
            )
    return pd.DataFrame(rows)


def critical_flag_summary_by_direct_relation(candidate_scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stratum, group in candidate_scores.groupby(DIRECT_RELATION_STRATUM_COLUMN):
        result = critical_flag_summary(group)
        if not result.empty:
            result.insert(1, DIRECT_RELATION_STRATUM_COLUMN, stratum)
            rows.extend(result.to_dict("records"))
    return pd.DataFrame(rows)


def score_distribution_by_direct_relation(candidate_scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stratum, group in candidate_scores.groupby(DIRECT_RELATION_STRATUM_COLUMN):
        for metric in SCORE_COLUMNS:
            values = group[metric].dropna().astype(float)
            rows.append(
                {
                    DIRECT_RELATION_STRATUM_COLUMN: stratum,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "median": float(values.median()),
                    "q25": float(values.quantile(0.25)),
                    "q75": float(values.quantile(0.75)),
                    "n_candidate_configuration_rows": int(len(values)),
                    "n_unique_cases": int(group["case_id"].nunique()),
                    "n_diseases": int(group["disease"].nunique()),
                    "n_configurations": int(group["configuration"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def configuration_scores_by_direct_relation(candidate_scores: pd.DataFrame) -> pd.DataFrame:
    aggregate = (
        candidate_scores.groupby(["configuration", DIRECT_RELATION_STRATUM_COLUMN])[SCORE_COLUMNS]
        .mean()
        .reset_index()
    )
    counts = (
        candidate_scores.groupby(["configuration", DIRECT_RELATION_STRATUM_COLUMN])
        .agg(n_cases=("case_id", "nunique"), n_diseases=("disease", "nunique"))
        .reset_index()
    )
    aggregate = aggregate.merge(counts, on=["configuration", DIRECT_RELATION_STRATUM_COLUMN], how="left")
    aggregate["rank_within_stratum"] = (
        aggregate.groupby(DIRECT_RELATION_STRATUM_COLUMN)["primary_score"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    return aggregate.sort_values([DIRECT_RELATION_STRATUM_COLUMN, "rank_within_stratum", "configuration"])


def configuration_rank_stability(
    full_summary: pd.DataFrame,
    stratified_summary: pd.DataFrame,
) -> pd.DataFrame:
    full = full_summary[["configuration", "primary_score"]].copy()
    full["full_rank"] = full["primary_score"].rank(method="min", ascending=False).astype(int)
    full = full.rename(columns={"primary_score": "full_primary_score"})
    no_direct = stratified_summary[
        stratified_summary[DIRECT_RELATION_STRATUM_COLUMN] == "direct_relation_absent"
    ][["configuration", "primary_score", "rank_within_stratum"]].copy()
    no_direct = no_direct.rename(
        columns={
            "primary_score": "no_direct_primary_score",
            "rank_within_stratum": "no_direct_rank",
        }
    )
    result = full.merge(no_direct, on="configuration", how="left", validate="one_to_one")
    result["no_direct_minus_full_score"] = result["no_direct_primary_score"] - result["full_primary_score"]
    result["no_direct_rank_minus_full_rank"] = result["no_direct_rank"] - result["full_rank"]
    return result.sort_values("full_rank")


def _write_direct_relation_readme(out: Path, case_metadata: pd.DataFrame) -> None:
    n_present = int(case_metadata["direct_development_relation_present"].sum())
    n_total = int(len(case_metadata))
    n_absent = n_total - n_present
    text = f"""# Direct-development-relation sensitivity analysis

The fixed evaluation set contains {n_present} candidates with and {n_absent} candidates without a direct drug-disease development relation ({n_total} total).

These analyses are prespecified sensitivity analyses for a report-generation task. They are not causal leakage estimates because direct-relation availability was not randomized and may be associated with candidate maturity, evidence density, disease area, and report-generation difficulty.

Interpretation rules:
- A higher evidence-groundedness score in the direct-relation-present stratum can reflect genuinely more direct supplied evidence.
- Mechanistic consistency, scientific specificity, uncertainty calibration, and experimental actionability must not be awarded merely because an indication or tested-indication relation exists.
- Equivalent KG and RAG records for the same underlying relation are duplicate provenance, not independent corroboration.
- The no-direct subset ranking is the primary robustness check for whether configuration selection depends on known development relations.
- An identical-candidate masked paired ablation would be required for a causal estimate of the effect of removing direct-development evidence.
"""
    (out / "DIRECT_RELATION_SENSITIVITY_README.md").write_text(text, encoding="utf-8")


def analyze(
    score_df: pd.DataFrame,
    qc_df: pd.DataFrame,
    output_dir: str | Path,
    n_bootstrap: int = 20000,
    seed: int = 42,
    case_metadata_df: pd.DataFrame | None = None,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    required_qc = {"configuration", "case_id", "disease", "qc_pass"}
    missing_qc = required_qc - set(qc_df.columns)
    if missing_qc:
        raise ValueError(f"QC table missing columns: {sorted(missing_qc)}")

    merged = score_df.merge(
        qc_df[["configuration", "case_id", "disease", "qc_pass"]],
        on=["configuration", "case_id"],
        how="left",
        validate="one_to_one",
    )
    if merged["disease"].isna().any() or merged["qc_pass"].isna().any():
        missing = merged.loc[
            merged["disease"].isna() | merged["qc_pass"].isna(),
            ["configuration", "case_id"],
        ]
        raise ValueError(f"Judge rows missing QC match: {missing.to_dict('records')}")
    if not merged["qc_pass"].astype(bool).all():
        raise ValueError("Judge scores include one or more hard-QC-failed panels")

    if case_metadata_df is not None and not case_metadata_df.empty:
        metadata = case_metadata_df.copy()
        metadata_disease = metadata[["case_id", "disease"]].rename(columns={"disease": "packet_disease"})
        merged = merged.merge(
            metadata.drop(columns=["disease"]),
            on="case_id",
            how="left",
            validate="many_to_one",
        )
        if merged["direct_development_relation_present"].isna().any():
            missing = merged.loc[
                merged["direct_development_relation_present"].isna(), ["configuration", "case_id"]
            ]
            raise ValueError(f"Judge rows missing case metadata: {missing.to_dict('records')}")
        check = qc_df[["case_id", "disease"]].drop_duplicates().merge(
            metadata_disease, on="case_id", how="left", validate="one_to_one"
        )
        mismatch = check[check["disease"] != check["packet_disease"]]
        if not mismatch.empty:
            raise ValueError(f"Disease mismatch between QC and packet metadata: {mismatch.to_dict('records')}")

    expected_cases_per_config = merged.groupby("configuration")["case_id"].nunique()
    if expected_cases_per_config.nunique() != 1:
        raise ValueError("Configurations have unequal judged case counts: " + repr(expected_cases_per_config.to_dict()))

    merged.to_csv(out / "candidate_level_scores.csv", index=False)
    disease_level = (
        merged.groupby(["configuration", "disease"])[SCORE_COLUMNS]
        .mean()
        .reset_index()
    )
    disease_level.to_csv(out / "disease_level_scores.csv", index=False)

    summary = merged.groupby("configuration")[SCORE_COLUMNS].mean().reset_index()
    qc_summary = qc_df.groupby("configuration")["qc_pass"].mean().rename("qc_pass_rate").reset_index()
    summary = summary.merge(qc_summary, on="configuration", how="left")
    summary = summary.sort_values(["qc_pass_rate", "primary_score"], ascending=False)
    summary.to_csv(out / "configuration_summary.csv", index=False)

    ci = cluster_bootstrap_ci(merged, "primary_score", n_bootstrap, seed)
    ci.to_csv(out / "configuration_primary_score_bootstrap.csv", index=False)
    role_assignment_primary_marginals(merged).to_csv(
        out / "role_assignment_primary_score_marginals.csv", index=False
    )
    role_diagnostic_marginals(merged).to_csv(out / "role_diagnostic_marginals.csv", index=False)
    critical_flag_summary(merged).to_csv(out / "critical_flag_summary.csv", index=False)

    top3 = summary.head(3)["configuration"].tolist()
    (out / "top3_configurations.txt").write_text("\n".join(top3) + "\n", encoding="utf-8")
    if len(top3) >= 2:
        paired = paired_cluster_bootstrap_difference(
            merged,
            top3[0],
            top3[1],
            "primary_score",
            n_bootstrap,
            seed,
        )
        paired.to_csv(out / "top_vs_runner_up_paired_bootstrap.csv", index=False)

    note_columns = ["configuration", "case_id", "disease", "strengths", "weaknesses", "critical_flags"]
    if DIRECT_RELATION_STRATUM_COLUMN in merged.columns:
        note_columns.insert(3, DIRECT_RELATION_STRATUM_COLUMN)
    merged[note_columns].to_csv(out / "qualitative_notes.csv", index=False)

    if case_metadata_df is None or case_metadata_df.empty:
        return

    case_metadata_df.to_csv(out / "direct_development_relation_case_map.csv", index=False)
    score_distribution_by_direct_relation(merged).to_csv(
        out / "score_distribution_by_direct_relation.csv", index=False
    )
    stratified = configuration_scores_by_direct_relation(merged)
    stratified.to_csv(out / "configuration_scores_by_direct_relation.csv", index=False)
    configuration_rank_stability(summary, stratified).to_csv(
        out / "configuration_rank_stability_no_direct.csv", index=False
    )
    cluster_bootstrap_ci_by_stratum(merged, SCORE_COLUMNS, n_bootstrap, seed).to_csv(
        out / "direct_relation_stratum_cluster_bootstrap.csv", index=False
    )
    direct_relation_cluster_bootstrap_association(
        merged,
        [*DIMENSIONS, "primary_score"],
        n_bootstrap,
        seed,
    ).to_csv(out / "direct_relation_score_association_by_configuration.csv", index=False)
    role_diagnostic_marginals_by_direct_relation(merged).to_csv(
        out / "role_diagnostic_marginals_by_direct_relation.csv", index=False
    )
    critical_flag_summary_by_direct_relation(merged).to_csv(
        out / "critical_flag_summary_by_direct_relation.csv", index=False
    )
    _write_direct_relation_readme(out, case_metadata_df)
