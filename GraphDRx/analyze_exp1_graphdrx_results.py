#!/usr/bin/env python3
"""
Analyze Exp1 GraphDRx ablation-study outputs.

Default layout:
  /GraphDRx/
    output/<run_folder>/
    analysis/

Run:
  cd /GraphDRx
  python analyze_exp1_graphdrx_results.py

Optional:
  python analyze_exp1_graphdrx_results.py --make-plots

Outputs:
  analysis/graphdrx_ablation_report.xlsx
  analysis/graphdrx_overall_metrics.csv
  analysis/graphdrx_area_metrics.csv
  analysis/graphdrx_disease_metrics.csv
  analysis/graphdrx_delta_vs_main_overall.csv
  analysis/graphdrx_delta_vs_main_area.csv
  analysis/graphdrx_output_inventory.csv
"""

from __future__ import annotations

import ast
import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import pandas as pd


RUN_INFO: Dict[str, Dict[str, str]] = {
    "main": {
        "label": "GraphDRx Main (T+B+C+D)",
        "description": "Final GraphDRx configuration combining T, B, C, and D retrieval components.",
    },
    "ablation_no_T": {
        "label": "GraphDRx -T (B+C+D)",
        "description": "Matched ablation removing target-disease biological-context retrieval (T).",
    },
    "ablation_no_B": {
        "label": "GraphDRx -B (T+C+D)",
        "description": "Matched ablation removing similar-disease biological-context retrieval (B).",
    },
    "ablation_no_C": {
        "label": "GraphDRx -C (T+B+D)",
        "description": "Matched ablation removing semantic therapeutic-prior support and prior-only KG verification (C).",
    },
    "ablation_no_D": {
        "label": "GraphDRx -D (T+B+C)",
        "description": "Matched ablation removing direct target-disease gene-action retrieval (D).",
    },
}

EXPECTED_RUNS = [
    "main",
    "ablation_no_T",
    "ablation_no_B",
    "ablation_no_C",
    "ablation_no_D",
]

METRIC_COLS = [
    "Hit@1", "Hit@10", "Hit@50", "Hit@100", "Hit@200",
    "MRR@100", "MacroR@100", "MicroR@100",
    "gold_hit@100", "gold_hit@200",
]

PREFERRED_OVERALL_ORDER = [
    "method", "run", "n_diseases", "total_gold",
    "Hit@1", "Hit@10", "Hit@50", "Hit@100", "Hit@200",
    "MRR@100", "MacroR@100", "MicroR@100",
    "gold_hit@100", "gold_hit@200",
]
PREFERRED_AREA_ORDER = ["area"] + PREFERRED_OVERALL_ORDER


def norm_col(c: Any) -> str:
    return str(c).strip().lower().replace(" ", "_").replace("-", "_")


def first_existing_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lower = {norm_col(c): c for c in df.columns}
    for cand in candidates:
        if norm_col(cand) in lower:
            return lower[norm_col(cand)]
    return None


def rename_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename: Dict[str, str] = {}
    for c in df.columns:
        k = norm_col(c)

        # GraphDRx disease-level summary columns.
        # These are the columns written by *_summary_indication_only.csv.
        if k in {"retrieval_hit_at_10", "hit@10", "hit_10", "hits@10", "hit10", "hitat10"}:
            rename[c] = "Hit@10"
        elif k in {"retrieval_hit_at_1", "hit@1", "hit_1", "hits@1", "hit1", "hitat1"}:
            rename[c] = "Hit@1"
        elif k in {"retrieval_hit_at_50", "hit@50", "hit_50", "hits@50", "hit50", "hitat50"}:
            rename[c] = "Hit@50"
        elif k in {"retrieval_hit_at_100", "hit@100", "hit_100", "hits@100", "hit100", "hitat100"}:
            rename[c] = "Hit@100"
        elif k in {"retrieval_hit_at_200", "hit@200", "hit_200", "hits@200", "hit200", "hitat200"}:
            rename[c] = "Hit@200"
        elif k in {"retrieval_mrr", "mrr", "mean_reciprocal_rank"}:
            rename[c] = "MRR@100"
        elif k in {"retrieval_gold_recall_at_100", "meanr@100", "mean_recall@100", "mean_recall_100", "meanr100", "recall@100", "recall_100"}:
            rename[c] = "MacroR@100"
        elif k in {"micror@100", "micro_recall@100", "micro_recall_100", "micror100"}:
            rename[c] = "MicroR@100"
        elif k in {"retrieval_gold_hit_count_at_100", "gold_hit@100", "gold_hits@100", "gold_hit_100", "n_gold_hit_100"}:
            rename[c] = "gold_hit@100"
        elif k in {"retrieval_gold_hit_count_at_200", "gold_hit@200", "gold_hits@200", "gold_hit_200", "n_gold_hit_200"}:
            rename[c] = "gold_hit@200"
        elif k in {"total_gold", "n_gold", "gold_total", "num_gold"}:
            rename[c] = "total_gold"
        elif k in {"n_diseases", "num_diseases"}:
            rename[c] = "n_diseases"
        elif k in {"disease_name", "query_disease"}:
            rename[c] = "disease"
        elif k in {"disease_area", "domain", "category"}:
            rename[c] = "area"
    return df.rename(columns=rename)


def read_csv_safely(path: Path) -> Optional[pd.DataFrame]:
    for enc in ("utf-8", "utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"[WARN] failed to read CSV {path}: {e}")
            return None
    print(f"[WARN] failed to read CSV {path}: encoding issue")
    return None


def load_json_safely(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] failed to read JSON {path}: {e}")
        return None


def flatten_metric_dict(obj: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                out.update(flatten_metric_dict(v, key))
            elif isinstance(v, (str, int, float, bool)) or v is None:
                out[key] = v
    return out


def safe_relative_to(path: Path, root: Path) -> str:
    """Return a stable display path without crashing when paths use different roots."""
    path = path.resolve()
    root = root.resolve()
    try:
        return str(path.relative_to(root))
    except ValueError:
        try:
            return str(path.relative_to(Path.cwd().resolve()))
        except ValueError:
            return str(path)


def first_rr_at_k(ranks: Any, k: int = 100) -> float:
    values = []

    if isinstance(ranks, dict):
        values = list(ranks.values())
    else:
        text = "" if ranks is None else str(ranks).strip()
        parsed = None

        if text and text.lower() not in {"nan", "none", "null"}:
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(text)
                    break
                except Exception:
                    pass

        if isinstance(parsed, dict):
            values = list(parsed.values())
        elif isinstance(parsed, (list, tuple, set)):
            values = list(parsed)

    valid = []
    for x in values:
        try:
            r = int(float(x))
        except (TypeError, ValueError):
            continue
        if 1 <= r <= k:
            valid.append(r)

    return 0.0 if not valid else 1.0 / min(valid)


def canonicalize_run_name(name: str) -> str:
    """Map current stable folders, older folders, and slugged suffix folders to a run id."""
    raw = str(name or "").strip().strip("_")
    s = raw.lower().replace("-", "_")
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")

    # Current runner names are canonical. Legacy graphdrx_* names remain
    # accepted so older output folders can still be analyzed.
    patterns = [
        ("ablation_no_t", "ablation_no_T"),
        ("ablation_no_b", "ablation_no_B"),
        ("ablation_no_c", "ablation_no_C"),
        ("ablation_no_d", "ablation_no_D"),
        ("main", "main"),
    ]
    for needle, run in patterns:
        if needle in s:
            return run
    return s or raw


def inspect_run_dir(run_dir: Path) -> Tuple[str, str]:
    """Return logical run id and method label, using run_manifest.json when available."""
    logical_run = canonicalize_run_name(run_dir.name)
    manifest_path = run_dir / "run_manifest.json"
    manifest = load_json_safely(manifest_path) if manifest_path.exists() else None
    if isinstance(manifest, dict):
        manifest_run = manifest.get("run_name") or manifest.get("run")
        if manifest_run:
            logical_run = canonicalize_run_name(str(manifest_run))
    method = RUN_INFO.get(logical_run, {}).get("label", logical_run)
    return logical_run, method


def find_run_dirs(output_root: Path) -> List[Path]:
    if not output_root.exists():
        return []
    return sorted([p for p in output_root.iterdir() if p.is_dir() and not p.name.startswith(".")])


def inventory_run(run_dir: Path, root: Path, run: Optional[str] = None) -> List[Dict[str, Any]]:
    rows = []
    run_id = run or canonicalize_run_name(run_dir.name)
    for p in sorted(run_dir.rglob("*")):
        if any(part.startswith(".") for part in p.relative_to(run_dir).parts):
            continue
        if p.is_file():
            rows.append({
                "run": run_id,
                "run_dir": run_dir.name,
                "path": safe_relative_to(p, root),
                "size_bytes": p.stat().st_size,
                "suffix": p.suffix,
            })
    return rows


def choose_summary_csv(run_dir: Path) -> Optional[Path]:
    candidates = []
    for pat in ["*summary_indication_only*.csv", "*summary*.csv", "*metrics*.csv"]:
        candidates.extend(run_dir.glob(pat))
        candidates.extend([p for p in run_dir.rglob(pat) if not any(part.startswith(".") for part in p.relative_to(run_dir).parts)])
    candidates = sorted(set(candidates), key=lambda p: (
        0 if "summary_indication_only" in p.name else 1,
        len(str(p)),
        str(p),
    ))
    return candidates[0] if candidates else None


def choose_area_metrics_csv(run_dir: Path) -> Optional[Path]:
    candidates = []
    for pat in ["*stage_area_metrics_indication_only*.csv", "*area_metrics*.csv", "*area*indication*.csv"]:
        candidates.extend(run_dir.glob(pat))
        candidates.extend([p for p in run_dir.rglob(pat) if not any(part.startswith(".") for part in p.relative_to(run_dir).parts)])
    candidates = sorted(set(candidates), key=lambda p: (len(str(p)), str(p)))
    return candidates[0] if candidates else None


def existing_metric_table(df: pd.DataFrame, run: str, method: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = rename_metric_columns(df.copy())
    if "retrieval_gold_ranks" in df.columns:
        df["MRR@100"] = df["retrieval_gold_ranks"].map(
            lambda x: first_rr_at_k(x, 100)
    )
    has_metric = any(c in df.columns for c in METRIC_COLS)
    if not has_metric:
        return pd.DataFrame(), pd.DataFrame()
    work = df.copy()
    work["run"] = run
    work["method"] = method

    overall_df = pd.DataFrame()
    area_df = pd.DataFrame()
    if "area" in work.columns and "disease" not in work.columns:
        area_df = work.copy()
    if "area" not in work.columns and "disease" not in work.columns:
        overall_df = work.copy()
    if "scope" in work.columns:
        mask = work["scope"].astype(str).str.lower().str.contains("overall|all|matched", regex=True, na=False)
        if mask.any():
            overall_df = work.loc[mask].copy()
    return overall_df, area_df


def case_set_id_from_diseases(values: Iterable[Any]) -> str:
    diseases = sorted({str(x).strip().lower() for x in values if str(x).strip()})
    joined = "\n".join(diseases)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12] if joined else "empty"


def aggregate_from_disease_rows(df: pd.DataFrame, run: str, method: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = rename_metric_columns(df.copy())
    disease_col = first_existing_col(df, ["disease", "disease_name", "query_disease"])
    area_col = first_existing_col(df, ["area", "disease_area", "domain", "category"])
    if disease_col is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if disease_col != "disease":
        df = df.rename(columns={disease_col: "disease"})
    if area_col and area_col != "area":
        df = df.rename(columns={area_col: "area"})
    if "area" not in df.columns:
        df["area"] = "unknown"
    if "retrieval_gold_ranks" in df.columns:
        df["MRR@100"] = df["retrieval_gold_ranks"].map(
            lambda x: first_rr_at_k(x, 100)
        )

    df["run"] = run
    df["method"] = method
    overall_case_set_id = case_set_id_from_diseases(df["disease"].tolist())
    for c in METRIC_COLS + ["total_gold"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    def agg_group(g: pd.DataFrame) -> pd.Series:
        # Use one gold count per disease to avoid double counting if duplicate rows exist.
        if "total_gold" in g.columns:
            total_gold = g.drop_duplicates("disease")["total_gold"].sum()
        else:
            total_gold = math.nan
        row = {
            "n_diseases": g["disease"].nunique(dropna=True),
            "total_gold": total_gold,
        }
        for c in [ "Hit@1", "Hit@10", "Hit@50", "Hit@100", "Hit@200", "MRR@100", "MacroR@100"]:
            row[c] = g[c].mean() if c in g.columns else math.nan
        for c in ["gold_hit@100", "gold_hit@200"]:
            row[c] = g[c].sum() if c in g.columns else math.nan
        if "gold_hit@100" in g.columns and "total_gold" in g.columns and total_gold and total_gold > 0:
            row["MicroR@100"] = g["gold_hit@100"].sum() / total_gold
        elif "MicroR@100" in g.columns and g["MicroR@100"].notna().any():
            row["MicroR@100"] = g["MicroR@100"].mean()
        else:
            row["MicroR@100"] = math.nan
        return pd.Series(row)

    disease_df = df.copy()
    if "total_gold" in disease_df.columns:
        evaluable_df = disease_df[pd.to_numeric(disease_df["total_gold"], errors="coerce").fillna(0) > 0].copy()
    else:
        evaluable_df = disease_df.copy()
    overall_df = pd.DataFrame([agg_group(evaluable_df)]) if not evaluable_df.empty else pd.DataFrame([{
        "n_diseases": 0, "total_gold": 0
    }])
    overall_df["n_cases_total"] = disease_df["disease"].nunique(dropna=True)
    overall_df["n_zero_gold_cases"] = overall_df["n_cases_total"] - overall_df["n_diseases"]
    overall_df["run"] = run
    overall_df["method"] = method
    overall_df["case_set_id"] = overall_case_set_id
    grouped = evaluable_df.groupby("area", dropna=False)
    if evaluable_df.empty:
        area_df = pd.DataFrame(columns=["area", "n_diseases", "total_gold"])
    else:
        try:
            area_df = grouped.apply(agg_group, include_groups=False).reset_index()
        except TypeError:  # pandas < 2.2
            area_df = grouped.apply(agg_group).reset_index()
    total_by_area = disease_df.groupby("area", dropna=False)["disease"].nunique().to_dict()
    if not area_df.empty:
        area_df["n_cases_total"] = area_df["area"].map(total_by_area)
        area_df["n_zero_gold_cases"] = area_df["n_cases_total"] - area_df["n_diseases"]
    area_df["run"] = run
    area_df["method"] = method
    area_case_sets = (
        disease_df.groupby("area", dropna=False)["disease"]
        .apply(lambda x: case_set_id_from_diseases(x.tolist()))
        .to_dict()
    )
    area_df["case_set_id"] = area_df["area"].map(area_case_sets) if "area" in area_df.columns else overall_case_set_id
    disease_df["case_set_id"] = overall_case_set_id
    return overall_df, area_df, disease_df


def extract_from_json_metrics(run_dir: Path, run: str, method: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    json_paths = sorted(set(list(run_dir.glob("*metrics*.json")) + list(run_dir.rglob("*metrics*.json"))))
    overall_rows: List[Dict[str, Any]] = []
    area_rows: List[Dict[str, Any]] = []
    for path in json_paths:
        obj = load_json_safely(path)
        if not isinstance(obj, dict):
            continue
        for key in ["overall", "overall_metrics", "strict_metrics", "summary"]:
            if isinstance(obj.get(key), dict):
                flat = flatten_metric_dict(obj[key])
                flat.update({"run": run, "method": method, "source_json": path.name})
                overall_rows.append(flat)
        for key in ["area_metrics", "areas", "by_area"]:
            area_obj = obj.get(key)
            if isinstance(area_obj, dict):
                for area, val in area_obj.items():
                    if isinstance(val, dict):
                        flat = flatten_metric_dict(val)
                        flat.update({"area": area, "run": run, "method": method, "source_json": path.name})
                        area_rows.append(flat)
            elif isinstance(area_obj, list):
                for val in area_obj:
                    if isinstance(val, dict):
                        flat = flatten_metric_dict(val)
                        flat.update({"run": run, "method": method, "source_json": path.name})
                        area_rows.append(flat)
    return rename_metric_columns(pd.DataFrame(overall_rows)), rename_metric_columns(pd.DataFrame(area_rows))


def reorder_columns(df: pd.DataFrame, preferred: List[str]) -> pd.DataFrame:
    if df.empty:
        return df
    cols = [c for c in preferred if c in df.columns]
    cols += [c for c in df.columns if c not in cols]
    return df[cols]


def add_run_order(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "run" not in df.columns:
        return df
    order = {run: i for i, run in enumerate(RUN_INFO)}
    df = df.copy()
    df["_run_order"] = df["run"].map(order).fillna(999).astype(int)
    sort_cols = ["_run_order"]
    if "area" in df.columns:
        sort_cols = ["area", "_run_order"]
    if "disease" in df.columns:
        sort_cols = ["area", "disease", "_run_order"] if "area" in df.columns else ["disease", "_run_order"]
    return df.sort_values(sort_cols).drop(columns=["_run_order"])


def compute_delta_vs_main(df: pd.DataFrame, index_cols: List[str], main_run: str) -> pd.DataFrame:
    """Compute metric deltas against the main run.

    For overall metrics, index_cols is empty, so the main row is broadcast to
    every ablation row. For area/disease metrics, deltas are matched by index_cols.
    """
    if df.empty or "run" not in df.columns:
        return pd.DataFrame()

    metric_cols = [c for c in METRIC_COLS if c in df.columns]
    if not metric_cols:
        return pd.DataFrame()

    base = df[df["run"] == main_run].copy()
    if base.empty:
        return pd.DataFrame()

    cur = df[df["run"] != main_run].copy()
    if cur.empty:
        return pd.DataFrame()

    # Overall table: only compare runs evaluated on the same disease set.
    # This prevents area-filtered sparse diagnostics from being subtracted from
    # runs with different case composition.
    if not index_cols:
        if "case_set_id" in df.columns and base["case_set_id"].notna().any():
            base_cols = ["case_set_id"] + metric_cols
            base = base[base_cols].copy()
            base = base.rename(columns={c: f"{c}_main" for c in metric_cols})
            merged = cur.merge(base, on="case_set_id", how="left")
            merged["delta_case_set_match"] = merged[f"{metric_cols[0]}_main"].notna()
        else:
            base_row = base.iloc[0]
            merged = cur.copy()
            for c in metric_cols:
                merged[f"{c}_main"] = base_row.get(c, math.nan)
            merged["delta_case_set_match"] = True
    else:
        join_cols = list(index_cols)
        if "case_set_id" in df.columns and "case_set_id" not in join_cols:
            join_cols = join_cols + ["case_set_id"]
        base_cols = join_cols + metric_cols
        base = base[base_cols].copy()
        base = base.rename(columns={c: f"{c}_main" for c in metric_cols})
        merged = cur.merge(base, on=join_cols, how="left")
        merged["delta_case_set_match"] = merged[f"{metric_cols[0]}_main"].notna()

    for c in metric_cols:
        merged[f"delta_{c}"] = (
            pd.to_numeric(merged[c], errors="coerce")
            - pd.to_numeric(merged[f"{c}_main"], errors="coerce")
        )

    delta_cols = (
        index_cols
        + (["case_set_id"] if "case_set_id" in merged.columns and "case_set_id" not in index_cols else [])
        + ["method", "run", "delta_case_set_match"]
        + [f"delta_{c}" for c in metric_cols]
        + metric_cols
        + [f"{c}_main" for c in metric_cols]
    )
    return merged[[c for c in delta_cols if c in merged.columns]]


def write_excel(path: Path, sheets: Dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            sheet = re.sub(r"[^A-Za-z0-9_]+", "_", name)[:31]
            df.to_excel(writer, index=False, sheet_name=sheet)
        wb = writer.book
        for ws in wb.worksheets:
            ws.freeze_panes = "A2"
            header = [cell.value for cell in ws[1]]
            for i, col_cells in enumerate(ws.columns, 1):
                values = ["" if c.value is None else str(c.value) for c in col_cells]
                ws.column_dimensions[col_cells[0].column_letter].width = max(10, min(max(map(len, values)) + 2, 55))
                col_name = header[i - 1] if i - 1 < len(header) else ""
                if col_name in {"Hit@1", "Hit@10", "Hit@50", "Hit@100", "Hit@200", "MRR@100", "MacroR@100", "MicroR@100"} or str(col_name).startswith("delta_"):
                    for row in ws.iter_rows(min_row=2, min_col=i, max_col=i):
                        for cell in row:
                            cell.number_format = "0.000"


def make_plots(analysis_dir: Path, overall_df: pd.DataFrame, area_df: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] matplotlib unavailable; skipping plots: {e}")
        return
    plot_dir = analysis_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    if not overall_df.empty:
        for metric in ["Hit@10", "Hit@100", "MRR@100", "MacroR@100", "MicroR@100"]:
            if metric not in overall_df.columns:
                continue
            data = overall_df[["method", metric]].dropna()
            if data.empty:
                continue
            plt.figure(figsize=(max(8, len(data) * 1.3), 5))
            plt.bar(data["method"], data[metric])
            plt.xticks(rotation=35, ha="right")
            plt.ylabel(metric)
            plt.title(f"Overall {metric}")
            plt.tight_layout()
            plt.savefig(plot_dir / f"overall_{metric.replace('@', '').replace('/', '_')}.png", dpi=200)
            plt.close()
    if not area_df.empty and "area" in area_df.columns:
        for metric in ["Hit@100", "MacroR@100", "MicroR@100"]:
            if metric not in area_df.columns:
                continue
            pivot = area_df.pivot_table(index="area", columns="method", values=metric, aggfunc="mean")
            if pivot.empty:
                continue
            ax = pivot.plot(kind="bar", figsize=(max(10, len(pivot.columns) * 1.4), max(5, len(pivot) * 0.55)))
            ax.set_ylabel(metric)
            ax.set_title(f"Area {metric}")
            plt.xticks(rotation=35, ha="right")
            plt.tight_layout()
            plt.savefig(plot_dir / f"area_{metric.replace('@', '').replace('/', '_')}.png", dpi=200)
            plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Exp1 GraphDRx ablation outputs.")
    parser.add_argument("--root", default=None, help="Project root. Defaults to CWD, or to --output-root's parent when --output-root is provided.")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--analysis-dir", default=None)
    parser.add_argument("--main-run", default="main")
    parser.add_argument("--make-plots", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve() if args.output_root else None
    if args.root:
        root = Path(args.root).resolve()
    elif output_root is not None:
        root = output_root.parent.resolve()
    else:
        root = Path.cwd().resolve()
    output_root = output_root or (root / "output")
    if args.analysis_dir:
        analysis_dir = Path(args.analysis_dir).resolve()
    else:
        if output_root.name == "output":
            analysis_dir = root / "analysis"
        else:
            safe_output_name = re.sub(r"[^A-Za-z0-9._-]+", "_", output_root.name).strip("_") or "output"
            analysis_dir = root / f"analysis_{safe_output_name}"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = find_run_dirs(output_root)
    if not run_dirs:
        print(f"[WARN] no run directories found under {output_root}")
    run_dir_info = {p: inspect_run_dir(p) for p in run_dirs}
    run_dir_by_logical: Dict[str, Path] = {}
    for p, (logical_run, _) in run_dir_info.items():
        # Keep the first directory for status reporting; duplicate logical runs are still analyzed.
        run_dir_by_logical.setdefault(logical_run, p)

    readme_df = pd.DataFrame([
        {
            "run": run,
            "method_label": info["label"],
            "description": info["description"],
            "expected_in_default_runner": run in EXPECTED_RUNS,
            "status": "found" if run in run_dir_by_logical else ("missing" if run in EXPECTED_RUNS else "optional_missing"),
            "run_dir": str(run_dir_by_logical[run]) if run in run_dir_by_logical else "",
        }
        for run, info in RUN_INFO.items()
    ])
    observed_runs = set(run_dir_by_logical)
    missing_df = pd.DataFrame(
        [{"run": r, "status": "missing_expected"} for r in EXPECTED_RUNS if r not in observed_runs]
        + [{"run": r, "status": "extra_observed"} for r in sorted(observed_runs) if r not in RUN_INFO]
    )

    overall_parts: List[pd.DataFrame] = []
    area_parts: List[pd.DataFrame] = []
    disease_parts: List[pd.DataFrame] = []
    inventory_rows: List[Dict[str, Any]] = []

    for run_dir in run_dirs:
        run, method = run_dir_info.get(run_dir, inspect_run_dir(run_dir))
        inventory_rows.extend(inventory_run(run_dir, root, run))

        got_overall = pd.DataFrame()
        got_area = pd.DataFrame()
        got_disease = pd.DataFrame()

        summary_csv = choose_summary_csv(run_dir)
        if summary_csv:
            df = read_csv_safely(summary_csv)
            if df is not None and not df.empty:
                disease_overall, disease_area, disease_df = aggregate_from_disease_rows(df, run, method)
                if not disease_df.empty:
                    got_overall, got_area, got_disease = disease_overall, disease_area, disease_df
                    got_disease["source_csv"] = safe_relative_to(summary_csv, root)
                else:
                    got_overall, got_area = existing_metric_table(df, run, method)

        area_csv = choose_area_metrics_csv(run_dir)
        if area_csv and got_area.empty:
            adf = read_csv_safely(area_csv)
            if adf is not None and not adf.empty:
                adf = rename_metric_columns(adf)
                adf["run"] = run
                adf["method"] = method
                got_area = adf

        if got_overall.empty and got_area.empty:
            got_overall, got_area = extract_from_json_metrics(run_dir, run, method)

        if not got_overall.empty:
            overall_parts.append(got_overall)
        if not got_area.empty:
            area_parts.append(got_area)
        if not got_disease.empty:
            disease_parts.append(got_disease)

    overall_df = pd.concat(overall_parts, ignore_index=True) if overall_parts else pd.DataFrame()
    area_df = pd.concat(area_parts, ignore_index=True) if area_parts else pd.DataFrame()
    disease_df = pd.concat(disease_parts, ignore_index=True) if disease_parts else pd.DataFrame()
    inventory_df = pd.DataFrame(inventory_rows)

    overall_df = add_run_order(reorder_columns(rename_metric_columns(overall_df), PREFERRED_OVERALL_ORDER))
    area_df = add_run_order(reorder_columns(rename_metric_columns(area_df), PREFERRED_AREA_ORDER))
    disease_df = add_run_order(rename_metric_columns(disease_df))

    delta_overall_df = add_run_order(compute_delta_vs_main(overall_df, [], args.main_run))
    delta_area_df = add_run_order(compute_delta_vs_main(area_df, ["area"], args.main_run))

    csvs = {
        "graphdrx_overall_metrics.csv": overall_df,
        "graphdrx_area_metrics.csv": area_df,
        "graphdrx_disease_metrics.csv": disease_df,
        "graphdrx_delta_vs_main_overall.csv": delta_overall_df,
        "graphdrx_delta_vs_main_area.csv": delta_area_df,
        "graphdrx_output_inventory.csv": inventory_df,
        "graphdrx_missing_runs.csv": missing_df,
    }
    for name, df in csvs.items():
        df.to_csv(analysis_dir / name, index=False)

    xlsx_path = analysis_dir / "graphdrx_ablation_report.xlsx"
    write_excel(xlsx_path, {
        "README": readme_df,
        "overall": overall_df,
        "area_metrics": area_df,
        "disease_metrics": disease_df,
        "delta_vs_main_overall": delta_overall_df,
        "delta_vs_main_area": delta_area_df,
        "output_inventory": inventory_df,
        "missing_runs": missing_df,
    })

    if args.make_plots:
        make_plots(analysis_dir, overall_df, area_df)

    print("[DONE] GraphDRx ablation analysis")
    print(f"[ROOT] {root}")
    print(f"[OUTPUT_ROOT] {output_root}")
    print(f"[ANALYSIS_DIR] {analysis_dir}")
    print(f"[EXCEL] {xlsx_path}")
    print("\n[RUN STATUS]")
    for _, row in readme_df[readme_df["expected_in_default_runner"]].iterrows():
        print(f"- {row['run']}: {row['status']}")
    optional_found = readme_df[(~readme_df["expected_in_default_runner"]) & (readme_df["status"] == "found")]
    if not optional_found.empty:
        print("[OPTIONAL RUNS FOUND]")
        for _, row in optional_found.iterrows():
            print(f"- {row['run']}: {row['status']}")


if __name__ == "__main__":
    main()
