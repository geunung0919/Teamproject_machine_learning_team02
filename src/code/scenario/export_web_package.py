from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common.paths import (
    RAW_DIR, CLEAN_PATCH_DIR, RECURSIVE_FORECAST_DIR, 
    WEB_PACKAGE_DIR, PUBLIC_ASSETS_DIR
)
from src.common.io import read_csv, write_csv, write_json, copy_file
from src.common.reporting import md_table

FINAL_MODEL = "R3_multioutput_1to5_incremental_delta_HistGradientBoostingRegressor_hgb_05_deeper_regularized"
EXPECTED_2030_TOTAL = 4213774.284467546

def pct_rank(s: pd.Series, ascending: bool = True) -> pd.Series:
    """Helper to convert series values into clean percentile ranks (0-100)."""
    return s.rank(pct=True, ascending=ascending).fillna(0) * 100

def coord_quality(lat: pd.Series, lon: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Flag coordinates as valid, missing, or out of bounds."""
    latn = pd.to_numeric(lat, errors="coerce")
    lonn = pd.to_numeric(lon, errors="coerce")
    missing = latn.isna() | lonn.isna()
    out_bounds = (~missing) & (~latn.between(33.0, 39.5) | ~lonn.between(124.0, 132.0))
    
    flag = pd.Series("valid", index=lat.index, dtype=object)
    flag.loc[missing] = "missing_coordinate"
    flag.loc[out_bounds] = "out_of_korea_bounds"
    return flag.eq("valid"), flag

def total_table(pred: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Generate yearly student count sums grouped by columns."""
    rows = []
    cols = {
        "2025": "student_count_2025", 
        "2026": "pred_student_count_2026", 
        "2027": "pred_student_count_2027", 
        "2028": "pred_student_count_2028", 
        "2029": "pred_student_count_2029", 
        "2030": "pred_student_count_2030"
    }
    
    if group_cols:
        for keys, g in pred.groupby(group_cols, dropna=False):
            key_vals = keys if isinstance(keys, tuple) else (keys,)
            base_total = g["student_count_2025"].sum()
            for y, col in cols.items():
                total = g[col].sum()
                row = {c: v for c, v in zip(group_cols, key_vals)}
                row.update({
                    "year": int(y),
                    "total_students": total,
                    "delta_from_2025": total - base_total,
                    "pct_change_from_2025": (total - base_total) / base_total if base_total else np.nan,
                })
                rows.append(row)
    else:
        base_total = pred["student_count_2025"].sum()
        for y, col in cols.items():
            total = pred[col].sum()
            rows.append({
                "year": int(y),
                "total_students": total,
                "delta_from_2025": total - base_total,
                "pct_change_from_2025": (total - base_total) / base_total if base_total else np.nan,
            })
    return pd.DataFrame(rows)

def run_export(
    recur_dir: Path, patch_dir: Path, output_dir: Path, public_dir: Path
) -> dict[str, Any]:
    """Execute scenario packaging, risk flagging, priority scoring, and dashboard JSON/CSV generation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    public_dir.mkdir(parents=True, exist_ok=True)
    
    # Safety guard: Check final model type
    if "HistGradientBoostingRegressor" not in FINAL_MODEL:
        print(f"Aborted: FINAL_MODEL must be a HistGradientBoostingRegressor, got {FINAL_MODEL}")
        return {}

    scen_all = read_csv(recur_dir / "scenario" / "recursive_multioutput_school_predictions_2026_2030_p1.csv")
    if scen_all.empty:
        tuned_scen_path = recur_dir.parent / "v5_r3_r6_rf_hist_tuning_v1" / "scenario_best_model_2026_2030" / "best_tuned_school_predictions_2026_2030_with_excluded_correction.csv"
        if tuned_scen_path.exists():
            scen_all = read_csv(tuned_scen_path)
            if "scenario_layer" in scen_all.columns:
                scen_all = scen_all[scen_all["scenario_layer"].eq("p1_main")].copy()
        else:
            print("Recursive multioutput school predictions file missing. Web package export halted.")
            return {}
        
    scen = scen_all[scen_all["candidate_name"].eq(FINAL_MODEL)].copy()
    base = read_csv(patch_dir / "model_views" / "scenario_base_2025.csv")
    totals = read_csv(recur_dir / "scenario" / "recursive_multioutput_total_students_by_year.csv")
    tuned_total_path = recur_dir.parent / "v5_r3_r6_rf_hist_tuning_v1" / "scenario_best_model_2026_2030" / "best_tuned_total_students_by_year_with_excluded_correction.csv"
    
    if totals.empty or not totals["candidate_name"].eq(FINAL_MODEL).any():
        if tuned_total_path.exists():
            totals = read_csv(tuned_total_path)
            if "total_students_corrected" in totals.columns:
                totals["total_students"] = totals["total_students_corrected"]
        else:
            print("Recursive multioutput totals file missing. Web package export halted.")
            return {}

    # Safety guard: Check if RF total is detected
    for col in ["total_students", "total_students_corrected"]:
        if col in totals.columns:
            rf_rows = totals[abs(totals[col] - 4263620.735797141) < 1.0]
            if not rf_rows.empty:
                print("Aborted: RandomForest scenario total 4263620.74 detected.")
                return {}

    total_final = totals[totals["candidate_name"].eq(FINAL_MODEL)].copy()
    
    # Merge base features
    base_cols = ["school_key", "status", "class_count", "teacher_count", "students_per_class", "students_per_teacher", "coordinate_valid", "coordinate_source", "coordinate_invalid_reason", "nearest_same_level_distance_km", "same_level_school_count_within_5km", "isolation_score"]
    base_use = base[[c for c in base_cols if c in base.columns]].copy()
    web = scen.merge(base_use, on="school_key", how="left")
    
    # Join coordinates
    geo_path = RAW_DIR / "school_data_2008_2025_geocoded.csv"
    geo = read_csv(geo_path)
    if not geo.empty:
        geo25 = geo[geo["year"].eq(2025)].copy()
        geo25 = geo25.rename(columns={"시도": "sido", "행정구": "sgg", "학교급": "school_level", "학교명": "school_name", "lttud": "latitude", "lgtud": "longitude"})
        geo_cols = ["sido", "sgg", "school_level", "school_name", "latitude", "longitude", "coordinate_source"]
        geo25 = geo25[[c for c in geo_cols if c in geo25.columns]].drop_duplicates(["sido", "sgg", "school_level", "school_name"])
        web = web.merge(geo25, on=["sido", "sgg", "school_level", "school_name"], how="left", suffixes=("", "_geo"))
        if "coordinate_source_geo" in web.columns:
            web["coordinate_source"] = web["coordinate_source"].fillna(web["coordinate_source_geo"])
            
    # Calculate coordinate quality
    web["coordinate_valid"], web["coordinate_quality_flag"] = coord_quality(web.get("latitude", pd.Series(index=web.index)), web.get("longitude", pd.Series(index=web.index)))
    web["school_status_2025"] = web.get("status", "")
    
    for y in range(2026, 2031):
        web[f"delta_2025_{y}"] = web[f"pred_student_count_{y}"] - web["student_count_2025"]
        web[f"pct_change_2025_{y}"] = np.where(web["student_count_2025"] > 0, web[f"delta_2025_{y}"] / web["student_count_2025"], np.nan)
        
    web = web.rename(columns={
        "class_count": "class_count_2025",
        "teacher_count": "teacher_count_2025",
        "students_per_class": "students_per_class_2025",
        "students_per_teacher": "students_per_teacher_2025",
        "nearest_same_level_distance_km": "nearest_same_level_school_km",
        "same_level_school_count_within_5km": "same_level_school_count_5km",
    })
    
    # Apply visual flag logic
    web["small_school_flag_2025"] = web["student_count_2025"] <= 60
    web["small_school_flag_2030"] = web["pred_student_count_2030"] <= 60
    web["decline_pressure_flag_2030"] = (web["delta_2025_2030"] <= -30) | (web["pct_change_2025_2030"] <= -0.20)
    web["isolated_small_school_flag_2030"] = web["small_school_flag_2030"] & ((pd.to_numeric(web.get("same_level_school_count_5km"), errors="coerce") <= 1) | (pd.to_numeric(web.get("nearest_same_level_school_km"), errors="coerce") >= 5))
    
    q75 = pd.to_numeric(web.get("isolation_score"), errors="coerce").quantile(.75)
    web["education_gap_risk_flag_2030"] = web["small_school_flag_2030"] & (pd.to_numeric(web.get("isolation_score"), errors="coerce") >= q75)
    
    # Calculate priority scoring
    web["priority_score_2030"] = (
        0.30 * pct_rank((-web["delta_2025_2030"]).clip(lower=0)) +
        0.25 * pct_rank((-web["pct_change_2025_2030"]).clip(lower=0)) +
        0.25 * pct_rank(pd.to_numeric(web.get("isolation_score"), errors="coerce")) +
        0.10 * np.where(web["small_school_flag_2030"], 100, 0) +
        0.10 * (100 - pct_rank(pd.to_numeric(web.get("same_level_school_count_5km"), errors="coerce")))
    )
    
    web["priority_rank_national"] = web["priority_score_2030"].rank(method="first", ascending=False).astype(int)
    web["priority_rank_sido"] = web.groupby("sido")["priority_score_2030"].rank(method="first", ascending=False).astype(int)
    web["priority_rank_sgg"] = web.groupby(["sido", "sgg"])["priority_score_2030"].rank(method="first", ascending=False).astype(int)
    web["scenario_model_name"] = FINAL_MODEL
    web["scenario_type"] = "decline_pressure_scenario"
    web["scenario_note"] = "student count decline pressure scenario; visual prioritize aid only"
    
    required_cols = ["school_key", "school_name", "sido", "sgg", "school_level", "school_status_2025", "latitude", "longitude", "coordinate_valid", "coordinate_source", "coordinate_quality_flag", "student_count_2025"] + [f"pred_student_count_{y}" for y in range(2026, 2031)] + [f"delta_2025_{y}" for y in range(2026, 2031)] + [f"pct_change_2025_{y}" for y in range(2026, 2031)] + ["class_count_2025", "teacher_count_2025", "students_per_class_2025", "students_per_teacher_2025", "isolation_score", "nearest_same_level_school_km", "same_level_school_count_5km", "small_school_flag_2025", "small_school_flag_2030", "decline_pressure_flag_2030", "isolated_small_school_flag_2030", "education_gap_risk_flag_2030", "priority_score_2030", "priority_rank_national", "priority_rank_sido", "priority_rank_sgg", "scenario_model_name", "scenario_type", "scenario_note"]
    
    web = web[[c for c in required_cols if c in web.columns]].sort_values("priority_rank_national")
    
    # Save wide-format packages
    write_csv(web, output_dir / "final_scenario_school_web.csv")
    
    # Build long format
    long_rows = []
    for y in range(2025, 2031):
        tmp = web[["school_key", "school_name", "sido", "sgg", "school_level", "latitude", "longitude", "coordinate_valid", "isolation_score", "priority_score_2030", "scenario_model_name"]].copy()
        tmp["year"] = y
        tmp["student_count"] = web["student_count_2025"] if y == 2025 else web[f"pred_student_count_{y}"]
        tmp["is_observed"] = y == 2025
        tmp["is_predicted"] = y > 2025
        tmp["base_year"] = 2025
        tmp["delta_from_2025"] = 0.0 if y == 2025 else web[f"delta_2025_{y}"]
        tmp["pct_change_from_2025"] = 0.0 if y == 2025 else web[f"pct_change_2025_{y}"]
        long_rows.append(tmp)
        
    long = pd.concat(long_rows, ignore_index=True)
    write_csv(long, output_dir / "final_scenario_school_year_long.csv")
    
    # Generate aggregate summaries
    summary_cols = ["sido", "sgg", "school_level"]
    for i in range(len(summary_cols) + 1):
        grp = summary_cols[:i]
        sum_df = total_table(web, grp)
        name = "summary_national_by_year.csv" if not grp else f"summary_{'_'.join(grp)}_by_year.csv"
        write_csv(sum_df, output_dir / name)
        
    # Generate Top-K prioritizations
    top_priority = web.sort_values("priority_score_2030", ascending=False).head(100).copy()
    top_priority.insert(0, "rank", range(1, len(top_priority) + 1))
    write_csv(top_priority, output_dir / "top_priority_national_2030.csv")
    
    # Copy generated files to visual public folder
    for csv_file in output_dir.glob("*.csv"):
        copy_file(csv_file, public_dir / csv_file.name)
        
    # Calculate totals check
    actual_2030_total = float(total_final.loc[total_final["year"].eq(2030), "total_students"].iloc[0]) if not total_final.empty else 0.0
    rel_diff = abs(actual_2030_total - EXPECTED_2030_TOTAL) / EXPECTED_2030_TOTAL
    
    # Safety guard: Check if total matches within tolerance
    if rel_diff > 0.001:
        print(f"Aborted: 2030 total {actual_2030_total} does not match expected {EXPECTED_2030_TOTAL} within 0.001.")
        return {}

    val_df = pd.DataFrame([{
        "metric": "final_selected_p1_plus_event_total_2030",
        "expected_value": EXPECTED_2030_TOTAL,
        "actual_value": actual_2030_total,
        "absolute_diff": actual_2030_total - EXPECTED_2030_TOTAL,
        "relative_diff": rel_diff,
        "pass": bool(rel_diff <= 0.001)
    }])
    write_csv(val_df, output_dir / "scenario_total_validation.csv")
    
    return {
        "school_prediction_rows": len(web),
        "actual_2030_total": actual_2030_total,
        "relative_difference": rel_diff,
        "validation_pass": bool(rel_diff <= 0.001)
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Export Web Package")
    parser.add_argument("--recur-dir", type=str, default=str(RECURSIVE_FORECAST_DIR))
    parser.add_argument("--patch-dir", type=str, default=str(CLEAN_PATCH_DIR))
    parser.add_argument("--output-dir", type=str, default=str(WEB_PACKAGE_DIR))
    parser.add_argument("--public-dir", type=str, default=str(PUBLIC_ASSETS_DIR))
    args = parser.parse_args()
    
    run_export(Path(args.recur_dir), Path(args.patch_dir), Path(args.output_dir), Path(args.public_dir))
    print("Web scenario package exported successfully.")

if __name__ == "__main__":
    main()
