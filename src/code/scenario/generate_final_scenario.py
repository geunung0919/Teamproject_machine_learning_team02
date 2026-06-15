from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common.paths import CLEAN_PATCH_DIR, POLICY_COMP_DIR, RAW_DIR
from src.common.io import read_csv, write_csv, write_json
from src.common.modeling import model_pipeline
from src.features.feature_policy import infer_feature_columns

BEST_MODELS = {
    1: ("P1_event_excluded_decline_focus", "R3", "r3_grade_flow_direct_1yr.csv", 2026),
    2: ("P1_event_excluded_decline_focus", "R3", "r3_grade_flow_direct_2yr.csv", 2027),
    3: ("P1_event_excluded_decline_focus", "R3", "r3_grade_flow_direct_3yr.csv", 2028),
    4: ("P1_event_excluded_decline_focus", "R3", "r3_grade_flow_direct_4yr.csv", 2029),
    5: ("P1_event_excluded_decline_focus", "R3", "r3_grade_flow_direct_5yr.csv", 2030),
}

def generate_scenario(
    patch_dir: Path, direct_dir: Path, excluded_path: Path, output_dir: Path
) -> pd.DataFrame:
    """Train direct horizon models and generate 2026~2030 student count forecasts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    scenario_base_path = patch_dir / "model_views" / "scenario_base_2025.csv"
    if not scenario_base_path.exists():
        print("Scenario base file scenario_base_2025.csv not found.")
        return pd.DataFrame()
        
    scenario = read_csv(scenario_base_path)
    excluded = read_csv(excluded_path)
    excluded_keys = set(excluded["school_key"].astype(str))
    
    # Exclude event-flagged/unstable schools for main visualization base
    main_base = scenario[~scenario["school_key"].astype(str).isin(excluded_keys)].copy()
    main_base = main_base[main_base["scenario_base_eligible"].fillna(True).astype(bool)].copy()
    
    metadata_cols = ["school_key", "school_name", "sido", "sgg", "school_level", "size_bucket", "student_count"]
    pred = main_base[[c for c in metadata_cols if c in main_base.columns]].copy()
    pred = pred.rename(columns={"size_bucket": "size_bucket_2025", "student_count": "student_count_2025"})
    
    model_meta = {}
    
    for h, (policy, stage, fname, target_year) in BEST_MODELS.items():
        train_path = direct_dir / "model_views" / policy / fname
        if not train_path.exists():
            # Fallback path lookup in clean patch dir
            train_path = patch_dir / "model_views" / f"{stage.lower()}_basic_{h}yr.csv"
            if not train_path.exists():
                train_path = patch_dir / "model_views" / f"{stage.lower()}_grade_flow_{h}yr.csv"
                
        train = read_csv(train_path)
        if train.empty:
            continue
            
        target = f"target_delta_{h}yr"
        train = train[train[f"target_available_{h}yr"].fillna(False).astype(bool)].copy()
        
        features = infer_feature_columns(train, stage)
        
        # Safe alignment of features in scenario base
        for c in features:
            if c not in main_base.columns:
                main_base[c] = np.nan
                
        # Assemble pipeline using HistGradientBoostingRegressor (tuned select estimator)
        pipe = model_pipeline("HistGradientBoostingRegressor", train, features)
        # Update model parameters with optimal tuned parameters
        model_obj = pipe.named_steps["model"]
        model_obj.set_params(
            max_iter=120,
            max_leaf_nodes=31,
            learning_rate=0.04,
            l2_regularization=0.1,
            random_state=42
        )
        print(f"training HistGradientBoostingRegressor (tuned) for horizon={h}yr target={target_year}", flush=True)
        pipe.fit(train[features], train[target].astype(float))
        
        delta = pipe.predict(main_base[features])
        pred_col = f"pred_student_count_{target_year}"
        pred[pred_col] = np.maximum(0, pred["student_count_2025"].to_numpy(float) + delta)
        pred[f"horizon_model_{target_year}"] = f"direct_{h}yr|{policy}|{stage}|HistGradientBoostingRegressor"
        
        model_meta[str(target_year)] = {
            "horizon": h,
            "policy": policy,
            "stage": stage,
            "model": "HistGradientBoostingRegressor",
            "train_rows": len(train),
            "feature_count": len(features)
        }
        
    pred["delta_2025_2030"] = pred["pred_student_count_2030"] - pred["student_count_2025"]
    pred["pct_change_2025_2030"] = np.where(pred["student_count_2025"] > 0, pred["delta_2025_2030"] / pred["student_count_2025"], np.nan)
    pred["scenario_policy"] = "P1_event_excluded_decline_focus"
    pred["scenario_note"] = "event-excluded stable school decline pressure scenario"
    
    write_csv(pred, output_dir / "scenario_school_predictions_2026_2030_p1.csv")
    write_json(model_meta, output_dir / "scenario_model_metadata.json")
    
    return pred

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Final Scenario Generation")
    parser.add_argument("--patch-dir", type=str, default=str(CLEAN_PATCH_DIR))
    parser.add_argument("--direct-dir", type=str, default=str(POLICY_COMP_DIR))
    parser.add_argument("--excluded-schools", type=str, default=str(RAW_DIR.parent / "v5_p1_excluded_school_list_v1" / "p1_excluded_schools_2173.csv"))
    parser.add_argument("--output-dir", type=str, default=str(POLICY_COMP_DIR.parent / "v5_final_2026_2030_scenario_generation_v1"))
    args = parser.parse_args()
    
    generate_scenario(Path(args.patch_dir), Path(args.direct_dir), Path(args.excluded_schools), Path(args.output_dir))
    print("Final scenario generation completed.")

if __name__ == "__main__":
    main()
