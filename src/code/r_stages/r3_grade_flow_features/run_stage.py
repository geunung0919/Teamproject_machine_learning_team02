from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.common.paths import CLEAN_PATCH_DIR, POLICY_COMP_DIR
from src.common.io import read_csv, write_csv
from src.common.metrics import calculate_metrics
from src.common.modeling import model_pipeline, rolling_folds
from src.features.feature_policy import infer_feature_columns

MODELS = ["LinearRegression", "Ridge", "RandomForestRegressor", "HistGradientBoostingRegressor"]

def run_r3(input_path: Path, output_dir: Path, horizon: int) -> pd.DataFrame:
    """Train and evaluate R3 grade/class flow modeling stage."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df = read_csv(input_path)
    
    target_delta = f"target_delta_{horizon}yr"
    target_student = f"target_student_count_{horizon}yr"
    target_avail = f"target_available_{horizon}yr"
    
    if target_avail in df.columns:
        df = df[df[target_avail].eq(True)].copy()
    else:
        df = df[df[target_student].notna()].copy()
        
    features = infer_feature_columns(df, "R3")
    
    metric_rows = []
    pred_rows = []
    
    for model_name in MODELS:
        for train, test, test_year in rolling_folds(df, horizon):
            pipe = model_pipeline(model_name, train, features)
            pipe.fit(train[features], train[target_delta])
            
            pred_delta = pipe.predict(test[features])
            pred_delta_before = pred_delta.copy()
            pred_students = np.maximum(0, test["student_count"].to_numpy(float) + pred_delta)
            
            actual_delta = test[target_delta].to_numpy(float)
            base_students = test["student_count"].to_numpy(float)
            actual_students = test[target_student].to_numpy(float)
            
            m = calculate_metrics(
                actual_delta=actual_delta,
                pred_delta=pred_delta,
                base_students=base_students,
                actual_students=actual_students,
                pred_before_clip=pred_delta_before,
                test_year=test_year
            )
            
            metric_rows.append({
                "stage": "R3",
                "horizon": horizon,
                "model": model_name,
                **m
            })
            
            keep = test[["school_key", "school_name", "year", "student_count", target_student]].copy()
            keep["pred_delta"] = pred_delta
            keep["pred_student_count"] = pred_students
            keep["model"] = model_name
            pred_rows.append(keep)
            
    metrics_df = pd.DataFrame(metric_rows)
    write_csv(metrics_df, output_dir / f"r3_grade_flow_{horizon}yr_metrics.csv")
    
    if pred_rows:
        write_csv(pd.concat(pred_rows, ignore_index=True), output_dir / f"r3_grade_flow_{horizon}yr_predictions.csv")
        
    return metrics_df

def main() -> None:
    parser = argparse.ArgumentParser(description="Run R3 Modeling")
    parser.add_argument("--input", type=str, default=str(CLEAN_PATCH_DIR / "model_views" / "r3_grade_flow_1yr.csv"))
    parser.add_argument("--output-dir", type=str, default=str(POLICY_COMP_DIR / "results"))
    parser.add_argument("--horizon", type=int, default=1)
    args = parser.parse_args()
    
    run_r3(Path(args.input), Path(args.output_dir), args.horizon)
    print(f"R3 modeling completed for horizon={args.horizon}yr.")

if __name__ == "__main__":
    main()
