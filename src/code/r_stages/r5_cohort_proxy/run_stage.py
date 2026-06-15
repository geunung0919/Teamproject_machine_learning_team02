from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.common.paths import CLEAN_PATCH_DIR, POLICY_COMP_DIR, COHORT_TRAIN_DIR
from src.common.io import read_csv, write_csv
from src.common.metrics import calculate_metrics
from src.common.modeling import model_pipeline, rolling_folds
from src.features.feature_policy import infer_feature_columns

MODELS = ["LinearRegression", "Ridge", "RandomForestRegressor", "HistGradientBoostingRegressor"]

def merge_r5_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merge R5 aggregate trend features into the baseline dataset."""
    r5_path = COHORT_TRAIN_DIR / "features" / "r5_aggregate_trend_features_by_school_year.csv"
    if not r5_path.exists():
        # Fallback to current features if missing
        return df
    r5 = read_csv(r5_path)
    # Exclude redundant geographic merge keys
    drop_cols = [c for c in ["sido", "sgg", "school_level"] if c in r5.columns]
    return df.merge(r5.drop(columns=drop_cols), on=["school_key", "year"], how="left")

def run_r5(input_path: Path, output_dir: Path, horizon: int) -> pd.DataFrame:
    """Train and evaluate R5 cohort proxy modeling stage."""
    output_dir.mkdir(parents=True, exist_ok=True)
    base_df = read_csv(input_path)
    
    # Merge trend features
    df = merge_r5_features(base_df)
    
    target_delta = f"target_delta_{horizon}yr"
    target_student = f"target_student_count_{horizon}yr"
    target_avail = f"target_available_{horizon}yr"
    
    if target_avail in df.columns:
        df = df[df[target_avail].eq(True)].copy()
    else:
        df = df[df[target_student].notna()].copy()
        
    features = infer_feature_columns(df, "R5")
    
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
                "stage": "R5",
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
    write_csv(metrics_df, output_dir / f"r5_cohort_proxy_{horizon}yr_metrics.csv")
    
    if pred_rows:
        write_csv(pd.concat(pred_rows, ignore_index=True), output_dir / f"r5_cohort_proxy_{horizon}yr_predictions.csv")
        
    return metrics_df

def main() -> None:
    parser = argparse.ArgumentParser(description="Run R5 Modeling")
    parser.add_argument("--input", type=str, default=str(CLEAN_PATCH_DIR / "model_views" / "r3_grade_flow_1yr.csv"))
    parser.add_argument("--output-dir", type=str, default=str(POLICY_COMP_DIR / "results"))
    parser.add_argument("--horizon", type=int, default=1)
    args = parser.parse_args()
    
    run_r5(Path(args.input), Path(args.output_dir), args.horizon)
    print(f"R5 modeling completed for horizon={args.horizon}yr.")

if __name__ == "__main__":
    main()
