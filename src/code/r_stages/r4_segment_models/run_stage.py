from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.common.paths import CLEAN_PATCH_DIR, POLICY_COMP_DIR
from src.common.io import read_csv, write_csv
from src.common.metrics import calculate_metrics
from src.common.modeling import rolling_folds

def run_r4(input_path: Path, output_dir: Path, segment_col: str, horizon: int) -> pd.DataFrame:
    """Train and evaluate R4 segment models (e.g. region_group or size_bucket mean delta)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df = read_csv(input_path)
    
    target_delta = f"target_delta_{horizon}yr"
    target_student = f"target_student_count_{horizon}yr"
    target_avail = f"target_available_{horizon}yr"
    
    if target_avail in df.columns:
        df = df[df[target_avail].eq(True)].copy()
    else:
        df = df[df[target_student].notna()].copy()
        
    segment_rows = []
    
    # Validation rolling folds split
    for train, test, test_year in rolling_folds(df, horizon):
        if train.empty or test.empty:
            continue
            
        # Segment mean calculation
        means = train.groupby(segment_col)[target_delta].mean()
        global_mean = train[target_delta].mean()
        
        pred_delta = test[segment_col].map(means).fillna(global_mean).to_numpy(float)
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
        
        segment_rows.append({
            "stage": "R4",
            "horizon": horizon,
            "segment_column": segment_col,
            "model": "segment_mean_delta",
            **m
        })
        
    metrics_df = pd.DataFrame(segment_rows)
    write_csv(metrics_df, output_dir / f"r4_segment_{segment_col}_{horizon}yr_metrics.csv")
    return metrics_df

def main() -> None:
    parser = argparse.ArgumentParser(description="Run R4 Segment Models")
    parser.add_argument("--input", type=str, default=str(CLEAN_PATCH_DIR / "model_views" / "r4_region_group_1yr.csv"))
    parser.add_argument("--output-dir", type=str, default=str(POLICY_COMP_DIR / "results"))
    parser.add_argument("--segment-col", type=str, default="region_group")
    parser.add_argument("--horizon", type=int, default=1)
    args = parser.parse_args()
    
    run_r4(Path(args.input), Path(args.output_dir), args.segment_col, args.horizon)
    print(f"R4 segment modeling completed for segment={args.segment_col} horizon={args.horizon}yr.")

if __name__ == "__main__":
    main()
