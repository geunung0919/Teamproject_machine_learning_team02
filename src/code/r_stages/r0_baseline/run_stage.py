from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.common.paths import CLEAN_PATCH_DIR, POLICY_COMP_DIR
from src.common.io import read_csv, write_csv
from src.common.metrics import calculate_metrics

def run_r0(input_path: Path, output_dir: Path, horizon: int) -> pd.DataFrame:
    """Run R0 baseline prediction (no ML, predict zero delta)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df = read_csv(input_path)
    
    target_delta = f"target_delta_{horizon}yr"
    target_student = f"target_student_count_{horizon}yr"
    target_avail = f"target_available_{horizon}yr"
    
    # Filter for standard eligible and target available records
    if target_avail in df.columns:
        df = df[df[target_avail].eq(True)].copy()
    else:
        df = df[df[target_student].notna()].copy()
        
    metric_rows = []
    # Loop over years for rolling temporal validation (2019 to 2025-horizon)
    max_test_year = 2025 - horizon
    for test_year in range(2019, max_test_year + 1):
        test = df[df["year"].eq(test_year)].copy()
        if test.empty:
            continue
            
        pred_delta = np.zeros(len(test))
        actual_delta = test[target_delta].to_numpy(float)
        base_students = test["student_count"].to_numpy(float)
        actual_students = test[target_student].to_numpy(float)
        
        m = calculate_metrics(
            actual_delta=actual_delta,
            pred_delta=pred_delta,
            base_students=base_students,
            actual_students=actual_students,
            pred_before_clip=pred_delta,
            test_year=test_year
        )
        
        metric_rows.append({
            "stage": "R0",
            "horizon": horizon,
            "model": "PersistenceDelta0",
            **m
        })
        
    metrics_df = pd.DataFrame(metric_rows)
    write_csv(metrics_df, output_dir / f"r0_baseline_{horizon}yr_metrics.csv")
    return metrics_df

def main() -> None:
    parser = argparse.ArgumentParser(description="Run R0 Baseline")
    parser.add_argument("--input", type=str, default=str(CLEAN_PATCH_DIR / "model_views" / "r0_baseline_1yr.csv"))
    parser.add_argument("--output-dir", type=str, default=str(POLICY_COMP_DIR / "results"))
    parser.add_argument("--horizon", type=int, default=1)
    args = parser.parse_args()
    
    run_r0(Path(args.input), Path(args.output_dir), args.horizon)
    print(f"R0 baseline completed for horizon={args.horizon}yr.")

if __name__ == "__main__":
    main()
