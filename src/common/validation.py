from __future__ import annotations

import re
from typing import Any
import numpy as np
import pandas as pd

def check_target_leakage(df: pd.DataFrame, features: list[str], horizon: int | None = None) -> list[str]:
    """Find any forbidden target or future leakage columns in the feature list."""
    forbidden_patterns = [
        r"student_delta_1yr", r"student_count_next", r"student_growth_1yr", r"target_year",
        r"target_missing_reason", r"next_year_", r"_next_year", r"_t_plus_", r"future_",
        r"forecast_", r"target_", r"closure_", r"closed_", r"risk_", r"priority_",
        r"survival_", r"label_", r"event_", r"status_after_", r"post_"
    ]
    
    # If a specific horizon is active, allow its target_available flag if needed,
    # but otherwise, target_* is strictly forbidden.
    leakage = []
    for col in features:
        # Check standard forbidden patterns
        for pat in forbidden_patterns:
            if re.search(pat, col.lower()):
                # Allow target_available_horizon flag or similar ONLY if explicitly bypassed
                if horizon is not None and f"target_available_{horizon}yr" in col:
                    continue
                leakage.append(col)
                break
    return leakage

def calculate_nonnull_rate(df: pd.DataFrame, cols: list[str]) -> float:
    """Calculate the average non-null rate across specified columns."""
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return np.nan
    return float(df[existing].notna().mean().mean())

def validate_model_view(df: pd.DataFrame, stage: str, horizon: int) -> dict[str, Any]:
    """Audit a stage view for rows, demographics, coordinates, leakage, and eligibility."""
    target_col = f"target_delta_{horizon}yr"
    avail_col = f"target_available_{horizon}yr"
    
    demo_cols = ["school_age_population_0_19", "age_0_4_pop", "age_5_9_pop", "age_10_14_pop", "age_15_19_pop"]
    birth_cols = ["birth_count", "total_fertility_rate"]
    mig_cols = ["net_migration_total", "in_migration_total", "out_migration_total"]
    iso_cols = ["nearest_same_level_distance_km", "isolation_score"]
    grade_cols = ["grade_student_sum", "entrants_total", "graduates_total"]
    
    demo_rate = calculate_nonnull_rate(df, demo_cols)
    birth_rate = calculate_nonnull_rate(df, birth_cols)
    mig_rate = calculate_nonnull_rate(df, mig_cols)
    
    iso_rate = calculate_nonnull_rate(df, iso_cols) if stage in {"R2", "R3", "R5", "R6"} else np.nan
    grade_rate = calculate_nonnull_rate(df, grade_cols) if stage in {"R3", "R5", "R6"} else np.nan
    
    target_avail_rate = float(df[avail_col].mean()) if avail_col in df.columns else 0.0
    
    features = [c for c in df.columns if c not in {target_col, avail_col, "school_key", "school_name"}]
    leakage = check_target_leakage(df, features, horizon)
    
    ready = len(leakage) == 0 and target_avail_rate >= 0.99
    if stage in {"R2", "R3", "R5", "R6"}:
        ready = ready and (pd.isna(iso_rate) or iso_rate >= 0.95)
    if stage in {"R3", "R5", "R6"}:
        ready = ready and (pd.isna(grade_rate) or grade_rate >= 0.95)
        
    return {
        "stage": stage,
        "horizon": horizon,
        "row_count": len(df),
        "column_count": len(df.columns),
        "demographics_nonnull_rate": demo_rate,
        "birth_fertility_nonnull_rate": birth_rate,
        "migration_nonnull_rate": mig_rate,
        "isolation_nonnull_rate": iso_rate,
        "grade_flow_nonnull_rate": grade_rate,
        "target_available_rate": target_avail_rate,
        "leakage_count": len(leakage),
        "leakage_columns": ",".join(leakage),
        "ready_for_training": ready
    }
