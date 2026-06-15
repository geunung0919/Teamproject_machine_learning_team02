from __future__ import annotations

import pandas as pd

from src.common.feature_sets import (
    R1_BASE_COLUMNS, R2_ISO_COLUMNS, R3_GRADE_FLOW_COLUMNS, 
    R4_SEGMENT_COLUMNS, EXCLUDE_BASE
)
from src.common.validation import check_target_leakage

def get_stage_feature_policy(stage: str) -> dict:
    """Return feature set description, prerequisites, and blacklists for a stage."""
    policies = {
        "R0": {
            "name": "Persistence Baseline",
            "description": "Uses previous student count as projection. No learning.",
            "prerequisites": ["student_count"],
        },
        "R1": {
            "name": "Base School Features",
            "description": "Base student history and administrative/data features. No isolation/grade-flow.",
            "prerequisites": R1_BASE_COLUMNS,
        },
        "R2": {
            "name": "R1 + Spatial Isolation",
            "description": "R1 columns plus haversine school-level proximity metrics.",
            "prerequisites": R1_BASE_COLUMNS + R2_ISO_COLUMNS,
        },
        "R3": {
            "name": "R2 + Grade/Class Flows",
            "description": "R2 columns plus grade-wise student distributions and entrants/graduates.",
            "prerequisites": R1_BASE_COLUMNS + R2_ISO_COLUMNS + R3_GRADE_FLOW_COLUMNS,
        },
        "R4": {
            "name": "Segment Models",
            "description": "Splits by region and size segments. Uses R1 + segment definitions.",
            "prerequisites": R1_BASE_COLUMNS + R4_SEGMENT_COLUMNS,
        },
        "R5": {
            "name": "Cohort Proxy Trends",
            "description": "R3 columns plus aggregate trend proxies (sgg/sido/national growth slopes).",
            "prerequisites": R1_BASE_COLUMNS + R2_ISO_COLUMNS + R3_GRADE_FLOW_COLUMNS,
        },
        "R6": {
            "name": "Actual Cohort Pressure",
            "description": "R3 columns plus municipal-level cohort pressure ratios and demographics growth.",
            "prerequisites": R1_BASE_COLUMNS + R2_ISO_COLUMNS + R3_GRADE_FLOW_COLUMNS,
        }
    }
    
    clean_stage = stage.upper().split("_")[0]
    return policies.get(clean_stage, {"name": "Unknown", "description": "", "prerequisites": []})

def infer_feature_columns(df: pd.DataFrame, stage: str) -> list[str]:
    """Dynamically determine feature columns from DataFrame matching stage requirements."""
    policy = get_stage_feature_policy(stage)
    prereqs = policy["prerequisites"]
    
    # Filter features based on stage
    cols = []
    clean_stage = stage.upper().split("_")[0]
    
    if clean_stage == "R0":
        return ["student_count"]
        
    for c in df.columns:
        if c in EXCLUDE_BASE or c.startswith("target_"):
            continue
            
        # Standard filter checks
        if clean_stage == "R1":
            if c in R1_BASE_COLUMNS:
                cols.append(c)
        elif clean_stage == "R2":
            if c in R1_BASE_COLUMNS or c in R2_ISO_COLUMNS:
                cols.append(c)
        elif clean_stage == "R3":
            if c in R1_BASE_COLUMNS or c in R2_ISO_COLUMNS or c in R3_GRADE_FLOW_COLUMNS:
                cols.append(c)
        elif clean_stage == "R4":
            if c in R1_BASE_COLUMNS or c in R4_SEGMENT_COLUMNS:
                cols.append(c)
        elif clean_stage == "R5":
            # R3 plus R5 aggregate trends (contains _t or starts with school_share)
            is_r5_extra = "_t" in c or c.startswith("school_share_in_")
            if c in R1_BASE_COLUMNS or c in R2_ISO_COLUMNS or c in R3_GRADE_FLOW_COLUMNS or is_r5_extra:
                cols.append(c)
        elif clean_stage == "R6":
            # R3 plus R6 cohort pressure (contains cohort_pressure or pressure or fertility/birth growth)
            is_r6_extra = "cohort_pressure" in c or "pressure" in c or "growth" in c or "fertility" in c or "birth_count" in c
            # Avoid duplicating R1 fields
            if c in R1_BASE_COLUMNS or c in R2_ISO_COLUMNS or c in R3_GRADE_FLOW_COLUMNS or is_r6_extra:
                cols.append(c)
        else:
            # Fallback: keep if matches prereqs
            if c in prereqs:
                cols.append(c)
                
    # Double-check for data leakage
    leakage = check_target_leakage(df, cols)
    if leakage:
        cols = [c for c in cols if c not in leakage]
        
    return cols

def validate_feature_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Verify that columns exist in DataFrame and are free of data leakage."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Features missing from DataFrame: {missing}")
        
    leakage = check_target_leakage(df, columns)
    if leakage:
        raise ValueError(f"Target data leakage detected in features list: {leakage}")
