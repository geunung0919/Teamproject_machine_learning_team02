from __future__ import annotations

import pandas as pd
from src.features.feature_policy import infer_feature_columns

def get_r2_feature_columns(df: pd.DataFrame) -> list[str]:
    """Retrieve features for the R2 spatial isolation stage."""
    return infer_feature_columns(df, "R2")

def build_r2_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Filter dataset for R2 features and target columns."""
    features = get_r2_feature_columns(df)
    targets = [c for c in df.columns if c.startswith("target_")]
    meta = ["school_key", "school_name", "year"]
    keep = list(set(meta + features + targets))
    return df[[c for c in keep if c in df.columns]].copy()
