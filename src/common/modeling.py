from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# Default Categorical fields
CATEGORICAL_HINTS = [
    "sido", "sgg", "school_level", "branch_type", "foundation_type",
    "region_group", "size_bucket", "student_size_bin", "metro_flag",
    "custom_size_bin", "level_size_segment", "school_level_x_size_bucket",
    "region_group_x_school_level", "sido_x_school_level"
]

def stringify_categoricals(x: Any) -> pd.DataFrame:
    """Safely convert categoricals to strings and fill missing values."""
    frame = pd.DataFrame(x).copy()
    return frame.astype("string").fillna("__missing__").astype(str)

def coerce_numeric(x: Any) -> pd.DataFrame:
    """Coerce columns to numeric values, converting errors to NaN."""
    frame = pd.DataFrame(x).copy()
    for col in frame.columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame

def make_preprocessor(df: pd.DataFrame, features: list[str], scale_numeric: bool, dense: bool = False) -> ColumnTransformer:
    """Generate ColumnTransformer for numeric and categorical pipelines."""
    categorical = [c for c in features if c in CATEGORICAL_HINTS or df[c].dtype == "object" or str(df[c].dtype) == "bool"]
    numeric = [c for c in features if c not in categorical]
    
    try:
        ohe = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=not dense)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=not dense)
        
    transformers = []
    
    if numeric:
        num_steps = [
            ("to_numeric", FunctionTransformer(coerce_numeric, validate=False)),
            ("imputer", SimpleImputer(strategy="median"))
        ]
        if scale_numeric:
            num_steps.append(("scaler", StandardScaler(with_mean=False if not dense else True)))
        transformers.append(("num", Pipeline(num_steps), numeric))
        
    if categorical:
        transformers.append((
            "cat", 
            Pipeline([
                ("stringify", FunctionTransformer(stringify_categoricals, validate=False)),
                ("onehot", ohe)
            ]), 
            categorical
        ))
        
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0 if dense else 0.3)

def base_estimator(name: str) -> Any:
    """Return configured model instances based on model name."""
    if name == "LinearRegression":
        return LinearRegression()
    if name == "Ridge":
        return Ridge(alpha=5.0)
    if name == "RandomForestRegressor":
        # Safe thread limit
        jobs = max(1, (os.cpu_count() or 1) // 2)
        return RandomForestRegressor(n_estimators=8, max_depth=8, min_samples_leaf=12, random_state=42, n_jobs=jobs)
    if name == "HistGradientBoostingRegressor":
        return HistGradientBoostingRegressor(max_iter=25, max_leaf_nodes=15, learning_rate=0.08, l2_regularization=0.01, random_state=42)
    raise ValueError(f"Unknown model name: {name}")

def model_pipeline(model_name: str, df: pd.DataFrame, features: list[str], multi: bool = False) -> Pipeline:
    """Build standard Pipeline combining preprocessing and estimator."""
    dense = model_name == "HistGradientBoostingRegressor"
    scale = model_name in ["LinearRegression", "Ridge"]
    est = base_estimator(model_name)
    
    if multi and model_name in ["LinearRegression", "Ridge", "HistGradientBoostingRegressor"]:
        est = MultiOutputRegressor(est)
        
    preprocess = make_preprocessor(df, features, scale_numeric=scale, dense=dense)
    return Pipeline([("preprocess", preprocess), ("model", est)])

def rolling_folds(df: pd.DataFrame, horizon: int) -> list[tuple[pd.DataFrame, pd.DataFrame, int]]:
    """Generate rolling temporal cross-validation folds (train: 2012~Y-1, test: Y)."""
    max_test_year = 2025 - horizon
    min_test_year = 2019
    folds = []
    for test_year in range(min_test_year, max_test_year + 1):
        train = df[df["year"].between(2012, test_year - 1)].copy()
        test = df[df["year"].eq(test_year)].copy()
        if len(train) and len(test):
            folds.append((train, test, test_year))
    return folds
