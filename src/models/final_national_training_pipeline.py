from __future__ import annotations

from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import folium
import joblib
import numpy as np
import pandas as pd
from folium.plugins import MarkerCluster
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.neighbors import BallTree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from project_config import (
    RISK_COLORS,
    RISK_LABEL_KO,
    YEARS,
    assign_policy_risk_label,
    compute_policy_risk_score,
    normalize_school_level,
    valid_sido_coord_mask,
)


ROOT = SRC.parent
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "outputs" / "reports"
MODELS = ROOT / "outputs" / "models"
MAPS = ROOT / "outputs" / "maps"

EARTH_RADIUS_KM = 6371.0088


def minmax_score(series: pd.Series, reverse: bool = False) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    lo = values.quantile(0.05)
    hi = values.quantile(0.95)
    if pd.isna(lo) or pd.isna(hi) or hi <= lo:
        score = pd.Series(50.0, index=series.index)
    else:
        score = ((values - lo) / (hi - lo) * 100).clip(0, 100)
    if reverse:
        score = 100 - score
    return score.fillna(score.median()).round(1)


def safe_mape(actual: pd.Series, pred: np.ndarray, min_denominator: float = 100.0) -> float:
    actual_values = pd.to_numeric(actual, errors="coerce").to_numpy(dtype=float)
    denominator = np.maximum(np.abs(actual_values), min_denominator)
    return float(np.mean(np.abs(actual_values - pred) / denominator))


def latest_by_year(frame: pd.DataFrame, year_col: str, key_col: str, latest_year: int | None = None) -> pd.DataFrame:
    df = frame.copy()
    df[key_col] = df[key_col].astype(str).str.zfill(5)
    if latest_year is None:
        latest_year = int(df[year_col].max())
    latest = df[df[year_col].eq(latest_year)].copy()
    if latest.empty:
        latest = df.sort_values(year_col).groupby(key_col).tail(1)
    return latest.drop_duplicates(key_col)


def add_sido_fallback(base: pd.DataFrame, cols: list[str], key_col: str = "sgg_code") -> pd.DataFrame:
    out = base.copy()
    out["sido_code"] = out[key_col].astype(str).str[:2]
    for col in cols:
        sido_mean = out.groupby("sido_code")[col].transform("mean")
        global_median = out[col].median()
        out[col] = out[col].fillna(sido_mean).fillna(global_median)
    return out


def load_current_school_features() -> pd.DataFrame:
    schools = pd.read_csv(PROCESSED / "national_current_objective_closure_risk.csv", low_memory=False)
    schools["sgg_code"] = schools["sggCd"].astype(str).str[:5].str.zfill(5)
    schools["sido_code"] = schools["sgg_code"].str[:2]
    schools["lttud"] = pd.to_numeric(schools["lttud"], errors="coerce")
    schools["lgtud"] = pd.to_numeric(schools["lgtud"], errors="coerce")
    invalid_coord = ~valid_sido_coord_mask(schools)
    schools.loc[invalid_coord, ["lttud", "lgtud"]] = np.nan
    schools["student_count_2025"] = pd.to_numeric(schools["student_count_2025"], errors="coerce")

    pop = pd.read_csv(PROCESSED / "national_population_features_sgg.csv", low_memory=False)
    pop = pop[pop["month"].eq(4)].copy()
    pop_latest = latest_by_year(pop, "year", "sgg_code", 2026)
    pop_cols = [
        "sgg_code",
        "school_age_pop_0_19",
        "pop_0_4",
        "pop_5_9",
        "pop_10_14",
        "pop_15_19",
        "school_age_pop_mom_rate",
    ]
    schools = schools.merge(pop_latest[pop_cols], on="sgg_code", how="left")

    fertility = pd.read_csv(PROCESSED / "national_fertility_features_sgg.csv", low_memory=False)
    fert_latest = latest_by_year(fertility, "year", "region_code", 2023).rename(columns={"region_code": "sgg_code"})
    schools = schools.merge(
        fert_latest[["sgg_code", "total_fertility_rate", "tfr_yoy_change", "tfr_yoy_rate"]],
        on="sgg_code",
        how="left",
        suffixes=("", "_fertility"),
    )

    birth = pd.read_csv(PROCESSED / "national_birth_features_sgg.csv", low_memory=False)
    birth_latest = latest_by_year(birth, "year", "region_code", 2023).rename(columns={"region_code": "sgg_code"})
    schools = schools.merge(
        birth_latest[["sgg_code", "birth_count", "birth_count_yoy_rate"]],
        on="sgg_code",
        how="left",
    )

    migration = pd.read_csv(PROCESSED / "national_migration_features_sgg.csv", low_memory=False)
    mig_latest = latest_by_year(migration, "year", "region_code", 2025).rename(columns={"region_code": "sgg_code"})
    schools = schools.merge(
        mig_latest[
            [
                "sgg_code",
                "in_migration_total",
                "out_migration_total",
                "net_migration_total",
                "in_migration_yoy_rate",
                "out_migration_yoy_rate",
            ]
        ],
        on="sgg_code",
        how="left",
    )

    shop = pd.read_csv(PROCESSED / "national_small_shop_sgg_summary.csv", low_memory=False)
    shop["sgg_code"] = shop["signguCd"].astype(str).str.zfill(5)
    schools = schools.merge(
        shop[
            [
                "sgg_code",
                "commercial_count",
                "education_business_count",
                "kids_business_count",
                "medical_business_count",
            ]
        ],
        on="sgg_code",
        how="left",
    )

    radius_path = PROCESSED / "school_radius_commercial_features.csv"
    radius_cols = [
        "schlCd",
        "radius_0_5km_all_shops",
        "radius_0_5km_education_shops",
        "radius_0_5km_kids_shops",
        "radius_0_5km_medical_shops",
        "radius_1_0km_all_shops",
        "radius_1_0km_education_shops",
        "radius_1_0km_kids_shops",
        "radius_1_0km_medical_shops",
        "radius_2_0km_all_shops",
        "radius_2_0km_education_shops",
        "radius_2_0km_kids_shops",
        "radius_2_0km_medical_shops",
    ]
    if radius_path.exists():
        radius = pd.read_csv(radius_path, usecols=lambda c: c in radius_cols, low_memory=False)
        schools = schools.merge(radius, on="schlCd", how="left")

    fill_cols = [
        "school_age_pop_0_19",
        "pop_0_4",
        "pop_5_9",
        "pop_10_14",
        "pop_15_19",
        "school_age_pop_mom_rate",
        "total_fertility_rate",
        "tfr_yoy_change",
        "tfr_yoy_rate",
        "birth_count",
        "birth_count_yoy_rate",
        "in_migration_total",
        "out_migration_total",
        "net_migration_total",
        "in_migration_yoy_rate",
        "out_migration_yoy_rate",
        "commercial_count",
        "education_business_count",
        "kids_business_count",
        "medical_business_count",
        "radius_0_5km_all_shops",
        "radius_0_5km_education_shops",
        "radius_0_5km_kids_shops",
        "radius_0_5km_medical_shops",
        "radius_1_0km_all_shops",
        "radius_1_0km_education_shops",
        "radius_1_0km_kids_shops",
        "radius_1_0km_medical_shops",
        "radius_2_0km_all_shops",
        "radius_2_0km_education_shops",
        "radius_2_0km_kids_shops",
        "radius_2_0km_medical_shops",
    ]
    fill_cols = [col for col in fill_cols if col in schools.columns]
    schools = add_sido_fallback(schools, fill_cols)
    schools["net_migration_rate_proxy"] = schools["net_migration_total"] / schools["school_age_pop_0_19"].replace(0, np.nan)
    schools["commercial_per_1000_child"] = schools["commercial_count"] / schools["school_age_pop_0_19"].replace(0, np.nan) * 1000
    schools["education_per_1000_child"] = (
        schools["education_business_count"] / schools["school_age_pop_0_19"].replace(0, np.nan) * 1000
    )
    schools["kids_per_1000_child"] = schools["kids_business_count"] / schools["school_age_pop_0_19"].replace(0, np.nan) * 1000
    schools = add_isolation_features(schools)
    schools = add_vulnerability_scores(schools)
    return schools


def add_isolation_features(schools: pd.DataFrame) -> pd.DataFrame:
    out = schools.copy()
    out["nearest_same_level_school_km"] = np.nan
    out["same_level_school_count_5km"] = 0
    valid = out.dropna(subset=["lttud", "lgtud"])
    for level, idx in valid.groupby("school_level").groups.items():
        idx = list(idx)
        if len(idx) <= 1:
            continue
        coords = np.radians(out.loc[idx, ["lttud", "lgtud"]].to_numpy())
        tree = BallTree(coords, metric="haversine")
        dist, _ = tree.query(coords, k=min(2, len(idx)))
        out.loc[idx, "nearest_same_level_school_km"] = dist[:, 1] * EARTH_RADIUS_KM
        out.loc[idx, "same_level_school_count_5km"] = tree.query_radius(coords, r=5 / EARTH_RADIUS_KM, count_only=True) - 1
    nearest_score = minmax_score(out["nearest_same_level_school_km"])
    same_shortage = minmax_score(out["same_level_school_count_5km"], reverse=True)
    out["school_isolation_score"] = (nearest_score * 0.65 + same_shortage * 0.35).round(1)
    return out


def add_vulnerability_scores(schools: pd.DataFrame) -> pd.DataFrame:
    out = schools.copy()
    sgg_shop_shortage = minmax_score(out["commercial_per_1000_child"], reverse=True)
    sgg_edu_shortage = minmax_score(out["education_per_1000_child"], reverse=True)
    sgg_kids_shortage = minmax_score(out["kids_per_1000_child"], reverse=True)
    out["sgg_commercial_vulnerability_score"] = (
        sgg_shop_shortage * 0.5 + sgg_edu_shortage * 0.3 + sgg_kids_shortage * 0.2
    ).round(1)

    radius_required = [
        "radius_1_0km_all_shops",
        "radius_1_0km_education_shops",
        "radius_1_0km_kids_shops",
    ]
    if all(col in out.columns for col in radius_required):
        shop_shortage = minmax_score(np.log1p(out["radius_1_0km_all_shops"]), reverse=True)
        edu_shortage = minmax_score(np.log1p(out["radius_1_0km_education_shops"]), reverse=True)
        kids_shortage = minmax_score(np.log1p(out["radius_1_0km_kids_shops"]), reverse=True)
        out["commercial_vulnerability_source"] = "school_radius_1km_log"
    else:
        shop_shortage = sgg_shop_shortage
        edu_shortage = sgg_edu_shortage
        kids_shortage = sgg_kids_shortage
        out["commercial_vulnerability_source"] = "sgg_fallback"
    out["commercial_vulnerability_score"] = (shop_shortage * 0.5 + edu_shortage * 0.3 + kids_shortage * 0.2).round(1)

    commercial_report = pd.DataFrame(
        [
            {
                "score": "sgg_commercial_vulnerability_score",
                "source": "sgg_per_1000_child",
                "mean": out["sgg_commercial_vulnerability_score"].mean(),
                "median": out["sgg_commercial_vulnerability_score"].median(),
                "p70": out["sgg_commercial_vulnerability_score"].quantile(0.70),
                "p90": out["sgg_commercial_vulnerability_score"].quantile(0.90),
                "share_70_plus": (out["sgg_commercial_vulnerability_score"] >= 70).mean(),
            },
            {
                "score": "commercial_vulnerability_score",
                "source": out["commercial_vulnerability_source"].iloc[0],
                "mean": out["commercial_vulnerability_score"].mean(),
                "median": out["commercial_vulnerability_score"].median(),
                "p70": out["commercial_vulnerability_score"].quantile(0.70),
                "p90": out["commercial_vulnerability_score"].quantile(0.90),
                "share_70_plus": (out["commercial_vulnerability_score"] >= 70).mean(),
            },
        ]
    )
    commercial_report.to_csv(REPORTS / "commercial_vulnerability_score_distribution.csv", index=False, encoding="utf-8-sig")

    migration_risk = minmax_score(out["net_migration_rate_proxy"], reverse=True)
    birth_risk = minmax_score(out["birth_count_yoy_rate"], reverse=True)
    fertility_risk = minmax_score(out["total_fertility_rate"], reverse=True)
    out["regional_decline_risk_score"] = (migration_risk * 0.35 + birth_risk * 0.35 + fertility_risk * 0.30).round(1)
    return out


def train_population_regression() -> tuple[pd.DataFrame, pd.DataFrame, Pipeline, Pipeline]:
    pop = pd.read_csv(PROCESSED / "national_population_features_sgg.csv", low_memory=False)
    pop = pop[pop["month"].eq(4)].copy()
    birth = pd.read_csv(PROCESSED / "national_birth_features_sgg.csv", low_memory=False).rename(
        columns={"region_code": "sgg_code"}
    )
    mig = pd.read_csv(PROCESSED / "national_migration_features_sgg.csv", low_memory=False).rename(
        columns={"region_code": "sgg_code"}
    )
    shop = pd.read_csv(PROCESSED / "national_small_shop_sgg_summary.csv", low_memory=False)
    shop["sgg_code"] = shop["signguCd"].astype(str).str.zfill(5)

    pop["sgg_code"] = pop["sgg_code"].astype(str).str.zfill(5)
    birth["sgg_code"] = birth["sgg_code"].astype(str).str.zfill(5)
    mig["sgg_code"] = mig["sgg_code"].astype(str).str.zfill(5)
    panel = pop.merge(
        birth[["sgg_code", "year", "birth_count", "total_fertility_rate", "birth_count_yoy_rate", "tfr_yoy_rate"]],
        on=["sgg_code", "year"],
        how="left",
    ).merge(
        mig[
            [
                "sgg_code",
                "year",
                "in_migration_total",
                "out_migration_total",
                "net_migration_total",
                "in_migration_yoy_rate",
                "out_migration_yoy_rate",
            ]
        ],
        on=["sgg_code", "year"],
        how="left",
    ).merge(
        shop[["sgg_code", "commercial_count", "education_business_count", "kids_business_count", "medical_business_count"]],
        on="sgg_code",
        how="left",
    )
    panel = add_sido_fallback(
        panel,
        [
            "birth_count",
            "total_fertility_rate",
            "birth_count_yoy_rate",
            "tfr_yoy_rate",
            "in_migration_total",
            "out_migration_total",
            "net_migration_total",
            "in_migration_yoy_rate",
            "out_migration_yoy_rate",
            "commercial_count",
            "education_business_count",
            "kids_business_count",
            "medical_business_count",
        ],
    )
    panel = panel.sort_values(["sgg_code", "year"])
    panel["target_next_year_school_age_pop"] = panel.groupby("sgg_code")["school_age_pop_0_19"].shift(-1)
    panel["school_age_pop_growth_1yr"] = panel.groupby("sgg_code")["school_age_pop_0_19"].pct_change()
    panel["birth_count_lag6"] = panel.groupby("sgg_code")["birth_count"].shift(6)
    panel["birth_count_lag7"] = panel.groupby("sgg_code")["birth_count"].shift(7)
    train_data = panel.dropna(subset=["target_next_year_school_age_pop"]).copy()

    feature_cols = [
        "sgg_code",
        "sido_code",
        "year",
        "school_age_pop_0_19",
        "pop_0_4",
        "pop_5_9",
        "pop_10_14",
        "pop_15_19",
        "school_age_pop_growth_1yr",
        "birth_count",
        "total_fertility_rate",
        "birth_count_yoy_rate",
        "tfr_yoy_rate",
        "in_migration_total",
        "out_migration_total",
        "net_migration_total",
        "in_migration_yoy_rate",
        "out_migration_yoy_rate",
        "commercial_count",
        "education_business_count",
        "kids_business_count",
        "medical_business_count",
    ]
    cat_cols = ["sgg_code", "sido_code"]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    preprocessor = ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_cols,
            ),
        ]
    )
    models = {
        "ridge": Ridge(alpha=3.0),
        "random_forest": RandomForestRegressor(n_estimators=260, min_samples_leaf=3, random_state=42, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            learning_rate=0.04, max_iter=260, max_leaf_nodes=31, l2_regularization=0.05, random_state=42
        ),
    }
    rows = []
    test_mask = train_data["year"].between(2022, 2025)
    train_mask = train_data["year"].between(2016, 2021)
    actual_baseline = train_data.loc[test_mask, "target_next_year_school_age_pop"]
    pred_baseline = train_data.loc[test_mask, "school_age_pop_0_19"]
    rows.append(
        {
            "model": "baseline_previous_year_population",
            "target": "next_year_sgg_school_age_pop_0_19",
            "train_years": "none",
            "test_years": "2022-2025",
            "mae": mean_absolute_error(actual_baseline, pred_baseline),
            "rmse": mean_squared_error(actual_baseline, pred_baseline) ** 0.5,
            "mape_raw": mean_absolute_percentage_error(actual_baseline, pred_baseline),
            "mape_safe_denominator_100": safe_mape(actual_baseline, pred_baseline.to_numpy()),
            "r2": r2_score(actual_baseline, pred_baseline),
            "note": "naive baseline: next year equals current year population",
        }
    )
    best_name = ""
    best_score = float("inf")
    best_pipe: Pipeline | None = None
    for name, model in models.items():
        pipe = Pipeline([("prep", clone(preprocessor)), ("model", clone(model))])
        pipe.fit(train_data.loc[train_mask, feature_cols], train_data.loc[train_mask, "target_next_year_school_age_pop"])
        pred = pipe.predict(train_data.loc[test_mask, feature_cols])
        actual = train_data.loc[test_mask, "target_next_year_school_age_pop"]
        mae = mean_absolute_error(actual, pred)
        rows.append(
            {
                "model": name,
                "target": "next_year_sgg_school_age_pop_0_19",
                "train_years": "2016-2021",
                "test_years": "2022-2025",
                "mae": mae,
                "rmse": mean_squared_error(actual, pred) ** 0.5,
                "mape_raw": mean_absolute_percentage_error(actual, pred),
                "mape_safe_denominator_100": safe_mape(actual, pred),
                "r2": r2_score(actual, pred),
                "note": "trained model",
            }
        )
        if mae < best_score:
            best_name = name
            best_score = mae
            best_pipe = pipe

    assert best_pipe is not None

    cv_rows = []
    cv_model = RandomForestRegressor(n_estimators=260, min_samples_leaf=3, random_state=42, n_jobs=-1)
    for fold, (train_start, train_end, test_year) in enumerate(
        [(2014, 2019, 2020), (2014, 2020, 2021), (2014, 2021, 2022), (2014, 2022, 2023)],
        start=1,
    ):
        fold_train = train_data["year"].between(train_start, train_end)
        fold_test = train_data["year"].eq(test_year)
        if not fold_train.any() or not fold_test.any():
            continue
        fold_pipe = Pipeline([("prep", clone(preprocessor)), ("model", clone(cv_model))])
        fold_pipe.fit(train_data.loc[fold_train, feature_cols], train_data.loc[fold_train, "target_next_year_school_age_pop"])
        fold_actual = train_data.loc[fold_test, "target_next_year_school_age_pop"]
        fold_pred = fold_pipe.predict(train_data.loc[fold_test, feature_cols])
        cv_rows.append(
            {
                "fold": fold,
                "train_years": f"{train_start}-{train_end}",
                "test_year": test_year,
                "mae": mean_absolute_error(fold_actual, fold_pred),
                "rmse": mean_squared_error(fold_actual, fold_pred) ** 0.5,
                "r2": r2_score(fold_actual, fold_pred),
                "safe_mape": safe_mape(fold_actual, fold_pred),
            }
        )
    if cv_rows:
        cv = pd.DataFrame(cv_rows)
        summary = {
            "fold": "mean",
            "train_years": "expanding",
            "test_year": "",
            "mae": cv["mae"].mean(),
            "rmse": cv["rmse"].mean(),
            "r2": cv["r2"].mean(),
            "safe_mape": cv["safe_mape"].mean(),
        }
        std = {
            "fold": "std",
            "train_years": "expanding",
            "test_year": "",
            "mae": cv["mae"].std(ddof=0),
            "rmse": cv["rmse"].std(ddof=0),
            "r2": cv["r2"].std(ddof=0),
            "safe_mape": cv["safe_mape"].std(ddof=0),
        }
        pd.concat([cv, pd.DataFrame([summary, std])], ignore_index=True).to_csv(
            REPORTS / "regression_time_cv_results.csv", index=False, encoding="utf-8-sig"
        )

    best_pred = best_pipe.predict(train_data.loc[test_mask, feature_cols])
    size_eval = train_data.loc[test_mask, ["sgg_code", "school_age_pop_0_19", "target_next_year_school_age_pop"]].copy()
    size_eval["model_pred"] = best_pred
    size_eval["baseline_pred"] = size_eval["school_age_pop_0_19"]
    low_cut = size_eval["school_age_pop_0_19"].quantile(0.33)
    high_cut = size_eval["school_age_pop_0_19"].quantile(0.67)
    size_eval["size_group"] = np.select(
        [
            size_eval["school_age_pop_0_19"] < low_cut,
            size_eval["school_age_pop_0_19"] >= high_cut,
        ],
        ["small_sgg", "large_sgg"],
        default="middle_sgg",
    )
    size_rows = []
    for group, group_df in size_eval.groupby("size_group"):
        actual_group = group_df["target_next_year_school_age_pop"]
        model_pred = group_df["model_pred"].to_numpy()
        baseline_pred = group_df["baseline_pred"].to_numpy()
        model_mae = mean_absolute_error(actual_group, model_pred)
        baseline_mae = mean_absolute_error(actual_group, baseline_pred)
        size_rows.append(
            {
                "size_group": group,
                "sgg_year_rows": len(group_df),
                "avg_population": group_df["school_age_pop_0_19"].mean(),
                "model_name": best_name,
                "model_mae": model_mae,
                "baseline_mae": baseline_mae,
                "mae_improvement_pct": (1 - model_mae / baseline_mae) * 100 if baseline_mae else np.nan,
                "model_safe_mape": safe_mape(actual_group, model_pred),
                "baseline_safe_mape": safe_mape(actual_group, baseline_pred),
            }
        )
    pd.DataFrame(size_rows).sort_values("avg_population").to_csv(
        REPORTS / "regression_performance_by_sgg_size.csv", index=False, encoding="utf-8-sig"
    )

    change_feature_cols = [col for col in feature_cols if col != "school_age_pop_0_19"]
    change_cat_cols = [col for col in cat_cols if col in change_feature_cols]
    change_num_cols = [col for col in change_feature_cols if col not in change_cat_cols]
    change_preprocessor = ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), change_num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                change_cat_cols,
            ),
        ]
    )
    train_data["target_pop_change"] = (
        train_data["target_next_year_school_age_pop"] - train_data["school_age_pop_0_19"]
    )
    change_pipe = Pipeline(
        [
            ("prep", change_preprocessor),
            ("model", RandomForestRegressor(n_estimators=260, min_samples_leaf=3, random_state=42, n_jobs=-1)),
        ]
    )
    change_pipe.fit(train_data.loc[train_mask, change_feature_cols], train_data.loc[train_mask, "target_pop_change"])
    change_actual = train_data.loc[test_mask, "target_pop_change"]
    change_pred = change_pipe.predict(train_data.loc[test_mask, change_feature_cols])
    rows.append(
        {
            "model": "change_target_baseline_zero",
            "target": "next_year_sgg_school_age_pop_change",
            "train_years": "none",
            "test_years": "2022-2025",
            "mae": mean_absolute_error(change_actual, np.zeros(len(change_actual))),
            "rmse": mean_squared_error(change_actual, np.zeros(len(change_actual))) ** 0.5,
            "mape_raw": np.nan,
            "mape_safe_denominator_100": safe_mape(change_actual, np.zeros(len(change_actual))),
            "r2": r2_score(change_actual, np.zeros(len(change_actual))),
            "note": "change target experiment: zero annual change baseline",
        }
    )
    rows.append(
        {
            "model": "change_target_random_forest",
            "target": "next_year_sgg_school_age_pop_change",
            "train_years": "2016-2021",
            "test_years": "2022-2025",
            "mae": mean_absolute_error(change_actual, change_pred),
            "rmse": mean_squared_error(change_actual, change_pred) ** 0.5,
            "mape_raw": np.nan,
            "mape_safe_denominator_100": safe_mape(change_actual, change_pred),
            "r2": r2_score(change_actual, change_pred),
            "note": "change target experiment: predict delta instead of absolute population",
        }
    )
    if hasattr(change_pipe.named_steps["model"], "feature_importances_"):
        transformed_names = change_pipe.named_steps["prep"].get_feature_names_out()
        change_importance = pd.DataFrame(
            {
                "feature": transformed_names,
                "importance": change_pipe.named_steps["model"].feature_importances_,
            }
        ).sort_values("importance", ascending=False)
        change_importance.head(30).to_csv(
            REPORTS / "change_target_regression_feature_importance.csv", index=False, encoding="utf-8-sig"
        )

    final_pipe = Pipeline([("prep", clone(preprocessor)), ("model", clone(models[best_name]))])
    final_pipe.fit(train_data[feature_cols], train_data["target_next_year_school_age_pop"])
    joblib.dump(final_pipe, MODELS / "final_national_sgg_population_regressor.pkl")

    final_change_pipe = Pipeline(
        [
            ("prep", clone(change_preprocessor)),
            ("model", RandomForestRegressor(n_estimators=260, min_samples_leaf=3, random_state=42, n_jobs=-1)),
        ]
    )
    final_change_pipe.fit(train_data[change_feature_cols], train_data["target_pop_change"])
    joblib.dump(final_change_pipe, MODELS / "final_national_sgg_population_change_regressor.pkl")

    change_clip_report = train_data["school_age_pop_growth_1yr"].describe(
        percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]
    )
    change_clip_report.rename("school_age_pop_growth_1yr").to_csv(
        REPORTS / "population_change_rate_distribution.csv", encoding="utf-8-sig"
    )

    metrics = pd.DataFrame(rows).sort_values(["target", "mae"])
    metrics.to_csv(REPORTS / "final_national_population_regression_metrics.csv", index=False, encoding="utf-8-sig")
    return panel, metrics, final_pipe, final_change_pipe


def forecast_sgg_population(panel: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
    latest_static = panel.sort_values("year").groupby("sgg_code").tail(1).copy()
    baseline_population = latest_static.set_index("sgg_code")["school_age_pop_0_19"].replace(0, np.nan)
    rows = []
    current = latest_static.copy()
    for year in YEARS:
        current["year"] = year - 1
        current["school_age_pop_growth_1yr"] = current.get("school_age_pop_growth_1yr", 0).fillna(0)
        feature_cols = list(model.named_steps["prep"].feature_names_in_)
        pred = model.predict(current[feature_cols]).clip(min=1)
        out = current[["sgg_code", "sgg_name", "sido_code"]].copy()
        out["forecast_year"] = year
        out["forecast_school_age_pop_0_19"] = pred
        baseline = out["sgg_code"].map(baseline_population)
        out["population_pressure_ratio"] = (pred / baseline).clip(0.2, 1.6)
        rows.append(out)
        current["school_age_pop_growth_1yr"] = (pred - current["school_age_pop_0_19"]) / current[
            "school_age_pop_0_19"
        ].replace(0, np.nan)
        current["school_age_pop_0_19"] = pred
    forecast = pd.concat(rows, ignore_index=True)
    forecast.to_csv(PROCESSED / "final_national_sgg_population_forecast_2026_2040.csv", index=False, encoding="utf-8-sig")
    return forecast


def forecast_sgg_population_by_change(panel: pd.DataFrame, change_model: Pipeline) -> pd.DataFrame:
    """Autoregressive forecast that predicts annual population delta, then adds it to current population.

    The annual delta is clipped to prevent unrealistic long-run explosion:
    a local school-age population is capped at -20% decline and +10% growth per year.
    """
    latest_static = panel.sort_values("year").groupby("sgg_code").tail(1).copy()
    baseline_population = latest_static.set_index("sgg_code")["school_age_pop_0_19"].replace(0, np.nan)
    change_feature_cols = list(change_model.named_steps["prep"].feature_names_in_)
    rows = []
    current = latest_static.copy()
    for year in YEARS:
        current["year"] = year - 1
        current["school_age_pop_growth_1yr"] = current.get("school_age_pop_growth_1yr", 0).fillna(0)
        current_pop = pd.to_numeric(current["school_age_pop_0_19"], errors="coerce").fillna(0).to_numpy(dtype=float)
        delta = change_model.predict(current[change_feature_cols])

        max_decrease = current_pop * -0.20
        max_increase = current_pop * 0.10
        delta = np.clip(delta, max_decrease, max_increase)
        pred = np.clip(current_pop + delta, 1, None)

        out = current[["sgg_code", "sgg_name", "sido_code"]].copy()
        out["forecast_year"] = year
        out["forecast_school_age_pop_0_19"] = pred
        baseline = out["sgg_code"].map(baseline_population)
        out["population_pressure_ratio"] = (pred / baseline).clip(0.2, 1.6)
        rows.append(out)

        previous_pop = current["school_age_pop_0_19"].replace(0, np.nan)
        current["school_age_pop_growth_1yr"] = delta / previous_pop
        current["school_age_pop_0_19"] = pred

    forecast = pd.concat(rows, ignore_index=True)
    forecast.to_csv(
        PROCESSED / "final_national_sgg_population_forecast_change_model_2026_2040.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return forecast


def build_final_school_scenario(
    schools: pd.DataFrame,
    sgg_forecast: pd.DataFrame,
    output_path: Path | None = None,
) -> pd.DataFrame:
    base = schools.copy()
    scenarios = []
    sensitivity = {"초등학교": 1.10, "중학교": 1.00, "고등학교": 0.90, "특수학교": 0.80, "각종학교": 0.85}
    thresholds = {"초등학교": 120, "중학교": 150, "고등학교": 180, "특수학교": 40, "각종학교": 60}
    for year in YEARS:
        yf = sgg_forecast[sgg_forecast["forecast_year"].eq(year)][
            ["sgg_code", "forecast_school_age_pop_0_19", "population_pressure_ratio"]
        ]
        df = base.merge(yf, on="sgg_code", how="left")
        df["forecast_year"] = year
        df["population_pressure_ratio"] = df["population_pressure_ratio"].fillna(1.0)
        df["school_level_sensitivity"] = df["school_level"].map(sensitivity).fillna(1.0)
        df["forecast_student_count"] = (
            df["student_count_2025"].fillna(0) * (df["population_pressure_ratio"] ** df["school_level_sensitivity"])
        ).round().clip(lower=0)
        df["low_student_threshold"] = df["school_level"].map(thresholds).fillna(100)
        df["pred_low_student_flag"] = (df["forecast_student_count"] <= df["low_student_threshold"]).astype(int)
        df["long_term_decline_flag"] = (df["population_pressure_ratio"] <= 0.85).astype(int)
        df["severe_decline_flag"] = (df["population_pressure_ratio"] <= 0.70).astype(int)
        df["replacement_near_flag"] = (
            (df["nearest_same_level_school_km"] <= 3) & (df["same_level_school_count_5km"] >= 2)
        ).astype(int)
        df["objective_top10_flag"] = (df["objective_closure_percentile"] >= 90).astype(int)
        df["isolation_high_flag"] = (df["school_isolation_score"] >= 70).astype(int)
        df["commercial_vulnerable_flag"] = (df["commercial_vulnerability_score"] >= 70).astype(int)
        df["regional_decline_high_flag"] = (df["regional_decline_risk_score"] >= 70).astype(int)
        df["risk_score"] = compute_policy_risk_score(df)
        df["differentiation_score"] = (
            df["school_isolation_score"] * 0.25
            + df["commercial_vulnerability_score"] * 0.20
            + df["regional_decline_risk_score"] * 0.25
            + df["objective_closure_percentile"] * 0.15
            + (1 - df["population_pressure_ratio"]).clip(0, 1) * 100 * 0.15
        ).round(1)

        df["risk_label"] = df.apply(assign_policy_risk_label, axis=1)
        scenarios.append(df)
    scenario = pd.concat(scenarios, ignore_index=True)
    save_path = output_path or (PROCESSED / "final_national_school_scenario_2026_2040.csv")
    scenario.to_csv(save_path, index=False, encoding="utf-8-sig")
    return scenario


def train_objective_classifier_with_context() -> pd.DataFrame:
    # Keeps the EDSS proxy as an auxiliary similarity feature audit, not as the
    # final classification model performance.
    existing = pd.read_csv(REPORTS / "auxiliary_edss_similarity_model_metrics.csv")
    existing["model_note"] = (
        "auxiliary EDSS similarity feature generator; final classification "
        "performance is reported by train_temporal_closure_classifier.py"
    )
    existing.to_csv(REPORTS / "edss_auxiliary_similarity_feature_audit.csv", index=False, encoding="utf-8-sig")
    return existing


def build_map(scenario: pd.DataFrame) -> None:
    valid = scenario[scenario["forecast_year"].eq(2040)].dropna(subset=["lttud", "lgtud"]).copy()
    valid["lttud"] = pd.to_numeric(valid["lttud"], errors="coerce")
    valid["lgtud"] = pd.to_numeric(valid["lgtud"], errors="coerce")
    valid = valid[valid_sido_coord_mask(valid)].copy()
    fmap = folium.Map(location=[36.3, 127.8], zoom_start=7, tiles="cartodbpositron")
    title = """
    <div style="position: fixed; top: 12px; left: 50px; z-index: 9999; background: white;
                padding: 12px 14px; border: 1px solid #d1d5db; border-radius: 6px;
                box-shadow: 0 2px 8px rgba(15,23,42,.12); font-family: Arial, 'Malgun Gothic', sans-serif;">
      <div style="font-weight:700;font-size:15px;">최종 전국 학교 통폐합 위험 시나리오 2040</div>
      <div style="font-size:12px;color:#475569;margin-top:4px;">인구·출산·출생·인구이동·상권·학교고립도·EDSS 객관점수 결합</div>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(title))
    for risk, group in valid.groupby("risk_label"):
        layer = folium.FeatureGroup(name=f"{RISK_LABEL_KO.get(risk, risk)} ({len(group):,})", show=True)
        cluster = MarkerCluster(disableClusteringAtZoom=10)
        layer.add_child(cluster)
        color = RISK_COLORS.get(risk, "#64748b")
        for _, row in group.iterrows():
            html = f"""
            <b>{row['schlNm']}</b><br>
            지역: {row['requested_sido_name']} / {row['sgg_code']}<br>
            학교급: {row['school_level']}<br>
            2025 학생수: {row['student_count_2025']:.0f}명<br>
            2040 예측 학생수: {row['forecast_student_count']:.0f}명<br>
            학령인구 압력비: {row['population_pressure_ratio']:.2f}<br>
            위험등급: {RISK_LABEL_KO.get(row['risk_label'], row['risk_label'])}<br>
            위험점수: {row['risk_score']:.0f}<br>
            학교 고립도: {row['school_isolation_score']:.1f}<br>
            상권 취약도: {row['commercial_vulnerability_score']:.1f}<br>
            지역 감소위험: {row['regional_decline_risk_score']:.1f}<br>
            EDSS 객관 백분위: {row['objective_closure_percentile']:.1f}%<br>
            최근접 같은 학교급: {row['nearest_same_level_school_km']:.2f}km
            """
            folium.CircleMarker(
                location=[row["lttud"], row["lgtud"]],
                radius=4 if row["risk_score"] < 60 else 7,
                color=color,
                weight=1,
                fill=True,
                fill_color=color,
                fill_opacity=0.76,
                tooltip=f"{row['requested_sido_name']} {row['schlNm']} | {RISK_LABEL_KO.get(risk, risk)}",
                popup=folium.Popup(html, max_width=380),
            ).add_to(cluster)
        layer.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    output = MAPS / "final_national_school_risk_2040.html"
    fmap.save(output)
    print("saved map:", output)


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    MAPS.mkdir(parents=True, exist_ok=True)

    schools = load_current_school_features()
    schools.to_csv(PROCESSED / "final_national_current_school_features.csv", index=False, encoding="utf-8-sig")
    match_report = pd.DataFrame(
        [
            {"feature": col, "missing_rate": float(schools[col].isna().mean())}
            for col in [
                "school_age_pop_0_19",
                "total_fertility_rate",
                "birth_count",
                "net_migration_total",
                "commercial_count",
                "school_isolation_score",
            ]
        ]
    )
    match_report.to_csv(REPORTS / "final_national_feature_match_report.csv", index=False, encoding="utf-8-sig")

    pop_panel, pop_metrics, pop_model, change_model = train_population_regression()
    sgg_forecast = forecast_sgg_population(pop_panel, pop_model)
    sgg_forecast_change = forecast_sgg_population_by_change(pop_panel, change_model)
    build_final_school_scenario(
        schools,
        sgg_forecast_change,
        PROCESSED / "final_national_school_scenario_change_model_2026_2040.csv",
    )
    scenario = build_final_school_scenario(schools, sgg_forecast)
    # 최종 룰 수식을 100% 진짜 머신러닝 예측값(tuned_histgb_policy_multiclass_classifier)으로 대체
    from models.train_policy_multiclass_classifier import train_policy_multiclass_model
    policy_model = train_policy_multiclass_model()
    policy_le = joblib.load(MODELS / "policy_multiclass_label_encoder.pkl")
    feature_cols = [
        "requested_sido_name",
        "school_level",
        "foundation",
        "student_count_2025",
        "forecast_student_count",
        "population_pressure_ratio",
        "school_isolation_score",
        "commercial_vulnerability_score",
        "regional_decline_risk_score",
        "objective_closure_percentile",
        "nearest_same_level_school_km",
        "same_level_school_count_5km",
        "risk_score"
    ]

    # 기본 시나리오 ML 예측 적용
    scenario_clean = scenario[scenario["school_level"].isin(["초등학교", "중학교", "고등학교"])].copy()
    pred_encoded = policy_model.predict(scenario_clean[feature_cols])
    scenario_clean["risk_label"] = policy_le.inverse_transform(pred_encoded)
    scenario_clean.to_csv(PROCESSED / "final_national_school_scenario_2026_2040.csv", index=False, encoding="utf-8-sig")
    scenario = scenario_clean

    # 변화량 시나리오 ML 예측 적용
    change_path = PROCESSED / "final_national_school_scenario_change_model_2026_2040.csv"
    if change_path.exists():
        change_scenario = pd.read_csv(change_path, low_memory=False)
        change_clean = change_scenario[change_scenario["school_level"].isin(["초등학교", "중학교", "고등학교"])].copy()
        pred_change = policy_model.predict(change_clean[feature_cols])
        change_clean["risk_label"] = policy_le.inverse_transform(pred_change)
        change_clean.to_csv(change_path, index=False, encoding="utf-8-sig")

    train_objective_classifier_with_context()

    summary = (
        scenario.groupby(["forecast_year", "requested_sido_name", "risk_label"], as_index=False)
        .agg(
            schools=("schlCd", "count"),
            avg_forecast_student_count=("forecast_student_count", "mean"),
            avg_risk_score=("risk_score", "mean"),
            avg_isolation=("school_isolation_score", "mean"),
            avg_commercial_vulnerability=("commercial_vulnerability_score", "mean"),
            avg_regional_decline=("regional_decline_risk_score", "mean"),
        )
    )
    summary.to_csv(REPORTS / "final_national_scenario_summary_2026_2040.csv", index=False, encoding="utf-8-sig")
    top = scenario.sort_values(["forecast_year", "risk_score", "differentiation_score"], ascending=[True, False, False])
    top.groupby("forecast_year").head(300).to_csv(
        REPORTS / "final_national_top_risk_schools_2026_2040.csv", index=False, encoding="utf-8-sig"
    )
    build_map(scenario)
    print("current school features:", len(schools))
    print("scenario rows:", len(scenario))
    print(pop_metrics.to_string(index=False))
    print(match_report.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

