from __future__ import annotations

from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = SRC.parent
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "outputs" / "reports"
MODELS = ROOT / "outputs" / "models"


SIDO_CODE_TO_NAME = {
    "11": "서울",
    "26": "부산",
    "27": "대구",
    "28": "인천",
    "29": "광주",
    "30": "대전",
    "31": "울산",
    "36": "세종",
    "41": "경기",
    "42": "강원",
    "51": "강원",
    "43": "충북",
    "44": "충남",
    "45": "전북",
    "52": "전북",
    "46": "전남",
    "47": "경북",
    "48": "경남",
    "50": "제주",
}


def load_panel_with_next_year_label() -> pd.DataFrame:
    panel = pd.read_csv(PROCESSED / "edss_national_school_panel_2009_2023.csv", low_memory=False)
    panel["base_year"] = pd.to_numeric(panel["year"], errors="coerce").astype("Int64")
    panel["sido_name"] = panel["시도명"].astype(str)
    max_year = int(panel["base_year"].max())
    observed = panel.groupby("school_id")["base_year"].apply(set).to_dict()
    panel["closed_next_year_proxy"] = panel.apply(
        lambda row: np.nan
        if int(row["base_year"]) >= max_year
        else int((int(row["base_year"]) + 1) not in observed.get(row["school_id"], set())),
        axis=1,
    )
    return panel[panel["base_year"].between(2009, 2023)].copy()


def build_sido_context() -> pd.DataFrame:
    birth = pd.read_csv(PROCESSED / "national_birth_features_sgg.csv", low_memory=False)
    birth["sido_code"] = birth["sido_code"].astype(str).str.zfill(2)
    birth["sido_name"] = birth["sido_code"].map(SIDO_CODE_TO_NAME)
    birth_ctx = (
        birth.groupby(["sido_name", "year"], as_index=False)
        .agg(
            birth_count=("birth_count", "sum"),
            avg_total_fertility_rate=("total_fertility_rate", "mean"),
            avg_birth_count_yoy_rate=("birth_count_yoy_rate", "mean"),
            avg_tfr_yoy_rate=("tfr_yoy_rate", "mean"),
        )
    )

    migration = pd.read_csv(PROCESSED / "national_migration_features_sgg.csv", low_memory=False)
    migration["sido_code"] = migration["sido_code"].astype(str).str.zfill(2)
    migration["sido_name"] = migration["sido_code"].map(SIDO_CODE_TO_NAME)
    migration_ctx = (
        migration.groupby(["sido_name", "year"], as_index=False)
        .agg(
            in_migration_total=("in_migration_total", "sum"),
            out_migration_total=("out_migration_total", "sum"),
            net_migration_total=("net_migration_total", "sum"),
            avg_in_migration_yoy_rate=("in_migration_yoy_rate", "mean"),
            avg_out_migration_yoy_rate=("out_migration_yoy_rate", "mean"),
        )
    )

    shop = pd.read_csv(PROCESSED / "national_small_shop_sgg_summary.csv", low_memory=False)
    shop["ctprvnCd"] = shop["ctprvnCd"].astype(str).str.zfill(2)
    shop["sido_name"] = shop["ctprvnCd"].map(SIDO_CODE_TO_NAME)
    shop_ctx = (
        shop.groupby("sido_name", as_index=False)
        .agg(
            commercial_count=("commercial_count", "sum"),
            education_business_count=("education_business_count", "sum"),
            kids_business_count=("kids_business_count", "sum"),
            medical_business_count=("medical_business_count", "sum"),
        )
    )
    return birth_ctx.merge(migration_ctx, on=["sido_name", "year"], how="outer").merge(
        shop_ctx, on="sido_name", how="left"
    )


def build_temporal_dataset() -> pd.DataFrame:
    panel = load_panel_with_next_year_label()
    context = build_sido_context()
    merged = panel.merge(context, left_on=["sido_name", "base_year"], right_on=["sido_name", "year"], how="left")
    context_cols = [c for c in context.columns if c not in ["sido_name", "year"]]
    for col in context_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
        merged[col] = merged[col].fillna(merged.groupby("sido_name")[col].transform("median")).fillna(
            merged[col].median()
        )
    merged["commercial_per_birth"] = merged["commercial_count"] / merged["birth_count"].replace(0, np.nan)
    merged["education_per_birth"] = merged["education_business_count"] / merged["birth_count"].replace(0, np.nan)
    merged["net_migration_per_birth"] = merged["net_migration_total"] / merged["birth_count"].replace(0, np.nan)
    return merged.replace([np.inf, -np.inf], np.nan)


def train_and_score(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str], model_name: str, model) -> tuple:
    target = "closed_next_year_proxy"
    categorical = ["sido_name", "학제명", "설립구분명"]
    numeric = [c for c in feature_cols if c not in categorical]
    prep = ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ]
    )
    pipe = Pipeline([("prep", prep), ("clf", model)])
    pipe.fit(train[feature_cols], train[target])
    proba = pipe.predict_proba(test[feature_cols])[:, 1]
    precision_curve, recall_curve, thresholds = precision_recall_curve(test[target], proba)
    f1_curve = 2 * precision_curve * recall_curve / np.clip(precision_curve + recall_curve, 1e-12, None)
    best_idx = int(np.nanargmax(f1_curve[:-1])) if len(thresholds) else 0
    threshold = float(thresholds[best_idx]) if len(thresholds) else 0.5
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(test[target], pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    row = {
        "model": model_name,
        "target": target,
        "train_years": "2009-2018",
        "test_years": "2019-2022",
        "train_rows": len(train),
        "test_rows": len(test),
        "positive_rate_train": train[target].mean(),
        "positive_rate_test": test[target].mean(),
        "roc_auc": roc_auc_score(test[target], proba),
        "pr_auc": average_precision_score(test[target], proba),
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "note": "Temporal EDSS disappearance proxy validation. Current coordinate-only features such as school isolation and replacement distance are unavailable in historical EDSS.",
    }
    predictions = test[["school_id", "base_year", "sido_name", "학제명", "student_count", target]].copy()
    predictions["model"] = model_name
    predictions["probability"] = proba
    predictions["prediction"] = pred
    return pipe, row, predictions


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    data = build_temporal_dataset()
    target = "closed_next_year_proxy"
    data = data[data["학제명"].isin(["초등학교", "중학교", "고등학교"])].copy()
    train = data[data["base_year"].between(2009, 2018) & data[target].notna()].copy()
    test = data[data["base_year"].between(2019, 2022) & data[target].notna()].copy()
    train[target] = train[target].astype(int)
    test[target] = test[target].astype(int)

    feature_cols = [
        "sido_name",
        "학제명",
        "설립구분명",
        "student_count",
        "students_per_class",
        "students_per_teacher",
        "student_growth_1yr",
        "birth_count",
        "avg_total_fertility_rate",
        "avg_birth_count_yoy_rate",
        "net_migration_total",
        "in_migration_total",
        "out_migration_total",
        "commercial_count",
        "education_business_count",
        "kids_business_count",
        "commercial_per_birth",
        "education_per_birth",
        "net_migration_per_birth",
    ]
    models = {
        "base_logistic_temporal_closure": LogisticRegression(class_weight="balanced", max_iter=1200, random_state=42),
        "tuned_histgb_temporal_closure": HistGradientBoostingClassifier(
            learning_rate=0.045,
            max_iter=260,
            max_leaf_nodes=27,
            l2_regularization=0.12,
            random_state=42,
            class_weight="balanced",
        ),
    }
    rows = []
    predictions = []
    fitted = {}
    for name, model in models.items():
        pipe, row, pred = train_and_score(train, test, feature_cols, name, model)
        rows.append(row)
        predictions.append(pred)
        fitted[name] = pipe

    metrics = pd.DataFrame(rows).sort_values("f1", ascending=False)
    metrics.to_csv(REPORTS / "temporal_closure_classifier_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(predictions, ignore_index=True).to_csv(
        REPORTS / "temporal_closure_classifier_predictions.csv", index=False, encoding="utf-8-sig"
    )
    joblib.dump(fitted["base_logistic_temporal_closure"], MODELS / "base_logistic_temporal_closure.pkl")
    joblib.dump(fitted["tuned_histgb_temporal_closure"], MODELS / "tuned_histgb_temporal_closure.pkl")
    print(metrics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
