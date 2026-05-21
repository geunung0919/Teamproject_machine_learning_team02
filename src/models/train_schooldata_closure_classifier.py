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

PANEL_PATH = PROCESSED / "schooldata_modeling_panel_2008_2025_geocoded.csv"
TARGET = "closure_within_3yr_label"

COL_YEAR = "데이터_연도"
COL_SIDO = "시도"
COL_SGG = "행정구"
COL_LEVEL = "학교급"
COL_NAME = "학교명"
COL_STATUS = "상태"
COL_FOUNDATION = "설립"
COL_BRANCH = "본분교"

STUDENT_LIMIT_BY_LEVEL = {"초등학교": 80, "중학교": 330, "고등학교": 330}
CATEGORICAL_FEATURES = ["sido_name", COL_LEVEL, COL_FOUNDATION, COL_BRANCH]
NUMERIC_FEATURES = [
    "student_count",
    "class_count",
    "teacher_count",
    "students_per_class",
    "students_per_teacher",
    "student_growth_1yr",
    "student_diff_1yr",
    "land_area",
    "land_area_per_student",
    "nearest_same_level_school_km",
    "same_level_school_count_5km",
    "school_isolation_score",
    "replacement_available_score",
    "isolation_protection_flag",
    "closure_feasibility_score",
    "radius_0_5km_all_shops",
    "radius_0_5km_education_shops",
    "radius_0_5km_kids_shops",
    "radius_1_0km_all_shops",
    "radius_1_0km_education_shops",
    "radius_1_0km_kids_shops",
]
FEATURE_COLS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
ID_COLS = [
    "year",
    COL_YEAR,
    "school_key",
    COL_SIDO,
    COL_SGG,
    COL_LEVEL,
    COL_NAME,
    COL_STATUS,
    "student_count",
    "lttud",
    "lgtud",
    "coordinate_source",
    TARGET,
]


def add_policy_candidate_flag(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    student_limit = data[COL_LEVEL].map(STUDENT_LIMIT_BY_LEVEL).fillna(120)
    nearest = pd.to_numeric(data["nearest_same_level_school_km"], errors="coerce").fillna(99)
    same_5km = pd.to_numeric(data["same_level_school_count_5km"], errors="coerce").fillna(0)
    isolation = pd.to_numeric(data["school_isolation_score"], errors="coerce").fillna(100)
    data["replacement_available_score"] = (
        (1 - nearest.clip(0, 10) / 10) * 60 + (same_5km.clip(0, 10) / 10) * 40
    ).clip(0, 100)
    data["isolation_protection_flag"] = ((nearest >= 5) | (same_5km <= 0) | (isolation >= 60)).astype(int)
    data["closure_feasibility_score"] = np.where(
        data["isolation_protection_flag"].eq(1),
        data["replacement_available_score"] * 0.35,
        data["replacement_available_score"],
    )
    is_existing = data[COL_STATUS].astype(str).str.contains("기존", na=False)
    has_valid_student = data["student_count"].fillna(0).gt(0)
    is_low_student = data["student_count"].le(student_limit)
    is_branch = data[COL_BRANCH].astype(str).str.contains("분교", na=False)
    data["candidate_eligible"] = (is_existing & has_valid_student & (is_low_student | is_branch)).astype(int)
    data["candidate_reason"] = np.select(
        [~is_existing, ~has_valid_student, is_branch, is_low_student],
        ["신설/비기존 학교 제외", "학생수 0 또는 결측", "분교/분교장", "학교급별 저학생수"],
        default="학생수 규모 기준 제외",
    )
    return data


def apply_feasibility_adjustment(data: pd.DataFrame, proba: np.ndarray) -> np.ndarray:
    feasibility = pd.to_numeric(data["closure_feasibility_score"], errors="coerce").fillna(0).to_numpy() / 100
    protection = pd.to_numeric(data["isolation_protection_flag"], errors="coerce").fillna(0).to_numpy()
    multiplier = 0.25 + 0.75 * feasibility
    multiplier = np.where(protection >= 1, multiplier * 0.35, multiplier)
    return np.clip(proba * multiplier, 0, 1)


def add_future_closure_target(data: pd.DataFrame, horizon: int = 3) -> pd.DataFrame:
    data = data.sort_values(["school_key", "year"]).copy()
    data[TARGET] = 0.0
    for step in range(1, horizon + 1):
        data[f"future_status_{step}yr"] = data.groupby("school_key")[COL_STATUS].shift(-step)
        data[f"future_year_{step}yr"] = data.groupby("school_key")["year"].shift(-step)
        immediate = data[f"future_year_{step}yr"].eq(data["year"] + step)
        closed = data[f"future_status_{step}yr"].astype(str).str.contains("폐", na=False)
        data[TARGET] = np.where(immediate & closed, 1.0, data[TARGET])
    max_trainable_year = int(data["year"].max()) - horizon
    data.loc[data["year"].gt(max_trainable_year), TARGET] = np.nan
    return data


def f1_optimal_threshold(y_true: pd.Series, proba: np.ndarray) -> tuple[float, np.ndarray]:
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, proba)
    if len(thresholds) == 0:
        return 0.5, (proba >= 0.5).astype(int)
    f1_curve = 2 * precision_curve * recall_curve / np.clip(precision_curve + recall_curve, 1e-12, None)
    best_idx = int(np.nanargmax(f1_curve[:-1]))
    threshold = float(thresholds[best_idx])
    return threshold, (proba >= threshold).astype(int)


def topk_rows(y_true: pd.Series, proba: np.ndarray, model_name: str, segment: str = "all") -> list[dict[str, float | int | str]]:
    y = np.asarray(y_true).astype(int)
    order = np.argsort(-proba)
    positives = int(y.sum())
    total = len(y)
    base_rate = positives / total if total else 0.0
    candidate_ks = [10, 20, 50, 100, 200, 500, 1000, positives]
    rows = []
    for k in sorted({int(k) for k in candidate_ks if int(k) > 0}):
        k = min(k, total)
        idx = order[:k]
        hits = int(y[idx].sum())
        precision = hits / k if k else 0.0
        recall = hits / positives if positives else 0.0
        rows.append(
            {
                "model": model_name,
                "segment": segment,
                "k": k,
                "actual_positive_total": positives,
                "hits_at_k": hits,
                "precision_at_k": precision,
                "recall_at_k": recall,
                "capture_rate_at_k": recall,
                "lift_at_k": precision / base_rate if base_rate else 0.0,
            }
        )
    return rows


def make_pipeline(model) -> Pipeline:
    prep = ColumnTransformer(
        [
            (
                "num",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline([("prep", prep), ("clf", model)])


def evaluate_predictions(y_true: pd.Series, proba: np.ndarray, threshold: float | None = None) -> tuple[dict, np.ndarray, float]:
    if threshold is None:
        threshold, pred = f1_optimal_threshold(y_true, proba)
    else:
        pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    row = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "threshold": threshold,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
    return row, pred, threshold


def train_and_evaluate(train: pd.DataFrame, test: pd.DataFrame, name: str, model) -> tuple[Pipeline, dict, pd.DataFrame, list[dict], pd.DataFrame]:
    pipe = make_pipeline(model)
    pipe.fit(train[FEATURE_COLS], train[TARGET])
    proba = pipe.predict_proba(test[FEATURE_COLS])[:, 1]
    feasibility_proba = apply_feasibility_adjustment(test, proba)
    base_row, pred, threshold = evaluate_predictions(test[TARGET], proba)
    row = {
        "model": name,
        "target": TARGET,
        "task_definition": "candidate schools only, closure within 3 years",
        "train_years": "2008-2018",
        "test_years": "2019-2022",
        "train_rows": len(train),
        "test_rows": len(test),
        "positive_rate_train": train[TARGET].mean(),
        "positive_rate_test": test[TARGET].mean(),
        **base_row,
        "feature_note": "Candidate-filtered school-year panel + recovered coordinates + isolation + commercial radius features.",
    }

    keep_cols = [c for c in ID_COLS + ["candidate_eligible", "candidate_reason"] if c in test.columns]
    predictions = test[keep_cols].copy()
    predictions["model"] = name
    predictions["probability"] = proba
    predictions["closure_feasibility_probability"] = feasibility_proba
    predictions["prediction"] = pred
    predictions["rank_in_test"] = predictions["probability"].rank(ascending=False, method="first").astype(int)

    topk = topk_rows(test[TARGET], proba, name, "all")
    by_level_rows = []
    for level, group in test.assign(_proba=proba).groupby(COL_LEVEL):
        if group[TARGET].nunique() < 2:
            continue
        row_part, _, _ = evaluate_predictions(group[TARGET], group["_proba"].to_numpy(), threshold=threshold)
        row_part = {
            "model": name,
            "segment": str(level),
            "rows": len(group),
            "positive_rate": group[TARGET].mean(),
            **row_part,
        }
        by_level_rows.append(row_part)
        topk.extend(topk_rows(group[TARGET], group["_proba"].to_numpy(), name, str(level)))

    return pipe, row, predictions, topk, pd.DataFrame(by_level_rows)


def predict_current_2025(source: pd.DataFrame, fitted: dict[str, Pipeline], metrics: pd.DataFrame) -> pd.DataFrame:
    current = source[(source["year"].eq(2025)) & source["lttud"].notna() & source["lgtud"].notna()].copy()
    current = add_policy_candidate_flag(current)
    current = current[current["candidate_eligible"].eq(1)].copy()
    best_model_name = str(metrics.sort_values(["f1", "pr_auc"], ascending=False).iloc[0]["model"])
    for name, pipe in fitted.items():
        raw = pipe.predict_proba(current[FEATURE_COLS])[:, 1]
        feasibility = apply_feasibility_adjustment(current, raw)
        current[f"{name}_probability"] = raw
        current[f"{name}_feasibility_probability"] = feasibility
    current["best_model"] = best_model_name
    current["closure_probability"] = current[f"{best_model_name}_probability"]
    current["closure_feasibility_probability"] = current[f"{best_model_name}_feasibility_probability"]
    current["risk_rank"] = current["closure_probability"].rank(ascending=False, method="first").astype(int)
    bins = [-0.001, 0.05, 0.15, 0.35, 1.0]
    labels = ["낮음", "관찰", "높음", "매우 높음"]
    current["model_risk_group"] = pd.cut(current["closure_probability"], bins=bins, labels=labels).astype(str)
    current["final_policy_category"] = np.where(
        current["isolation_protection_flag"].eq(1),
        "교육공백 보호대상",
        "통폐합 검토 후보",
    )
    keep_cols = [
        "year",
        "school_key",
        COL_SIDO,
        COL_SGG,
        COL_LEVEL,
        COL_NAME,
        COL_STATUS,
        COL_BRANCH,
        "student_count",
        "lttud",
        "lgtud",
        "coordinate_source",
        "nearest_same_level_school_km",
        "same_level_school_count_5km",
        "school_isolation_score",
        "replacement_available_score",
        "isolation_protection_flag",
        "closure_feasibility_score",
        "radius_1_0km_all_shops",
        "radius_1_0km_education_shops",
        "base_logistic_schooldata_closure_probability",
        "base_logistic_schooldata_closure_feasibility_probability",
        "tuned_histgb_schooldata_closure_probability",
        "tuned_histgb_schooldata_closure_feasibility_probability",
        "closure_probability",
        "closure_feasibility_probability",
        "candidate_eligible",
        "candidate_reason",
        "risk_rank",
        "model_risk_group",
        "final_policy_category",
        "best_model",
    ]
    return current[[c for c in keep_cols if c in current.columns]].sort_values("risk_rank")


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(PANEL_PATH, low_memory=False).replace([np.inf, -np.inf], np.nan)
    source = add_policy_candidate_flag(source)
    source = add_future_closure_target(source, horizon=3)
    data = source[source[TARGET].notna()].copy()
    data = data[data["candidate_eligible"].eq(1)].copy()
    data = data[data["lttud"].notna() & data["lgtud"].notna()].copy()
    data[TARGET] = data[TARGET].astype(int)

    train = data[data["year"].between(2008, 2018)].copy()
    test = data[data["year"].between(2019, 2022)].copy()
    models = {
        "base_logistic_schooldata_closure": LogisticRegression(class_weight="balanced", max_iter=1200, random_state=42),
        "tuned_histgb_schooldata_closure": HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=300,
            max_leaf_nodes=15,
            min_samples_leaf=25,
            l2_regularization=0.2,
            random_state=42,
            class_weight="balanced",
        ),
    }

    metric_rows = []
    topk_rows_all = []
    prediction_frames = []
    by_level_frames = []
    fitted = {}
    for name, model in models.items():
        pipe, row, pred, topk, by_level = train_and_evaluate(train, test, name, model)
        metric_rows.append(row)
        topk_rows_all.extend(topk)
        prediction_frames.append(pred)
        by_level_frames.append(by_level)
        fitted[name] = pipe

    metrics = pd.DataFrame(metric_rows).sort_values(["f1", "pr_auc"], ascending=False)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    topk_metrics = pd.DataFrame(topk_rows_all)
    by_level_metrics = pd.concat(by_level_frames, ignore_index=True) if by_level_frames else pd.DataFrame()
    current = predict_current_2025(source, fitted, metrics)

    metrics.to_csv(REPORTS / "schooldata_closure_classifier_metrics.csv", index=False, encoding="utf-8-sig")
    by_level_metrics.to_csv(REPORTS / "schooldata_closure_classifier_metrics_by_level.csv", index=False, encoding="utf-8-sig")
    topk_metrics.to_csv(REPORTS / "schooldata_closure_classifier_topk_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(REPORTS / "schooldata_closure_classifier_predictions.csv", index=False, encoding="utf-8-sig")
    current.to_csv(PROCESSED / "schooldata_current_closure_risk_2025.csv", index=False, encoding="utf-8-sig")
    joblib.dump(fitted["base_logistic_schooldata_closure"], MODELS / "base_logistic_schooldata_closure.pkl")
    joblib.dump(fitted["tuned_histgb_schooldata_closure"], MODELS / "tuned_histgb_schooldata_closure.pkl")

    print(metrics.to_string(index=False))
    print("\nBy level")
    print(by_level_metrics.to_string(index=False))
    print("\nTop-K")
    print(topk_metrics[topk_metrics["segment"].eq("all")].to_string(index=False))
    print("\nCurrent candidates saved:", PROCESSED / "schooldata_current_closure_risk_2025.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
