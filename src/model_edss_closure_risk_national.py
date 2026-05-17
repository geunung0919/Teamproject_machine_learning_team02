from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
EDSS = ROOT / "data" / "edss"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
REPORTS = OUTPUTS / "reports"
MODELS = OUTPUTS / "models"

HORIZON_YEARS = 5
TEST_BASE_YEARS = [2016, 2017, 2018]


def read_edss_csv(prefix: str, usecols: list[str]) -> pd.DataFrame:
    matches = sorted(EDSS.glob(f"{prefix}*.csv"))
    if not matches:
        raise FileNotFoundError(f"EDSS file not found: {prefix}")
    frame = pd.read_csv(matches[0], encoding="cp949", usecols=usecols, low_memory=False)
    frame["개방ID"] = frame["개방ID"].astype(str)
    return frame


def load_panel() -> pd.DataFrame:
    attrs = read_edss_csv(
        "0001.",
        [
            "조사년도",
            "개방ID",
            "시도명",
            "학제명",
            "학제유형명",
            "설립구분명",
            "본교분교구분명",
            "학교설립일자",
            "지역행정구분명",
            "남녀공학구분명",
            "학교수",
        ],
    )
    overview = read_edss_csv(
        "0002.",
        [
            "조사년도",
            "개방ID",
            "유초중등학교개황_학생수",
            "유초중등학교개황_학급수",
            "유초중등학교개황_교원수",
            "유초중등학교개황_입학생수",
            "유초중등학교개황_졸업생수",
            "유초중등학교개황_학교용지대지면적",
            "유초중등학교개황_학교용지체육장면적",
            "유초중등학교개황_정규교실수",
        ],
    )
    students = read_edss_csv(
        "0004.",
        [
            "조사년도",
            "개방ID",
            "유초중등학생_학생수",
            "유초중등학생_입학생수",
            "유초중등학생_졸업생수",
            "도내전입학생수",
            "도내전출학생수",
            "도외전입학생수",
            "도외전출학생수",
        ],
    )

    panel = attrs.merge(overview, on=["조사년도", "개방ID"], how="left").merge(
        students, on=["조사년도", "개방ID"], how="left"
    )
    panel = panel[~panel["학제명"].fillna("").str.contains("유치원")].copy()
    panel = panel[panel["학교수"].fillna(1).astype(float).gt(0)].copy()
    panel["year"] = pd.to_numeric(panel["조사년도"], errors="coerce").astype("Int64")
    panel["school_id"] = panel["개방ID"].astype(str)
    panel["student_count"] = pd.to_numeric(
        panel["유초중등학생_학생수"].fillna(panel["유초중등학교개황_학생수"]), errors="coerce"
    )
    panel["class_count"] = pd.to_numeric(panel["유초중등학교개황_학급수"], errors="coerce")
    panel["teacher_count"] = pd.to_numeric(panel["유초중등학교개황_교원수"], errors="coerce")
    panel["entrants"] = pd.to_numeric(
        panel["유초중등학생_입학생수"].fillna(panel["유초중등학교개황_입학생수"]), errors="coerce"
    )
    panel["graduates"] = pd.to_numeric(
        panel["유초중등학생_졸업생수"].fillna(panel["유초중등학교개황_졸업생수"]), errors="coerce"
    )
    panel["internal_in"] = pd.to_numeric(panel["도내전입학생수"], errors="coerce")
    panel["internal_out"] = pd.to_numeric(panel["도내전출학생수"], errors="coerce")
    panel["external_in"] = pd.to_numeric(panel["도외전입학생수"], errors="coerce")
    panel["external_out"] = pd.to_numeric(panel["도외전출학생수"], errors="coerce")
    panel["land_area"] = pd.to_numeric(panel["유초중등학교개황_학교용지대지면적"], errors="coerce")
    panel["playground_area"] = pd.to_numeric(panel["유초중등학교개황_학교용지체육장면적"], errors="coerce")
    panel["regular_classrooms"] = pd.to_numeric(panel["유초중등학교개황_정규교실수"], errors="coerce")
    panel["school_age"] = panel["year"] - pd.to_numeric(
        panel["학교설립일자"].astype(str).str[:4], errors="coerce"
    )
    panel["students_per_class"] = panel["student_count"] / panel["class_count"].replace(0, np.nan)
    panel["students_per_teacher"] = panel["student_count"] / panel["teacher_count"].replace(0, np.nan)
    panel["net_transfer"] = (
        panel["internal_in"].fillna(0)
        + panel["external_in"].fillna(0)
        - panel["internal_out"].fillna(0)
        - panel["external_out"].fillna(0)
    )
    panel["entrant_graduate_ratio"] = panel["entrants"] / panel["graduates"].replace(0, np.nan)
    panel = panel.sort_values(["school_id", "year"])
    panel["student_count_lag1"] = panel.groupby("school_id")["student_count"].shift(1)
    panel["student_growth_1yr"] = (
        panel["student_count"] - panel["student_count_lag1"]
    ) / panel["student_count_lag1"].replace(0, np.nan)
    return panel


def add_closure_labels(panel: pd.DataFrame) -> pd.DataFrame:
    years_by_school = panel.groupby("school_id")["year"].apply(lambda s: set(s.dropna().astype(int))).to_dict()
    rows = []
    for _, row in panel.iterrows():
        year = int(row["year"])
        if year > 2023 - HORIZON_YEARS:
            continue
        future_years = set(range(year + 1, year + HORIZON_YEARS + 1))
        observed_future = years_by_school.get(row["school_id"], set()) & future_years
        new_row = row.copy()
        new_row["closed_within_5yr_proxy"] = int(len(observed_future) < HORIZON_YEARS)
        new_row["base_year"] = year
        rows.append(new_row)
    return pd.DataFrame(rows)


def train_classifier(dataset: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame, str]:
    feature_cols = [
        "시도명",
        "학제명",
        "학제유형명",
        "설립구분명",
        "본교분교구분명",
        "지역행정구분명",
        "남녀공학구분명",
        "student_count",
        "class_count",
        "teacher_count",
        "students_per_class",
        "students_per_teacher",
        "entrants",
        "graduates",
        "entrant_graduate_ratio",
        "net_transfer",
        "land_area",
        "playground_area",
        "regular_classrooms",
        "school_age",
        "student_growth_1yr",
    ]
    target = "closed_within_5yr_proxy"
    train = dataset[~dataset["base_year"].isin(TEST_BASE_YEARS)].copy()
    test = dataset[dataset["base_year"].isin(TEST_BASE_YEARS)].copy()
    categorical = ["시도명", "학제명", "학제유형명", "설립구분명", "본교분교구분명", "지역행정구분명", "남녀공학구분명"]
    numeric = [col for col in feature_cols if col not in categorical]
    train[feature_cols] = train[feature_cols].replace([pd.NA, np.inf, -np.inf], np.nan)
    test[feature_cols] = test[feature_cols].replace([pd.NA, np.inf, -np.inf], np.nan)

    preprocessor = ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        sparse_threshold=0.0,
    )
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=220,
        max_leaf_nodes=23,
        l2_regularization=0.1,
        random_state=42,
        class_weight="balanced",
    )
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    pipe.fit(train[feature_cols], train[target])
    proba = pipe.predict_proba(test[feature_cols])[:, 1]

    precision, recall, thresholds = precision_recall_curve(test[target], proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best_idx = int(np.nanargmax(f1[:-1])) if len(thresholds) else 0
    threshold = float(thresholds[best_idx]) if len(thresholds) else 0.5
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(test[target], pred).ravel()
    metrics = pd.DataFrame(
        [
            {
                "scope": "national",
                "label_definition": f"EDSS school_id disappears within {HORIZON_YEARS} years",
                "train_base_years": f"{int(train['base_year'].min())}-{int(train['base_year'].max())}",
                "test_base_years": ",".join(map(str, TEST_BASE_YEARS)),
                "train_rows": len(train),
                "test_rows": len(test),
                "positive_rate_train": train[target].mean(),
                "positive_rate_test": test[target].mean(),
                "roc_auc": roc_auc_score(test[target], proba),
                "pr_auc": average_precision_score(test[target], proba),
                "threshold": threshold,
                "tn": cm[0],
                "fp": cm[1],
                "fn": cm[2],
                "tp": cm[3],
            }
        ]
    )

    per_sido = []
    test_out = test[["school_id", "base_year", "시도명", "학제명", "student_count", target]].copy()
    test_out["closure_probability"] = proba
    test_out["pred_closed_within_5yr"] = pred
    for sido, group in test_out.groupby("시도명"):
        if group[target].nunique() < 2:
            continue
        per_sido.append(
            {
                "시도명": sido,
                "rows": len(group),
                "positives": int(group[target].sum()),
                "positive_rate": group[target].mean(),
                "roc_auc": roc_auc_score(group[target], group["closure_probability"]),
                "pr_auc": average_precision_score(group[target], group["closure_probability"]),
            }
        )
    pd.DataFrame(per_sido).sort_values("pr_auc", ascending=False).to_csv(
        REPORTS / "edss_closure_classifier_metrics_by_sido_national.csv", index=False, encoding="utf-8-sig"
    )
    test_out.to_csv(REPORTS / "edss_closure_classifier_test_predictions_national.csv", index=False, encoding="utf-8-sig")
    return pipe, metrics, classification_report(test[target], pred, zero_division=0)


def main() -> int:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    panel.to_csv(PROCESSED / "edss_national_school_panel_2009_2023.csv", index=False, encoding="utf-8-sig")
    dataset = add_closure_labels(panel)
    dataset.to_csv(PROCESSED / "edss_national_closure_proxy_dataset.csv", index=False, encoding="utf-8-sig")
    model, metrics, report = train_classifier(dataset)
    metrics.to_csv(REPORTS / "edss_closure_classifier_metrics_national.csv", index=False, encoding="utf-8-sig")
    (REPORTS / "edss_closure_classifier_report_national.txt").write_text(report, encoding="utf-8")
    joblib.dump(model, MODELS / "edss_closure_proxy_classifier_national.pkl")
    print("national panel rows:", len(panel))
    print("national closure dataset rows:", len(dataset))
    print(metrics.to_string(index=False))
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
