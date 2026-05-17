from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "outputs" / "reports"
MODELS = ROOT / "outputs" / "models"

SIDO_NAME_BY_CODE = {
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


def normalize_sido(value: object) -> str:
    text = str(value)
    for name in ["서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종", "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"]:
        if name in text:
            return name
    return text


def normalize_school_level(value: object) -> str:
    text = str(value)
    if "초등" in text or text == "E":
        return "초등학교"
    if "중학교" in text or text == "M":
        return "중학교"
    if "고등" in text or text == "H":
        return "고등학교"
    if "특수" in text or text == "S":
        return "특수학교"
    if "각종" in text or text == "T":
        return "각종학교"
    return "기타학교"


def build_sido_year_context() -> pd.DataFrame:
    birth = pd.read_csv(PROCESSED / "national_birth_features_sgg.csv", low_memory=False)
    birth["sido_code"] = birth["sido_code"].astype(str).str.zfill(2)
    birth["sido_name"] = birth["sido_code"].map(SIDO_NAME_BY_CODE)
    birth_ctx = (
        birth.groupby(["sido_name", "year"], as_index=False)
        .agg(
            birth_count=("birth_count", "sum"),
            avg_total_fertility_rate=("total_fertility_rate", "mean"),
            avg_birth_count_yoy_rate=("birth_count_yoy_rate", "mean"),
            avg_tfr_yoy_rate=("tfr_yoy_rate", "mean"),
        )
    )

    mig = pd.read_csv(PROCESSED / "national_migration_features_sgg.csv", low_memory=False)
    mig["sido_code"] = mig["sido_code"].astype(str).str.zfill(2)
    mig["sido_name"] = mig["sido_code"].map(SIDO_NAME_BY_CODE)
    mig_ctx = (
        mig.groupby(["sido_name", "year"], as_index=False)
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
    shop["sido_name"] = shop["ctprvnCd"].map(SIDO_NAME_BY_CODE)
    shop_ctx = (
        shop.groupby("sido_name", as_index=False)
        .agg(
            commercial_count=("commercial_count", "sum"),
            education_business_count=("education_business_count", "sum"),
            kids_business_count=("kids_business_count", "sum"),
            medical_business_count=("medical_business_count", "sum"),
        )
    )

    ctx = birth_ctx.merge(mig_ctx, on=["sido_name", "year"], how="outer").merge(shop_ctx, on="sido_name", how="left")
    for col in ctx.columns:
        if col not in ["sido_name", "year"]:
            ctx[col] = pd.to_numeric(ctx[col], errors="coerce")
    return ctx


def load_supervised_closure_dataset() -> pd.DataFrame:
    dataset = pd.read_csv(PROCESSED / "edss_national_closure_proxy_dataset.csv", low_memory=False)
    dataset["sido_name"] = dataset["시도명"].apply(normalize_sido)
    ctx = build_sido_year_context()
    merged = dataset.merge(ctx, left_on=["sido_name", "base_year"], right_on=["sido_name", "year"], how="left")
    context_cols = [c for c in ctx.columns if c not in ["sido_name", "year"]]
    for col in context_cols:
        merged[col] = merged[col].fillna(merged.groupby("sido_name")[col].transform("median")).fillna(merged[col].median())
    merged["commercial_per_birth"] = merged["commercial_count"] / merged["birth_count"].replace(0, np.nan)
    merged["education_per_birth"] = merged["education_business_count"] / merged["birth_count"].replace(0, np.nan)
    merged["net_migration_per_birth"] = merged["net_migration_total"] / merged["birth_count"].replace(0, np.nan)
    return merged.replace([np.inf, -np.inf], np.nan)


def train_final_classifier(dataset: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame]:
    feature_cols = [
        "sido_name",
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
        "birth_count",
        "avg_total_fertility_rate",
        "avg_birth_count_yoy_rate",
        "avg_tfr_yoy_rate",
        "in_migration_total",
        "out_migration_total",
        "net_migration_total",
        "avg_in_migration_yoy_rate",
        "avg_out_migration_yoy_rate",
        "commercial_count",
        "education_business_count",
        "kids_business_count",
        "medical_business_count",
        "commercial_per_birth",
        "education_per_birth",
        "net_migration_per_birth",
    ]
    target = "closed_within_5yr_proxy"
    train = dataset[dataset["base_year"].between(2011, 2015)].copy()
    test = dataset[dataset["base_year"].between(2016, 2018)].copy()

    categorical = ["sido_name", "학제명", "학제유형명", "설립구분명", "본교분교구분명", "지역행정구분명", "남녀공학구분명"]
    numeric = [c for c in feature_cols if c not in categorical]
    preprocessor = ColumnTransformer(
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
    models = {
        "baseline_logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1200,
            solver="lbfgs",
            random_state=42,
        ),
        "final_context_hist_gradient_boosting": HistGradientBoostingClassifier(
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
    reports = []
    fitted: dict[str, Pipeline] = {}
    for model_name, model in models.items():
        pipe = Pipeline([("prep", preprocessor), ("clf", model)])
        pipe.fit(train[feature_cols], train[target])
        proba = pipe.predict_proba(test[feature_cols])[:, 1]
        precision_curve, recall_curve, thresholds = precision_recall_curve(test[target], proba)
        f1_curve = 2 * precision_curve * recall_curve / np.clip(precision_curve + recall_curve, 1e-12, None)
        best_idx = int(np.nanargmax(f1_curve[:-1])) if len(thresholds) else 0
        threshold = float(thresholds[best_idx]) if len(thresholds) else 0.5
        pred = (proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(test[target], pred).ravel()
        precision_value = tp / (tp + fp) if (tp + fp) else 0.0
        recall_value = tp / (tp + fn) if (tp + fn) else 0.0
        f1_value = 2 * precision_value * recall_value / (precision_value + recall_value) if (precision_value + recall_value) else 0.0
        rows.append(
            {
                "model": model_name,
                "label_definition": "EDSS school_id disappears within 5 years",
                "train_base_years": "2011-2015",
                "test_base_years": "2016-2018",
                "train_rows": len(train),
                "test_rows": len(test),
                "positive_rate_train": train[target].mean(),
                "positive_rate_test": test[target].mean(),
                "roc_auc": roc_auc_score(test[target], proba),
                "pr_auc": average_precision_score(test[target], proba),
                "threshold": threshold,
                "precision": precision_value,
                "recall": recall_value,
                "f1": f1_value,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        )
        out_one = test[["school_id", "base_year", "sido_name", feature_cols[1], "student_count", target]].copy()
        out_one["model"] = model_name
        out_one["closure_probability"] = proba
        out_one["closure_pred"] = pred
        predictions.append(out_one)
        reports.append(f"## {model_name}\n\n{classification_report(test[target], pred)}")
        fitted[model_name] = pipe
    metrics = pd.DataFrame(rows).sort_values("f1", ascending=False)
    pipe = fitted["final_context_hist_gradient_boosting"]
    REPORTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(REPORTS / "final_supervised_closure_classifier_metrics.csv", index=False, encoding="utf-8-sig")
    (REPORTS / "final_supervised_closure_classifier_report.txt").write_text("\n\n".join(reports), encoding="utf-8")
    pd.concat(predictions, ignore_index=True).to_csv(
        REPORTS / "final_supervised_closure_classifier_test_predictions.csv", index=False, encoding="utf-8-sig"
    )
    joblib.dump(fitted["baseline_logistic_regression"], MODELS / "baseline_logistic_closure_classifier.pkl")
    joblib.dump(pipe, MODELS / "final_supervised_closure_classifier.pkl")
    return pipe, metrics
    (REPORTS / "final_supervised_closure_classifier_report.txt").write_text(
        classification_report(test[target], pred), encoding="utf-8"
    )
    out = test[["school_id", "base_year", "sido_name", "학제명", "student_count", target]].copy()
    out["closure_probability"] = proba
    out["closure_pred"] = pred
    out.to_csv(REPORTS / "final_supervised_closure_classifier_test_predictions.csv", index=False, encoding="utf-8-sig")
    joblib.dump(pipe, MODELS / "final_supervised_closure_classifier.pkl")
    return pipe, metrics


def apply_classifier_to_current(pipe: Pipeline) -> pd.DataFrame:
    current = pd.read_csv(PROCESSED / "final_national_current_school_features.csv", low_memory=False)
    feature_names = list(pipe.named_steps["prep"].feature_names_in_)
    X = pd.DataFrame(index=current.index)
    X["sido_name"] = current["requested_sido_name"]
    X["학제명"] = current["school_level"]
    X["학제유형명"] = current["school_level"]
    X["설립구분명"] = current["foundation"]
    X["본교분교구분명"] = "본교"
    X["지역행정구분명"] = "기타"
    X["남녀공학구분명"] = "남녀공학"
    X["student_count"] = current["student_count_2025"]
    X["class_count"] = current["class_count_2025"]
    X["teacher_count"] = current["teacher_count_2025"]
    X["students_per_class"] = current["students_per_class"]
    X["students_per_teacher"] = current["student_count_2025"] / current["teacher_count_2025"].replace(0, np.nan)
    X["entrants"] = np.nan
    X["graduates"] = np.nan
    X["entrant_graduate_ratio"] = np.nan
    X["net_transfer"] = np.nan
    X["land_area"] = np.nan
    X["playground_area"] = np.nan
    X["regular_classrooms"] = np.nan
    X["school_age"] = current["school_age"]
    X["student_growth_1yr"] = np.nan
    detail_path = RAW / "eduinfo_current_school_detail_national.csv"
    if detail_path.exists():
        detail_cols = {
            "chart_schlCd",
            "sheet_schlCd",
            "chart_mainSchlCd",
            "sheet_mainSchlCd",
            "chart_coeduCd",
            "sheet_coeduCd",
            "chart_stdtECnt1",
            "sheet_stdtECnt1",
            "chart_stdtECnt6",
            "sheet_stdtECnt6",
            "chart_stdtMCnt1",
            "sheet_stdtMCnt1",
            "chart_stdtMCnt3",
            "sheet_stdtMCnt3",
            "chart_stdtHCnt1",
            "sheet_stdtHCnt1",
            "chart_stdtHCnt3",
            "sheet_stdtHCnt3",
            "chart_totClassCnt",
            "sheet_totClassCnt",
        }
        detail = pd.read_csv(detail_path, usecols=lambda col: col in detail_cols, low_memory=False)
        school_code_col = "chart_schlCd" if detail.get("chart_schlCd", pd.Series(dtype=object)).notna().any() else "sheet_schlCd"
        if school_code_col in detail.columns:
            detail = detail.rename(columns={school_code_col: "schlCd"}).dropna(subset=["schlCd"]).drop_duplicates("schlCd")
            detail["schlCd"] = detail["schlCd"].astype(str)
            current_detail = current[["schlCd", "school_level"]].copy()
            current_detail["schlCd"] = current_detail["schlCd"].astype(str)
            current_detail = current_detail.merge(detail, on="schlCd", how="left")
            for col in detail_cols - {"chart_schlCd", "sheet_schlCd", "chart_mainSchlCd", "sheet_mainSchlCd", "chart_coeduCd", "sheet_coeduCd"}:
                if col in current_detail.columns:
                    current_detail[col] = pd.to_numeric(current_detail[col], errors="coerce")
            level = current_detail["school_level"].astype(str)
            def first_available(*cols: str) -> pd.Series:
                values = pd.Series(np.nan, index=current_detail.index)
                for col in cols:
                    if col in current_detail.columns:
                        values = values.fillna(current_detail[col])
                return values

            X["entrants"] = np.select(
                [level.str.contains("초등"), level.str.contains("중"), level.str.contains("고")],
                [
                    first_available("chart_stdtECnt1", "sheet_stdtECnt1"),
                    first_available("chart_stdtMCnt1", "sheet_stdtMCnt1"),
                    first_available("chart_stdtHCnt1", "sheet_stdtHCnt1"),
                ],
                default=np.nan,
            )
            X["graduates"] = np.select(
                [level.str.contains("초등"), level.str.contains("중"), level.str.contains("고")],
                [
                    first_available("chart_stdtECnt6", "sheet_stdtECnt6"),
                    first_available("chart_stdtMCnt3", "sheet_stdtMCnt3"),
                    first_available("chart_stdtHCnt3", "sheet_stdtHCnt3"),
                ],
                default=np.nan,
            )
            X["entrant_graduate_ratio"] = X["entrants"] / pd.Series(X["graduates"]).replace(0, np.nan)
            X["regular_classrooms"] = first_available("chart_totClassCnt", "sheet_totClassCnt")
            main_branch_col = next((name for name in feature_names if "본교" in name and "분교" in name), None)
            main_code = first_available("chart_mainSchlCd", "sheet_mainSchlCd")
            if main_branch_col:
                has_main_code = main_code.fillna("").astype(str).str.strip().ne("")
                X[main_branch_col] = np.where(has_main_code, "분교", "본교")
            coedu_col = next((name for name in feature_names if "남녀" in name and "공학" in name), None)
            coedu_code_source = first_available("chart_coeduCd", "sheet_coeduCd")
            if coedu_col:
                coedu_code = coedu_code_source.fillna("").astype(str).str.strip()
                X[coedu_col] = np.select(
                    [coedu_code.eq("1"), coedu_code.eq("2"), coedu_code.eq("3")],
                    ["남녀공학", "남학교", "여학교"],
                    default=X[coedu_col],
                )
    # Current-school detail data often lacks fields that exist in the historical
    # EDSS panel. Keep API-derived values when present, then add transparent
    # proxies so current inference is not dominated by 100% missing features.
    level_text = current["school_level"].astype(str)
    student_count = pd.to_numeric(current["student_count_2025"], errors="coerce")
    estimated_entrants = np.select(
        [level_text.str.contains("초등"), level_text.str.contains("중"), level_text.str.contains("고")],
        [student_count / 6, student_count / 3, student_count / 3],
        default=np.nan,
    )
    X["entrants"] = pd.Series(X["entrants"], index=current.index).fillna(pd.Series(estimated_entrants, index=current.index))
    X["graduates"] = pd.Series(X["graduates"], index=current.index).fillna(pd.Series(estimated_entrants, index=current.index))
    X["entrant_graduate_ratio"] = pd.Series(X["entrant_graduate_ratio"], index=current.index).fillna(
        X["entrants"] / pd.Series(X["graduates"], index=current.index).replace(0, np.nan)
    )
    X["net_transfer"] = pd.Series(X["net_transfer"], index=current.index).fillna(0)

    edss_panel_path = PROCESSED / "edss_national_school_panel_2009_2023.csv"
    if edss_panel_path.exists():
        edss_cols = [
            "year",
            "시도명",
            "학제명",
            "student_growth_1yr",
            "land_area",
            "playground_area",
            "regular_classrooms",
            "net_transfer",
        ]
        edss = pd.read_csv(edss_panel_path, usecols=edss_cols, low_memory=False)
        latest_edss = edss[edss["year"].eq(edss["year"].max())].copy()
        for col in ["student_growth_1yr", "land_area", "playground_area", "regular_classrooms", "net_transfer"]:
            latest_edss[col] = pd.to_numeric(latest_edss[col], errors="coerce")
        by_sido_level = latest_edss.groupby(["시도명", "학제명"], dropna=False)[
            ["student_growth_1yr", "land_area", "playground_area", "regular_classrooms", "net_transfer"]
        ].median()
        by_level = latest_edss.groupby("학제명", dropna=False)[
            ["student_growth_1yr", "land_area", "playground_area", "regular_classrooms", "net_transfer"]
        ].median()
        keys = pd.MultiIndex.from_arrays([current["requested_sido_name"], current["school_level"]])
        level_keys = current["school_level"]
        for col in ["student_growth_1yr", "land_area", "playground_area", "regular_classrooms", "net_transfer"]:
            sido_values = pd.Series(by_sido_level[col].reindex(keys).to_numpy(), index=current.index)
            level_values = pd.Series(by_level[col].reindex(level_keys).to_numpy(), index=current.index)
            global_value = latest_edss[col].median()
            X[col] = pd.Series(X[col], index=current.index).fillna(sido_values).fillna(level_values).fillna(global_value)

    X["birth_count"] = current["birth_count"]
    X["avg_total_fertility_rate"] = current["total_fertility_rate"]
    X["avg_birth_count_yoy_rate"] = current["birth_count_yoy_rate"]
    X["avg_tfr_yoy_rate"] = current["tfr_yoy_rate"]
    X["in_migration_total"] = current["in_migration_total"]
    X["out_migration_total"] = current["out_migration_total"]
    X["net_migration_total"] = current["net_migration_total"]
    X["avg_in_migration_yoy_rate"] = current["in_migration_yoy_rate"]
    X["avg_out_migration_yoy_rate"] = current["out_migration_yoy_rate"]
    X["commercial_count"] = current["commercial_count"]
    X["education_business_count"] = current["education_business_count"]
    X["kids_business_count"] = current["kids_business_count"]
    X["medical_business_count"] = current["medical_business_count"]
    X["commercial_per_birth"] = current["commercial_count"] / current["birth_count"].replace(0, np.nan)
    X["education_per_birth"] = current["education_business_count"] / current["birth_count"].replace(0, np.nan)
    X["net_migration_per_birth"] = current["net_migration_total"] / current["birth_count"].replace(0, np.nan)
    missing_report = (
        X[feature_names]
        .isna()
        .mean()
        .rename("missing_rate")
        .reset_index()
        .rename(columns={"index": "feature"})
        .sort_values("missing_rate", ascending=False)
    )
    missing_report.to_csv(REPORTS / "final_supervised_current_feature_missing_rate.csv", index=False, encoding="utf-8-sig")
    proba = pipe.predict_proba(X[feature_names])[:, 1]
    result = current.copy()
    result["final_supervised_closure_probability"] = proba
    result["final_supervised_closure_percentile"] = pd.Series(proba).rank(pct=True).mul(100).round(1)
    result.to_csv(PROCESSED / "final_national_current_school_features.csv", index=False, encoding="utf-8-sig")
    return result


def update_scenario_with_supervised_scores(current: pd.DataFrame) -> None:
    scenario_path = PROCESSED / "final_national_school_scenario_2026_2040.csv"
    if not scenario_path.exists():
        return
    scenario = pd.read_csv(scenario_path, low_memory=False)
    scores = current[["schlCd", "final_supervised_closure_probability", "final_supervised_closure_percentile"]]
    scenario = scenario.drop(
        columns=[c for c in ["final_supervised_closure_probability", "final_supervised_closure_percentile"] if c in scenario.columns]
    )
    scenario = scenario.merge(scores, on="schlCd", how="left")
    scenario["final_supervised_closure_percentile"] = scenario["final_supervised_closure_percentile"].fillna(
        scenario["final_supervised_closure_percentile"].median()
    )
    scenario.to_csv(scenario_path, index=False, encoding="utf-8-sig")


def main() -> int:
    dataset = load_supervised_closure_dataset()
    dataset.to_csv(PROCESSED / "final_supervised_closure_training_dataset.csv", index=False, encoding="utf-8-sig")
    pipe, metrics = train_final_classifier(dataset)
    current = apply_classifier_to_current(pipe)
    update_scenario_with_supervised_scores(current)
    print(metrics.to_string(index=False))
    print("saved:", REPORTS / "final_supervised_closure_classifier_metrics.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
