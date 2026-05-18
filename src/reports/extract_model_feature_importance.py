from __future__ import annotations

import os
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from models.train_temporal_closure_classifier import build_temporal_dataset


ROOT = SRC.parent
REPORTS = ROOT / "outputs" / "reports"
FIGURES = ROOT / "outputs" / "figures"
MODELS = ROOT / "outputs" / "models"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


GROUP_LABELS = {
    "sgg_code": "시군구 고정효과",
    "sido_code": "시도 고정효과",
    "sido_name": "시도",
    "학제명": "학교급",
    "학제유형명": "학교급 유형",
    "설립구분명": "설립구분",
    "본교분교구분명": "본교/분교",
    "지역행정구분명": "지역행정구분",
    "남녀공학구분명": "남녀공학",
}


def collapse_onehot_name(name: str) -> str:
    if name.startswith("num__"):
        return name.replace("num__", "")
    if name.startswith("cat__"):
        rest = name.replace("cat__", "")
        for base in GROUP_LABELS:
            if rest == base or rest.startswith(f"{base}_"):
                return GROUP_LABELS[base]
        return rest.split("_")[0]
    return name


def feature_importance(pipe, model_step: str) -> pd.DataFrame:
    preprocessor = pipe.named_steps["prep"]
    names = preprocessor.get_feature_names_out()
    model = pipe.named_steps[model_step]
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(np.ravel(model.coef_))
    else:
        raise TypeError(f"Model step {model_step} has no direct feature importance.")
    frame = pd.DataFrame({"encoded_feature": names, "importance": values})
    frame["feature"] = frame["encoded_feature"].map(collapse_onehot_name)
    grouped = frame.groupby("feature", as_index=False)["importance"].sum()
    grouped["importance_share"] = grouped["importance"] / grouped["importance"].sum()
    return grouped.sort_values("importance_share", ascending=False)


def permutation_feature_importance(
    pipe,
    dataset_path: Path,
    target_col: str,
    test_year_min: int,
    test_year_max: int,
    output_name: str,
) -> pd.DataFrame:
    data = pd.read_csv(dataset_path, low_memory=False)
    feature_cols = list(pipe.named_steps["prep"].feature_names_in_)
    year_col = "base_year" if "base_year" in data.columns else "forecast_year"
    test = data[data[year_col].between(test_year_min, test_year_max)].dropna(subset=[target_col]).copy()
    positives = test[test[target_col].eq(1)]
    negatives = test[test[target_col].eq(0)].sample(
        n=min(max(len(positives) * 8, 2000), int(test[target_col].eq(0).sum())),
        random_state=42,
    )
    sample = pd.concat([positives, negatives], ignore_index=True).sample(frac=1, random_state=42)
    result = permutation_importance(
        pipe,
        sample[feature_cols],
        sample[target_col],
        scoring="average_precision",
        n_repeats=5,
        random_state=42,
        n_jobs=1,
    )
    frame = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": np.maximum(result.importances_mean, 0),
            "importance_std": result.importances_std,
        }
    ).sort_values("importance", ascending=False)
    total = frame["importance"].sum()
    frame["importance_share"] = frame["importance"] / total if total > 0 else 0
    frame["sample_rows"] = len(sample)
    frame["sample_positive_rate"] = sample[target_col].mean()
    frame.to_csv(REPORTS / output_name, index=False, encoding="utf-8-sig")
    return frame


def save_bar(frame: pd.DataFrame, title: str, output: Path) -> None:
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    top = frame.head(20).sort_values("importance_share")
    plt.figure(figsize=(9, 7))
    plt.barh(top["feature"], top["importance_share"] * 100, color="#2563eb")
    plt.title(title)
    plt.xlabel("Importance share (%)")
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def clean_change_feature_name(name: str) -> str:
    clean = name.replace("num__", "").replace("cat__", "")
    mapping = {
        "pop_15_19": "15~19세 인구",
        "net_migration_total": "순이동 인구",
        "out_migration_total": "전출 인구",
        "school_age_pop_growth_1yr": "전년 성장률",
        "pop_0_4": "0~4세 인구",
        "pop_10_14": "10~14세 인구",
        "pop_5_9": "5~9세 인구",
        "year": "연도",
        "out_migration_yoy_rate": "전출 변화율",
        "commercial_count": "상권 업종 수",
        "in_migration_total": "전입 인구",
        "education_business_count": "교육 업종 수",
        "kids_business_count": "아동 업종 수",
        "medical_business_count": "의료 업종 수",
        "birth_count": "출생아 수",
        "total_fertility_rate": "합계출산율",
    }
    if clean.startswith("sido_code_") or clean.startswith("sgg_code_"):
        return "시도/시군구 고정효과"
    return mapping.get(clean, clean)


def save_change_target_importance_chart() -> None:
    path = REPORTS / "change_target_regression_feature_importance.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    if frame.empty or "feature" not in frame.columns or "importance" not in frame.columns:
        return
    frame["feature_label"] = frame["feature"].map(clean_change_feature_name)
    grouped = frame.groupby("feature_label", as_index=False)["importance"].sum()
    grouped = grouped.sort_values("importance", ascending=False)
    total = grouped["importance"].sum()
    grouped["importance_share"] = grouped["importance"] / total if total else 0
    top = grouped.head(10).sort_values("importance_share")

    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    colors = ["#2563eb" if label in set(grouped.head(2)["feature_label"]) else "#93c5fd" for label in top["feature_label"]]
    plt.figure(figsize=(9, 5.8))
    bars = plt.barh(top["feature_label"], top["importance_share"] * 100, color=colors)
    for bar, value in zip(bars, top["importance_share"] * 100):
        plt.text(value + 0.6, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center", fontsize=9)
    plt.title("변화량 타깃 회귀모델 피처 중요도")
    plt.xlabel("Importance share (%)")
    plt.xlim(0, max(45, float((top["importance_share"] * 100).max()) + 8))
    plt.tight_layout()
    plt.savefig(FIGURES / "change_target_regression_feature_importance.png", dpi=160)
    plt.close()


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    regression = joblib.load(MODELS / "final_national_sgg_population_regressor.pkl")
    reg_imp = feature_importance(regression, "model")
    reg_imp.to_csv(REPORTS / "regression_feature_importance.csv", index=False, encoding="utf-8-sig")
    save_bar(reg_imp, "Regression Feature Importance", FIGURES / "regression_feature_importance.png")

    temporal_dataset_path = REPORTS / "_tmp_temporal_closure_classifier_dataset.csv"
    temporal_data = build_temporal_dataset()
    temporal_data = temporal_data[temporal_data["학제명"].isin(["초등학교", "중학교", "고등학교"])].copy()
    temporal_data.to_csv(temporal_dataset_path, index=False, encoding="utf-8-sig")

    context_classifier = joblib.load(MODELS / "tuned_histgb_temporal_closure.pkl")
    context_imp = permutation_feature_importance(
        context_classifier,
        temporal_dataset_path,
        "closed_next_year_proxy",
        2019,
        2022,
        "classification_temporal_closure_feature_importance.csv",
    )
    save_bar(
        context_imp,
        "Temporal Closure Classification Feature Importance",
        FIGURES / "classification_temporal_closure_feature_importance.png",
    )
    temporal_dataset_path.unlink(missing_ok=True)
    save_change_target_importance_chart()

    print("regression top features")
    print(reg_imp.head(15).to_string(index=False))
    print("\nfinal classification top features")
    print(context_imp.head(15).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

