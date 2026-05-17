from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "outputs" / "reports"
FIGURES = ROOT / "outputs" / "figures"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OLD_TO_CURRENT_SIDO = {
    "11": "11",
    "21": "26",
    "22": "27",
    "23": "28",
    "24": "29",
    "25": "30",
    "26": "31",
    "29": "36",
    "31": "41",
    "32": "51",
    "33": "43",
    "34": "44",
    "35": "52",
    "36": "46",
    "37": "47",
    "38": "48",
    "39": "50",
}


def normalize_sido_code(series: pd.Series) -> pd.Series:
    codes = series.astype(str).str.zfill(2)
    return codes.map(OLD_TO_CURRENT_SIDO).fillna(codes)


def make_panel() -> pd.DataFrame:
    pop = pd.read_csv(PROCESSED / "national_population_features_sgg.csv", low_memory=False)
    pop = pop[pop["month"].eq(4)].copy()
    pop["sido_code"] = pop["sido_code"].astype(str).str.zfill(2)
    pop_sido = (
        pop.groupby(["sido_code", "year"], as_index=False)
        .agg(
            pop_0_4=("pop_0_4", "sum"),
            pop_5_9=("pop_5_9", "sum"),
            pop_10_14=("pop_10_14", "sum"),
            pop_15_19=("pop_15_19", "sum"),
            school_age_pop_0_19=("school_age_pop_0_19", "sum"),
        )
        .sort_values(["sido_code", "year"])
    )

    birth = pd.read_csv(PROCESSED / "national_birth_features_sgg.csv", low_memory=False)
    birth["sido_code"] = normalize_sido_code(birth["sido_code"])
    birth_sido = (
        birth.groupby(["sido_code", "year"], as_index=False)
        .agg(
            birth_count=("birth_count", "sum"),
            total_fertility_rate=("total_fertility_rate", "mean"),
        )
        .sort_values(["sido_code", "year"])
    )
    birth_sido["birth_count_roll5"] = birth_sido.groupby("sido_code")["birth_count"].transform(
        lambda s: s.rolling(5, min_periods=3).sum()
    )
    birth_sido["tfr_roll5_mean"] = birth_sido.groupby("sido_code")["total_fertility_rate"].transform(
        lambda s: s.rolling(5, min_periods=3).mean()
    )

    migration = pd.read_csv(PROCESSED / "national_migration_features_sgg.csv", low_memory=False)
    migration["sido_code"] = migration["sido_code"].astype(str).str.zfill(2)
    mig_sido = (
        migration.groupby(["sido_code", "year"], as_index=False)
        .agg(
            net_migration_total=("net_migration_total", "sum"),
            in_migration_total=("in_migration_total", "sum"),
            out_migration_total=("out_migration_total", "sum"),
        )
        .sort_values(["sido_code", "year"])
    )
    mig_sido["net_migration_roll5"] = mig_sido.groupby("sido_code")["net_migration_total"].transform(
        lambda s: s.rolling(5, min_periods=3).sum()
    )

    panel = pop_sido.merge(birth_sido, on=["sido_code", "year"], how="left").merge(
        mig_sido, on=["sido_code", "year"], how="left"
    )
    return panel


def model_importance(panel: pd.DataFrame, target: str, features: list[str], model_name: str) -> tuple[pd.DataFrame, dict]:
    data = panel.dropna(subset=[target]).copy()
    train = data[data["year"].le(2020)].copy()
    test = data[data["year"].between(2021, 2023)].copy()
    cat = [c for c in features if c == "sido_code"]
    num = [c for c in features if c not in cat]
    prep = ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat,
            ),
        ]
    )
    pipe = Pipeline(
        [
            ("prep", prep),
            ("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1)),
        ]
    )
    pipe.fit(train[features], train[target])
    pred = pipe.predict(test[features])
    metrics = {
        "model_name": model_name,
        "target": target,
        "train_years": "2016-2020",
        "test_years": "2021-2023",
        "rows_train": len(train),
        "rows_test": len(test),
        "mae": mean_absolute_error(test[target], pred),
        "r2": r2_score(test[target], pred),
    }
    perm = permutation_importance(
        pipe,
        test[features],
        test[target],
        scoring="neg_mean_absolute_error",
        n_repeats=8,
        random_state=42,
        n_jobs=1,
    )
    imp = pd.DataFrame(
        {
            "model_name": model_name,
            "target": target,
            "feature": features,
            "importance_mae_reduction": np.maximum(perm.importances_mean, 0),
            "importance_std": perm.importances_std,
        }
    )
    total = imp["importance_mae_reduction"].sum()
    imp["importance_share"] = imp["importance_mae_reduction"] / total if total > 0 else 0
    return imp.sort_values("importance_share", ascending=False), metrics


def ridge_coefficients(panel: pd.DataFrame) -> pd.DataFrame:
    features = ["birth_count_roll5", "tfr_roll5_mean", "net_migration_roll5", "year"]
    target = "pop_0_4"
    data = panel.dropna(subset=[target]).copy()
    pipe = Pipeline(
        [
            ("prep", ColumnTransformer([("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), features)])),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    pipe.fit(data[features], data[target])
    return pd.DataFrame(
        {
            "target": target,
            "feature": features,
            "standardized_coefficient": np.ravel(pipe.named_steps["model"].coef_),
        }
    ).sort_values("standardized_coefficient", key=lambda s: s.abs(), ascending=False)


def save_pathway_figure(panel: pd.DataFrame) -> None:
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    yearly = (
        panel.groupby("year", as_index=False)
        .agg(
            total_birth_count=("birth_count", "sum"),
            avg_tfr=("total_fertility_rate", "mean"),
            pop_0_4=("pop_0_4", "sum"),
            school_age_pop_0_19=("school_age_pop_0_19", "sum"),
        )
        .dropna()
    )
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    ax1.plot(yearly["year"], yearly["total_birth_count"], marker="o", color="#2563eb", label="출생아 수")
    ax1.plot(yearly["year"], yearly["pop_0_4"], marker="o", color="#16a34a", label="0~4세 인구")
    ax2.plot(yearly["year"], yearly["avg_tfr"], marker="o", color="#dc2626", label="합계출산율")
    ax1.set_ylabel("인구/출생아 수")
    ax2.set_ylabel("합계출산율")
    ax1.set_title("출산율 -> 출생아 수 -> 0~4세 인구 경로")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "fertility_pathway_trend.png", dpi=160)
    plt.close(fig)


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    panel = make_panel()
    panel.to_csv(REPORTS / "fertility_pathway_sido_panel.csv", index=False, encoding="utf-8-sig")

    tasks = [
        (
            "birth_from_fertility",
            "birth_count",
            ["total_fertility_rate", "net_migration_total", "year", "sido_code"],
        ),
        (
            "child_pop_from_births",
            "pop_0_4",
            ["birth_count_roll5", "tfr_roll5_mean", "net_migration_roll5", "year", "sido_code"],
        ),
        (
            "school_age_from_age_structure",
            "school_age_pop_0_19",
            ["pop_0_4", "pop_5_9", "pop_10_14", "pop_15_19", "birth_count_roll5", "tfr_roll5_mean", "sido_code"],
        ),
    ]
    importance_frames = []
    metric_rows = []
    for model_name, target, features in tasks:
        imp, metrics = model_importance(panel, target, features, model_name)
        importance_frames.append(imp)
        metric_rows.append(metrics)
    importance = pd.concat(importance_frames, ignore_index=True)
    importance.to_csv(REPORTS / "fertility_pathway_feature_importance.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(metric_rows).to_csv(REPORTS / "fertility_pathway_model_metrics.csv", index=False, encoding="utf-8-sig")
    ridge_coefficients(panel).to_csv(REPORTS / "fertility_pathway_ridge_coefficients.csv", index=False, encoding="utf-8-sig")
    save_pathway_figure(panel)

    summary = """# 출산율 경로 보조 분석

## 목적

최종 회귀모델의 직접 중요도에서는 현재 학령인구와 연령대별 인구가 크게 나타난다. 이는 출산율이 중요하지 않다는 뜻이 아니라, 출산율의 영향이 이미 출생아 수와 0~4세 인구에 반영되어 있기 때문이다.

## 해석 구조

```text
출산율 하락
-> 출생아 수 감소
-> 0~4세 인구 감소
-> 5~9세, 10~14세, 15~19세 구조 변화
-> 학교별 학생수 감소 압력 증가
```

## 산출물

```text
outputs/reports/fertility_pathway_feature_importance.csv
outputs/reports/fertility_pathway_model_metrics.csv
outputs/reports/fertility_pathway_ridge_coefficients.csv
outputs/figures/fertility_pathway_trend.png
```

## 발표용 문장

출산율은 장기 학령인구 감소의 원인 변수이고, 현재 아동 연령대별 인구는 그 결과가 누적된 직접 예측 변수이다. 따라서 최종 예측모델의 중요도에서는 출산율보다 현재 0~19세 및 0~4세, 5~9세 인구가 크게 나타난다.
"""
    (ROOT / "출산율_경로_분석.md").write_text(summary, encoding="utf-8")
    print(pd.DataFrame(metric_rows).to_string(index=False))
    print(importance.groupby("model_name").head(5).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
