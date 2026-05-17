from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "outputs" / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "outputs" / "reports"
FIGURES = ROOT / "outputs" / "figures"
PROCESSED = ROOT / "data" / "processed"


def set_korean_font() -> None:
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def build_model_comparison() -> pd.DataFrame:
    base = pd.read_csv(REPORTS / "edss_closure_classifier_metrics_national.csv")
    final = pd.read_csv(REPORTS / "final_supervised_closure_classifier_metrics.csv")
    rows = []
    rows.append(
        {
            "model_name": "EDSS 전용 객관 분류모델",
            "feature_scope": "EDSS 학교/학생/시설 피처",
            "train_years": base.loc[0, "train_base_years"],
            "test_years": base.loc[0, "test_base_years"],
            "roc_auc": base.loc[0, "roc_auc"],
            "pr_auc": base.loc[0, "pr_auc"],
            "precision_positive": base.loc[0, "tp"] / (base.loc[0, "tp"] + base.loc[0, "fp"]),
            "recall_positive": base.loc[0, "tp"] / (base.loc[0, "tp"] + base.loc[0, "fn"]),
            "f1_positive": 2
            * (base.loc[0, "tp"] / (base.loc[0, "tp"] + base.loc[0, "fp"]))
            * (base.loc[0, "tp"] / (base.loc[0, "tp"] + base.loc[0, "fn"]))
            / (
                (base.loc[0, "tp"] / (base.loc[0, "tp"] + base.loc[0, "fp"]))
                + (base.loc[0, "tp"] / (base.loc[0, "tp"] + base.loc[0, "fn"]))
            ),
            "tn": base.loc[0, "tn"],
            "fp": base.loc[0, "fp"],
            "fn": base.loc[0, "fn"],
            "tp": base.loc[0, "tp"],
            "decision": "최종 객관 성능 기준 모델로 사용",
        }
    )
    rows.append(
        {
            "model_name": "지역맥락 피처 추가 분류모델",
            "feature_scope": "EDSS + 출생/출산율/인구이동/상권 시도×연도 피처",
            "train_years": final.loc[0, "train_base_years"],
            "test_years": final.loc[0, "test_base_years"],
            "roc_auc": final.loc[0, "roc_auc"],
            "pr_auc": final.loc[0, "pr_auc"],
            "precision_positive": final.loc[0, "tp"] / (final.loc[0, "tp"] + final.loc[0, "fp"]),
            "recall_positive": final.loc[0, "tp"] / (final.loc[0, "tp"] + final.loc[0, "fn"]),
            "f1_positive": 2
            * (final.loc[0, "tp"] / (final.loc[0, "tp"] + final.loc[0, "fp"]))
            * (final.loc[0, "tp"] / (final.loc[0, "tp"] + final.loc[0, "fn"]))
            / (
                (final.loc[0, "tp"] / (final.loc[0, "tp"] + final.loc[0, "fp"]))
                + (final.loc[0, "tp"] / (final.loc[0, "tp"] + final.loc[0, "fn"]))
            ),
            "tn": final.loc[0, "tn"],
            "fp": final.loc[0, "fp"],
            "fn": final.loc[0, "fn"],
            "tp": final.loc[0, "tp"],
            "decision": "위험 후보 탐색 및 해석 보조 모델",
        }
    )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(REPORTS / "final_model_comparison.csv", index=False, encoding="utf-8-sig")
    return comparison


def plot_model_comparison(comparison: pd.DataFrame) -> None:
    metrics = ["roc_auc", "pr_auc", "precision_positive", "recall_positive", "f1_positive"]
    labels = ["ROC-AUC", "PR-AUC", "Precision", "Recall", "F1"]
    plot_df = comparison.set_index("model_name")[metrics].T
    plot_df.index = labels
    ax = plot_df.plot(kind="bar", figsize=(10, 5), color=["#2563eb", "#f97316"])
    ax.set_ylim(0, 1.05)
    ax.set_title("분류모델 성능 비교")
    ax.set_ylabel("score")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES / "model_comparison_metrics.png", dpi=170)
    plt.close()


def plot_regression_metrics() -> None:
    metrics = pd.read_csv(REPORTS / "final_national_population_regression_metrics.csv")
    metrics = metrics.sort_values("mae")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].bar(metrics["model"], metrics["mae"], color="#2563eb")
    axes[0].set_title("회귀모델 MAE 비교")
    axes[0].set_ylabel("MAE")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(metrics["model"], metrics["mape_safe_denominator_100"], color="#10b981")
    axes[1].set_title("회귀모델 Safe MAPE 비교")
    axes[1].set_ylabel("Safe MAPE")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES / "regression_model_metrics.png", dpi=170)
    plt.close()


def plot_risk_visuals() -> None:
    scenario = pd.read_csv(PROCESSED / "final_national_school_scenario_2026_2040.csv", low_memory=False)
    risk_order = ["low_risk", "mid_risk", "high_risk_review", "consolidation_high_risk", "education_gap_high_risk"]
    risk_ko = {
        "low_risk": "저위험",
        "mid_risk": "중위험",
        "high_risk_review": "고위험 검토",
        "consolidation_high_risk": "통폐합 가능",
        "education_gap_high_risk": "교육공백 우려",
    }

    count_by_year = (
        scenario.groupby(["forecast_year", "risk_label"]).size().unstack(fill_value=0).reindex(columns=risk_order, fill_value=0)
    )
    count_by_year.rename(columns=risk_ko).plot(kind="area", stacked=True, figsize=(10.5, 5.2), colormap="Set2")
    plt.title("연도별 위험군 분포")
    plt.xlabel("예측 연도")
    plt.ylabel("학교 수")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES / "risk_distribution_by_year.png", dpi=170)
    plt.close()

    high = scenario[
        scenario["forecast_year"].eq(2040)
        & scenario["risk_label"].isin(["consolidation_high_risk", "education_gap_high_risk", "high_risk_review"])
    ]
    sido_counts = high.groupby("requested_sido_name").size().sort_values(ascending=True)
    ax = sido_counts.plot(kind="barh", figsize=(9, 6), color="#ef4444")
    ax.set_title("2040 시도별 고위험 학교 수")
    ax.set_xlabel("학교 수")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES / "high_risk_by_sido_2040.png", dpi=170)
    plt.close()

    factors = (
        high.groupby("risk_label")[
            ["school_isolation_score", "commercial_vulnerability_score", "regional_decline_risk_score", "objective_closure_percentile"]
        ]
        .mean()
        .rename(index=risk_ko)
    )
    factors.plot(kind="bar", figsize=(10.5, 5.2), color=["#6366f1", "#f59e0b", "#10b981", "#ef4444"])
    plt.title("2040 고위험 유형별 평균 위험요인")
    plt.ylabel("평균 점수")
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES / "risk_factor_profile_2040.png", dpi=170)
    plt.close()


def plot_target_design_feature_comparison() -> None:
    abs_path = REPORTS / "regression_feature_importance.csv"
    change_path = REPORTS / "change_target_regression_feature_importance.csv"
    if not abs_path.exists() or not change_path.exists():
        return

    abs_imp = pd.read_csv(abs_path).head(5)
    change_imp = pd.read_csv(change_path).head(5)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, frame, title, color in [
        (axes[0], abs_imp, "절대값 타겟", "#2563eb"),
        (axes[1], change_imp, "변화량 타겟", "#f97316"),
    ]:
        plot_df = frame.iloc[::-1].copy()
        ax.barh(plot_df["feature"], plot_df["importance"], color=color)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("importance")
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("타겟 설계가 피처 중요도를 결정한다", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "target_design_feature_comparison.png", dpi=170)
    plt.close(fig)


def _model_sido_year_totals() -> pd.DataFrame:
    scenario = pd.read_csv(PROCESSED / "final_national_school_scenario_2026_2040.csv", low_memory=False)
    pressure_col = (
        "pressure_model_forecast_student_count"
        if "pressure_model_forecast_student_count" in scenario.columns
        else "forecast_student_count"
    )
    base = (
        scenario.groupby(["requested_sido_name", "forecast_year"], as_index=False)
        .agg(
            abs_model_students=(pressure_col, "sum"),
            cohort_students=("forecast_student_count", "sum"),
            student_2025=("student_count_2025", "sum"),
        )
        .rename(columns={"requested_sido_name": "sido", "forecast_year": "year"})
    )
    change_path = PROCESSED / "final_national_school_scenario_change_model_2026_2040.csv"
    if change_path.exists():
        change = (
            pd.read_csv(change_path, low_memory=False)
            .groupby(["requested_sido_name", "forecast_year"], as_index=False)
            .agg(change_model_students=("forecast_student_count", "sum"))
            .rename(columns={"requested_sido_name": "sido", "forecast_year": "year"})
        )
        base = base.merge(change, on=["sido", "year"], how="left")
    else:
        base["change_model_students"] = pd.NA
    return base


def plot_migration_effect_comparison() -> None:
    totals = _model_sido_year_totals()
    selected = ["세종", "경기", "전남", "경북"]
    totals = totals[totals["sido"].isin(selected)].copy()
    if totals.empty:
        return

    model_cols = [
        ("abs_model_students", "압력비", "#64748b", "-"),
        ("change_model_students", "변화량", "#f97316", "-"),
        ("cohort_students", "코호트", "#dc2626", "-"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), sharex=True)
    axes = axes.flatten()
    for ax, sido in zip(axes, selected):
        sub = totals[totals["sido"].eq(sido)].sort_values("year")
        if sub.empty:
            continue
        base = float(sub["student_2025"].iloc[0])
        for col, label, color, linestyle in model_cols:
            if col not in sub or sub[col].isna().all():
                continue
            ax.plot(sub["year"], (sub[col] / base - 1) * 100, label=label, color=color, linestyle=linestyle, linewidth=2)
        ax.axhline(0, color="#94a3b8", linewidth=1, linestyle="--")
        ax.set_title(sido, fontweight="bold")
        ax.set_ylabel("2025 대비 변화율(%)")
        ax.grid(alpha=0.25)
    axes[0].legend(loc="best")
    fig.suptitle("시도별 감소율 차이 - 모델이 인구이동을 반영하는가?", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "migration_effect_comparison.png", dpi=170)
    plt.close(fig)


def build_regional_differentiation_summary() -> None:
    totals = _model_sido_year_totals()
    df2040 = totals[totals["year"].eq(2040)].copy()
    models = [
        ("절대값/압력비 모델", "다음 해 시군구 학령인구 절대값", "현재 학령인구 규모", "abs_model_students"),
        ("변화량 모델", "다음 해 시군구 학령인구 증감량", "인구이동, 연령구조", "change_model_students"),
        ("출생 코호트 기준", "학교급별 출생 코호트 진입", "출생아수, 학교급 진입 구조", "cohort_students"),
    ]
    rows = []
    for model, target, features, col in models:
        if col not in df2040 or df2040[col].isna().all():
            continue
        decrease = (df2040[col] / df2040["student_2025"].replace(0, pd.NA) - 1) * 100
        national_base = df2040["student_2025"].sum()
        national_forecast = df2040[col].sum()
        rows.append(
            {
                "model": model,
                "target": target,
                "key_features": features,
                "national_2040_change_pct": (national_forecast / national_base - 1) * 100,
                "sido_change_pct_std": decrease.std(),
            }
        )
    pd.DataFrame(rows).to_csv(
        REPORTS / "model_regional_differentiation_summary.csv", index=False, encoding="utf-8-sig"
    )


def write_markdown(comparison: pd.DataFrame) -> None:
    base = comparison.iloc[0]
    ctx = comparison.iloc[1]
    problem_md = f"""# 프로젝트 문제 정리

## 1. 데이터 수집 과정의 문제

### 행정코드 변경

전국 상권 데이터 수집 중 강원과 전북이 비어 있는 문제가 있었다.

```text
문제 파일:
data/raw/small_shop_42_강원.csv
data/raw/small_shop_45_전북.csv
```

원인은 행정코드 변경이었다. 실제 사용 파일은 아래와 같다.

```text
정상 파일:
data/raw/small_shop_51_강원.csv
data/raw/small_shop_52_전북.csv
```

### KOSIS 셀 제한

KOSIS 학령인구 데이터를 전국 시군구 전체로 한 번에 요청했을 때 40,000셀 제한에 걸렸다. 시군구 코드를 청크로 나누어 수집하도록 수정했다.

### API 일시 오류

소상공인 상권 API 수집 중 HTTP 502 오류가 발생했다. 페이지 단위 중간 저장과 재시도 로직을 추가해 해결했다.

## 2. 모델링 과정의 문제

### EDSS 과거 패널의 공간 정보 부족

EDSS 2009~2023 과거 패널에는 학교별 좌표와 시군구 코드가 없다. 따라서 과거 supervised 분류모델에는 시군구 단위 피처를 직접 붙일 수 없었다.

해결 방식:

```text
과거 분류모델: 시도×연도 단위 지역 피처 결합
현재 학교 시나리오: 학교별 sggCd 기반 시군구 피처 결합
```

### 새 피처 추가가 성능을 항상 올리지는 않음

출생·출산율·인구이동·상권 피처를 추가한 supervised 분류모델은 recall은 상승했지만 precision, F1, PR-AUC는 소폭 하락했다.

해석:

```text
새 모델은 위험 후보를 더 넓게 잡는다.
하지만 오탐도 늘어난다.
따라서 최종 객관 성능 모델은 EDSS 전용 모델을 기준으로 두고,
새 피처는 위험등급 해석과 정책적 근거 강화에 활용하는 것이 안전하다.
```

### MAPE 폭발 문제

일부 시군구의 실제 학령인구가 0 또는 매우 작은 값이라 raw MAPE가 비정상적으로 커졌다. 보고서에는 분모를 최소 100명으로 제한한 `mape_safe_denominator_100`을 함께 사용한다.

## 3. 현재 가장 안전한 결론

```text
회귀모델:
KOSIS 학령인구/출생/출산율/인구이동/상권 피처 기반 시군구 학령인구 예측

객관 분류모델:
EDSS 전용 supervised 분류모델

최종 위험등급:
회귀 예측 학생수
+ EDSS 객관 폐교위험
+ 학교 고립도
+ 상권 취약도
+ 지역 감소위험
을 결합한 정책 해석형 분류
```

## 4. 변화량 모델 통합 후 해석

장기 예측에서 기존 절대값/압력비 모델과 출생 코호트 모델의 차이가 커지는 문제를 설명하기 위해 변화량 모델을 추가했다.

```text
절대값 모델:
다음 해 학령인구 규모 자체를 예측한다. 현재 인구 규모의 영향이 매우 크다.

변화량 모델:
다음 해 학령인구 증감량을 예측한다. 인구이동과 연령구조의 영향이 더 잘 드러난다.

출생 코호트 모델:
출생아수 감소가 초중고 학교급에 순차적으로 진입하는 장기 구조를 반영한다.
```

변화량 모델은 장기 반복 예측에서 발산 위험이 있어 연간 변화량을 `-20% ~ +10%`로 제한했다. 이 모델은 최종 정답 모델이라기보다, “인구이동이 많은 지역과 유출 지역의 감소 속도가 다르게 나타난다”는 점을 보여주는 발표용 비교 모델로 사용한다.

2040 전국 학생수 합계:

```text
절대값/압력비 모델: 3,679,172명
변화량 모델: 2,648,950명
출생 코호트 기준: 2,898,445명
```

지역 차별화 정도는 시도별 감소율 표준편차로 비교했다.

```text
절대값/압력비 모델: 10.2
변화량 모델: 21.9
출생 코호트 기준: 14.1
```

따라서 발표에서는 “정확도 하나로 최종 모델을 결정했다”가 아니라, “절대값 모델은 단기 예측 성능, 변화량 모델은 인구이동 기반 지역 차별화, 코호트 모델은 출생아 감소의 장기 구조”로 역할을 나눠 설명하는 것이 안전하다.
"""
    (ROOT / "프로젝트_문제정리.md").write_text(problem_md, encoding="utf-8")

    compare_md = f"""# 모델 비교 리포트

## 1. 비교 대상

| 모델 | 입력 피처 | 역할 |
|---|---|---|
| EDSS 전용 객관 분류모델 | EDSS 학교/학생/시설 피처 | 최종 객관 성능 기준 |
| 지역맥락 피처 추가 분류모델 | EDSS + 출생/출산율/인구이동/상권 | 위험 후보 탐색 및 해석 보조 |

## 2. 성능 비교

| 지표 | EDSS 전용 | 지역맥락 추가 |
|---|---:|---:|
| ROC-AUC | {base['roc_auc']:.4f} | {ctx['roc_auc']:.4f} |
| PR-AUC | {base['pr_auc']:.4f} | {ctx['pr_auc']:.4f} |
| Precision | {base['precision_positive']:.4f} | {ctx['precision_positive']:.4f} |
| Recall | {base['recall_positive']:.4f} | {ctx['recall_positive']:.4f} |
| F1 | {base['f1_positive']:.4f} | {ctx['f1_positive']:.4f} |
| TP | {int(base['tp'])} | {int(ctx['tp'])} |
| FP | {int(base['fp'])} | {int(ctx['fp'])} |
| FN | {int(base['fn'])} | {int(ctx['fn'])} |

## 3. 해석

지역맥락 피처 추가 모델은 recall이 상승했지만 precision과 F1이 하락했다.

```text
EDSS 전용 모델:
정밀도가 높고 오탐이 적다.

지역맥락 추가 모델:
더 많은 위험 후보를 잡지만 오탐이 늘어난다.
```

따라서 최종 발표에서는 다음 구조가 가장 설득력 있다.

```text
객관 성능 기준: EDSS 전용 supervised 분류모델
차별점/정책 해석: 상권, 인구이동, 출산율, 학교 고립도 피처
최종 위험등급: 객관 위험확률 + 해석 피처 결합
```

## 4. 생성된 시각화

```text
outputs/figures/model_comparison_metrics.png
outputs/figures/regression_model_metrics.png
outputs/figures/risk_distribution_by_year.png
outputs/figures/high_risk_by_sido_2040.png
outputs/figures/risk_factor_profile_2040.png
outputs/maps/final_national_school_risk_2040.html
```
"""
    (ROOT / "모델_비교_리포트.md").write_text(compare_md, encoding="utf-8")


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    set_korean_font()
    comparison = build_model_comparison()
    plot_model_comparison(comparison)
    plot_regression_metrics()
    plot_risk_visuals()
    plot_target_design_feature_comparison()
    plot_migration_effect_comparison()
    build_regional_differentiation_summary()
    write_markdown(comparison)
    print("saved reports and figures")
    print(comparison.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
