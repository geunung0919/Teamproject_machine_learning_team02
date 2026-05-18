from __future__ import annotations

import os
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

ROOT = SRC.parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "outputs" / "reports"
FIGURES = ROOT / "outputs" / "figures"


def set_korean_font() -> None:
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def load_yearly_comparison() -> pd.DataFrame:
    school_scenario = pd.read_csv(PROCESSED / "final_national_school_scenario_2026_2040.csv", low_memory=False)
    pressure_col = (
        "pressure_model_forecast_student_count"
        if "pressure_model_forecast_student_count" in school_scenario.columns
        else "forecast_student_count"
    )
    pressure = (
        school_scenario.groupby("forecast_year", as_index=False)
        .agg(
            sgg_model_students=(pressure_col, "sum"),
            student_2025=("student_count_2025", "sum"),
        )
        .sort_values("forecast_year")
    )

    change_path = PROCESSED / "final_national_school_scenario_change_model_2026_2040.csv"
    if change_path.exists():
        change = (
            pd.read_csv(change_path, low_memory=False)
            .groupby("forecast_year", as_index=False)
            .agg(change_model_students=("forecast_student_count", "sum"))
        )
    else:
        change = pressure[["forecast_year"]].copy()
        change["change_model_students"] = pd.NA

    cohort = pd.read_csv(REPORTS / "school_level_cohort_scenario_total_summary.csv", low_memory=False)
    cohort_wide = cohort.pivot_table(
        index="forecast_year",
        columns="cohort_scenario",
        values="cohort_forecast_students",
        aggfunc="sum",
    ).reset_index()
    cohort_wide = cohort_wide.rename(
        columns={
            "baseline": "cohort_baseline_students",
            "optimistic": "cohort_optimistic_students",
            "pessimistic": "cohort_pessimistic_students",
        }
    )

    comparison = pressure.merge(change, on="forecast_year", how="left").merge(cohort_wide, on="forecast_year", how="left")
    comparison["sgg_vs_cohort_gap_pct"] = (
        (comparison["sgg_model_students"] - comparison["cohort_baseline_students"])
        / comparison["cohort_baseline_students"].replace(0, pd.NA)
        * 100
    )
    comparison.to_csv(REPORTS / "model_vs_cohort_comparison.csv", index=False, encoding="utf-8-sig")
    return comparison


def plot_three_model_lines(comparison: pd.DataFrame) -> None:
    base_2025 = float(comparison["student_2025"].iloc[0])
    years = [2025, *comparison["forecast_year"].astype(int).tolist()]

    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.plot(
        years,
        [base_2025, *comparison["sgg_model_students"].tolist()],
        marker="o",
        linewidth=2.4,
        color="#64748b",
        label="절대값 모델(시군구 압력비)",
    )
    if comparison["change_model_students"].notna().any():
        ax.plot(
            years,
            [base_2025, *comparison["change_model_students"].tolist()],
            marker="o",
            linewidth=2.4,
            color="#f97316",
            label="변화량 모델(인구이동 반영)",
        )
    ax.plot(
        years,
        [base_2025, *comparison["cohort_baseline_students"].tolist()],
        marker="o",
        linewidth=2.4,
        color="#dc2626",
        label="출생 코호트 기준",
    )
    ax.plot(
        years,
        [base_2025, *comparison["cohort_optimistic_students"].tolist()],
        linestyle="--",
        linewidth=1.9,
        color="#16a34a",
        label="출생 코호트 낙관",
    )
    ax.plot(
        years,
        [base_2025, *comparison["cohort_pessimistic_students"].tolist()],
        linestyle="--",
        linewidth=1.9,
        color="#b45309",
        label="출생 코호트 비관",
    )
    ax.set_title("3개 예측 모델 전국 학생수 비교 (2026~2040)", fontsize=15, fontweight="bold")
    ax.set_xlabel("연도")
    ax.set_ylabel("전국 학생수 합계")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    ax.yaxis.set_major_formatter(lambda value, _: f"{int(value):,}")
    fig.tight_layout()
    fig.savefig(FIGURES / "three_model_comparison.png", dpi=170)
    fig.savefig(FIGURES / "model_vs_cohort_comparison.png", dpi=170)
    plt.close(fig)


def build_sido_comparison() -> pd.DataFrame:
    years = [2026, 2030, 2035, 2040]
    scenario = pd.read_csv(PROCESSED / "final_national_school_scenario_2026_2040.csv", low_memory=False)
    pressure_col = (
        "pressure_model_forecast_student_count"
        if "pressure_model_forecast_student_count" in scenario.columns
        else "forecast_student_count"
    )
    pressure = (
        scenario[scenario["forecast_year"].isin(years)]
        .groupby(["requested_sido_name", "forecast_year"], as_index=False)
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
            .query("forecast_year in @years")
            .groupby(["requested_sido_name", "forecast_year"], as_index=False)
            .agg(change_model_students=("forecast_student_count", "sum"))
            .rename(columns={"requested_sido_name": "sido", "forecast_year": "year"})
        )
        out = pressure.merge(change, on=["sido", "year"], how="left")
    else:
        out = pressure.copy()
        out["change_model_students"] = pd.NA
    out = out[["sido", "year", "abs_model_students", "change_model_students", "cohort_students", "student_2025"]]
    out.to_csv(REPORTS / "three_model_comparison_by_sido.csv", index=False, encoding="utf-8-sig")
    return out


def plot_sido_decrease_heatmap(sido_comparison: pd.DataFrame) -> None:
    df2040 = sido_comparison[sido_comparison["year"].eq(2040)].copy()
    model_cols = {
        "절대값 모델": "abs_model_students",
        "변화량 모델": "change_model_students",
        "출생 코호트": "cohort_students",
    }
    rows = []
    for model_name, col in model_cols.items():
        if col not in df2040 or df2040[col].isna().all():
            continue
        row = {"model": model_name}
        for _, r in df2040.iterrows():
            base = r["student_2025"]
            row[r["sido"]] = (r[col] / base - 1) * 100 if base else 0
        rows.append(row)
    heat = pd.DataFrame(rows).set_index("model")
    heat = heat.reindex(sorted(heat.columns), axis=1)

    fig, ax = plt.subplots(figsize=(12.5, 3.8))
    values = heat.to_numpy(dtype=float)
    im = ax.imshow(values, aspect="auto", cmap="Reds_r", vmin=-45, vmax=10)
    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels(heat.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(heat.index)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.1f}%", ha="center", va="center", fontsize=8, color="#111827")
    ax.set_title("시도별 2040 학생수 감소율 - 모델 간 비교", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="2025 대비 변화율(%)")
    fig.tight_layout()
    fig.savefig(FIGURES / "sido_decrease_heatmap_by_model.png", dpi=170)
    plt.close(fig)


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    set_korean_font()

    comparison = load_yearly_comparison()
    plot_three_model_lines(comparison)
    sido_comparison = build_sido_comparison()
    plot_sido_decrease_heatmap(sido_comparison)

    last = comparison[comparison["forecast_year"].eq(2040)].iloc[0]
    print("saved:", REPORTS / "model_vs_cohort_comparison.csv")
    print("saved:", FIGURES / "three_model_comparison.png")
    print("saved:", REPORTS / "three_model_comparison_by_sido.csv")
    print("saved:", FIGURES / "sido_decrease_heatmap_by_model.png")
    print(
        "2040 comparison:",
        f"abs={last['sgg_model_students']:,.0f}",
        f"change={last['change_model_students']:,.0f}",
        f"cohort={last['cohort_baseline_students']:,.0f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

