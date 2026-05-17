from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "outputs" / "reports"

YEARS = list(range(2026, 2041))
SCENARIOS = {
    "pessimistic": {"birth_factor": 0.90, "migration_mode": "pessimistic"},
    "baseline": {"birth_factor": 1.00, "migration_mode": "baseline"},
    "optimistic": {"birth_factor": 1.10, "migration_mode": "optimistic"},
}

SIDO_SHORT_NAME = {
    "서울특별시": "서울",
    "부산광역시": "부산",
    "대구광역시": "대구",
    "인천광역시": "인천",
    "광주광역시": "광주",
    "대전광역시": "대전",
    "울산광역시": "울산",
    "세종특별자치시": "세종",
    "경기도": "경기",
    "강원특별자치도": "강원",
    "충청북도": "충북",
    "충청남도": "충남",
    "전북특별자치도": "전북",
    "전라남도": "전남",
    "경상북도": "경북",
    "경상남도": "경남",
    "제주특별자치도": "제주",
}


def level_birth_years(level: str, forecast_year: int) -> list[int]:
    # Korean school year approximation: 초1~6 = age 7~12, 중1~3 = 13~15, 고1~3 = 16~18.
    if level == "초등학교":
        return list(range(forecast_year - 12, forecast_year - 6))
    if level == "중학교":
        return list(range(forecast_year - 15, forecast_year - 12))
    if level == "고등학교":
        return list(range(forecast_year - 18, forecast_year - 15))
    return []


def build_birth_lookup(scenario_factor: float) -> pd.DataFrame:
    raw = pd.read_csv(ROOT / "data" / "raw" / "national_kosis_birth_tfr_sgg.csv", low_memory=False)
    birth = raw[(raw["region_level"].eq("sido")) & (raw["item_code"].eq("T1")) & (raw["region_name"].ne("전국"))].copy()
    birth["sido_name"] = birth["region_name"].map(SIDO_SHORT_NAME)
    birth = birth[["sido_name", "year", "value"]].rename(columns={"value": "birth_count"})
    birth["birth_count"] = pd.to_numeric(birth["birth_count"], errors="coerce")

    future_rows = []
    latest_mean = (
        birth[birth["year"].between(2023, 2025)]
        .groupby(["sido_name"], as_index=False)["birth_count"]
        .mean()
    )
    for year in range(2026, 2034):
        future = latest_mean.copy()
        future["year"] = year
        future["birth_count"] = future["birth_count"] * scenario_factor
        future_rows.append(future)
    if future_rows:
        birth = pd.concat([birth, *future_rows], ignore_index=True)
    return birth


def build_sido_migration_rate(schools: pd.DataFrame) -> pd.Series:
    if {"requested_sido_name", "net_migration_total", "school_age_pop_0_19"}.issubset(schools.columns):
        base = schools.copy()
        base["net_migration_total"] = pd.to_numeric(base["net_migration_total"], errors="coerce").fillna(0)
        base["school_age_pop_0_19"] = pd.to_numeric(base["school_age_pop_0_19"], errors="coerce").fillna(0)
        grouped = base.groupby("requested_sido_name", as_index=True).agg(
            net_migration_total=("net_migration_total", "sum"),
            school_age_pop_0_19=("school_age_pop_0_19", "sum"),
        )
        rate = grouped["net_migration_total"] / grouped["school_age_pop_0_19"].replace(0, np.nan)
        return rate.fillna(0).clip(-0.08, 0.08)
    return pd.Series(dtype=float)


def adjusted_migration_rate(rate: float, mode: str) -> float:
    if mode == "optimistic":
        return rate * (1.10 if rate >= 0 else 0.90)
    if mode == "pessimistic":
        return rate * (0.90 if rate >= 0 else 1.10)
    return rate


def cohort_sum(birth: pd.DataFrame, sgg_code: str, years: list[int]) -> float:
    if not years:
        return np.nan
    values = birth[(birth["sgg_code"].eq(sgg_code)) & (birth["year"].isin(years))]["birth_count"]
    if values.empty:
        return np.nan
    return float(values.sum())


def build_scenario_one(scenario_name: str, scenario_config: dict[str, float | str]) -> pd.DataFrame:
    schools = pd.read_csv(PROCESSED / "final_national_current_school_features.csv", low_memory=False)
    schools["sgg_code"] = schools["sgg_code"].astype(str).str.zfill(5)
    schools["student_count_2025"] = pd.to_numeric(schools["student_count_2025"], errors="coerce")
    birth = build_birth_lookup(float(scenario_config["birth_factor"]))
    birth_index = birth.set_index(["sido_name", "year"])["birth_count"].sort_index()
    migration_rate = build_sido_migration_rate(schools)
    migration_mode = str(scenario_config["migration_mode"])

    rows = []
    for year in YEARS:
        df = schools.copy()
        df["forecast_year"] = year
        df["cohort_scenario"] = scenario_name
        df["baseline_birth_years"] = df["school_level"].map(lambda level: level_birth_years(level, 2025))
        df["forecast_birth_years"] = df["school_level"].map(lambda level: level_birth_years(level, year))

        def sum_from_index(row: pd.Series, years_col: str) -> float:
            years = row[years_col]
            if not years:
                return np.nan
            total = 0.0
            found = False
            for birth_year in years:
                key = (row["requested_sido_name"], birth_year)
                if key in birth_index.index:
                    total += float(birth_index.loc[key])
                    found = True
            return total if found else np.nan

        df["baseline_cohort_births"] = df.apply(sum_from_index, axis=1, years_col="baseline_birth_years")
        df["forecast_cohort_births"] = df.apply(sum_from_index, axis=1, years_col="forecast_birth_years")
        df["cohort_pressure_ratio"] = (
            df["forecast_cohort_births"] / df["baseline_cohort_births"].replace(0, np.nan)
        ).clip(0.15, 1.8)
        df["cohort_pressure_ratio"] = df["cohort_pressure_ratio"].fillna(1.0)
        df["sido_net_migration_rate"] = df["requested_sido_name"].map(migration_rate).fillna(0)
        df["adjusted_net_migration_rate"] = df["sido_net_migration_rate"].map(
            lambda rate: adjusted_migration_rate(float(rate), migration_mode)
        )
        df["migration_adjustment"] = (
            1 + df["adjusted_net_migration_rate"] * max(year - 2025, 0) * 0.1
        ).clip(0.85, 1.15)
        df["cohort_pressure_ratio"] = (df["cohort_pressure_ratio"] * df["migration_adjustment"]).clip(0.15, 1.8)
        df["cohort_forecast_student_count"] = (
            df["student_count_2025"].fillna(0) * df["cohort_pressure_ratio"]
        ).round().clip(lower=0)
        rows.append(
            df[
                [
                    "cohort_scenario",
                    "forecast_year",
                    "requested_sido_name",
                    "sgg_code",
                    "schlCd",
                    "schlNm",
                    "school_level",
                    "student_count_2025",
                    "baseline_birth_years",
                    "forecast_birth_years",
                    "baseline_cohort_births",
                    "forecast_cohort_births",
                    "sido_net_migration_rate",
                    "adjusted_net_migration_rate",
                    "migration_adjustment",
                    "cohort_pressure_ratio",
                    "cohort_forecast_student_count",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    scenario = pd.concat(
        [build_scenario_one(name, config) for name, config in SCENARIOS.items()],
        ignore_index=True,
    )
    scenario.to_csv(PROCESSED / "school_level_cohort_scenario_2026_2040.csv", index=False, encoding="utf-8-sig")

    summary = (
        scenario.groupby(["cohort_scenario", "forecast_year", "school_level"], as_index=False)
        .agg(
            school_count=("schlCd", "count"),
            student_2025=("student_count_2025", "sum"),
            cohort_forecast_students=("cohort_forecast_student_count", "sum"),
            avg_migration_adjustment=("migration_adjustment", "mean"),
            avg_cohort_pressure=("cohort_pressure_ratio", "mean"),
        )
    )
    summary["change_pct"] = (summary["cohort_forecast_students"] / summary["student_2025"].replace(0, np.nan) - 1) * 100
    summary.to_csv(REPORTS / "school_level_cohort_scenario_summary.csv", index=False, encoding="utf-8-sig")

    total = (
        scenario.groupby(["cohort_scenario", "forecast_year"], as_index=False)
        .agg(
            school_count=("schlCd", "count"),
            student_2025=("student_count_2025", "sum"),
            cohort_forecast_students=("cohort_forecast_student_count", "sum"),
            avg_migration_adjustment=("migration_adjustment", "mean"),
            avg_cohort_pressure=("cohort_pressure_ratio", "mean"),
        )
    )
    total["change_pct"] = (total["cohort_forecast_students"] / total["student_2025"].replace(0, np.nan) - 1) * 100
    total.to_csv(REPORTS / "school_level_cohort_scenario_total_summary.csv", index=False, encoding="utf-8-sig")
    print(total[total["forecast_year"].isin([2029, 2035, 2040])].to_string(index=False))
    print("saved:", PROCESSED / "school_level_cohort_scenario_2026_2040.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
