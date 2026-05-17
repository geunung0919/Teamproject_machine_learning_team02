from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from project_config import assign_policy_risk_label, compute_policy_risk_score


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "outputs" / "reports"

LOW_STUDENT_THRESHOLDS = {"초등학교": 120, "중학교": 150, "고등학교": 180, "특수학교": 40, "각종학교": 60}


def main() -> int:
    scenario_path = PROCESSED / "final_national_school_scenario_2026_2040.csv"
    cohort_path = PROCESSED / "school_level_cohort_scenario_2026_2040.csv"
    if not scenario_path.exists() or not cohort_path.exists():
        raise FileNotFoundError("Scenario and cohort files must exist before applying cohort risk.")

    scenario = pd.read_csv(scenario_path, low_memory=False)
    cohort = pd.read_csv(cohort_path, low_memory=False)
    cohort = cohort[cohort["cohort_scenario"].eq("baseline")].copy()
    cohort_cols = [
        "schlCd",
        "forecast_year",
        "cohort_forecast_student_count",
        "cohort_pressure_ratio",
        "baseline_cohort_births",
        "forecast_cohort_births",
        "migration_adjustment",
    ]
    cohort = cohort[cohort_cols].drop_duplicates(["schlCd", "forecast_year"])

    preserve_map = {
        "forecast_student_count": "pressure_model_forecast_student_count",
        "population_pressure_ratio": "pressure_model_population_pressure_ratio",
        "risk_score": "pressure_model_risk_score",
        "risk_label": "pressure_model_risk_label",
    }
    for original, preserved in preserve_map.items():
        if original in scenario.columns and preserved not in scenario.columns:
            scenario[preserved] = scenario[original]

    scenario = scenario.drop(
        columns=[c for c in cohort_cols if c not in {"schlCd", "forecast_year"} and c in scenario.columns]
    )
    scenario = scenario.merge(cohort, on=["schlCd", "forecast_year"], how="left")

    scenario["forecast_student_count"] = scenario["cohort_forecast_student_count"].fillna(
        scenario["forecast_student_count"]
    )
    scenario["population_pressure_ratio"] = scenario["cohort_pressure_ratio"].fillna(
        scenario["population_pressure_ratio"]
    )
    scenario["forecast_student_count"] = pd.to_numeric(scenario["forecast_student_count"], errors="coerce").fillna(0)
    scenario["population_pressure_ratio"] = pd.to_numeric(
        scenario["population_pressure_ratio"], errors="coerce"
    ).fillna(1.0)

    scenario["low_student_threshold"] = scenario["school_level"].map(LOW_STUDENT_THRESHOLDS).fillna(100)
    scenario["pred_low_student_flag"] = (
        scenario["forecast_student_count"] <= scenario["low_student_threshold"]
    ).astype(int)
    scenario["long_term_decline_flag"] = (scenario["population_pressure_ratio"] <= 0.85).astype(int)
    scenario["severe_decline_flag"] = (scenario["population_pressure_ratio"] <= 0.70).astype(int)
    scenario["risk_score"] = compute_policy_risk_score(scenario)
    scenario["differentiation_score"] = (
        scenario["school_isolation_score"] * 0.25
        + scenario["commercial_vulnerability_score"] * 0.20
        + scenario["regional_decline_risk_score"] * 0.25
        + scenario["objective_closure_percentile"] * 0.15
        + (1 - scenario["population_pressure_ratio"]).clip(0, 1) * 100 * 0.15
    ).round(1)
    scenario["risk_label"] = scenario.apply(assign_policy_risk_label, axis=1)
    scenario["forecast_model_basis"] = np.where(
        scenario["cohort_forecast_student_count"].notna(), "birth_cohort_baseline", "sgg_pressure_fallback"
    )

    scenario.to_csv(scenario_path, index=False, encoding="utf-8-sig")

    changed = scenario["pressure_model_risk_label"].ne(scenario["risk_label"]).sum()
    summary = (
        scenario.groupby(["forecast_year", "risk_label"], as_index=False)
        .agg(schools=("schlCd", "count"), avg_forecast_student_count=("forecast_student_count", "mean"))
    )
    summary.to_csv(REPORTS / "cohort_applied_risk_summary_2026_2040.csv", index=False, encoding="utf-8-sig")
    print("saved:", scenario_path)
    print("saved:", REPORTS / "cohort_applied_risk_summary_2026_2040.csv")
    print(f"risk label changed rows: {changed:,} / {len(scenario):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
