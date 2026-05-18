from __future__ import annotations

from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd


ROOT = SRC.parent
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "outputs" / "reports"


CORE_ID_COLUMNS = [
    "schlCd",
    "schlNm",
    "requested_sido_name",
    "requested_sido_code",
    "sgg_code",
    "sggCd",
    "school_level",
    "foundation",
    "schulRdnma",
    "openDate",
    "lttud",
    "lgtud",
]

CURRENT_SCHOOL_COLUMNS = [
    "student_count_2025",
    "class_count_2025",
    "teacher_count_2025",
    "students_per_class",
    "school_age",
]

REGIONAL_POPULATION_COLUMNS = [
    "school_age_pop_0_19",
    "pop_0_4",
    "pop_5_9",
    "pop_10_14",
    "pop_15_19",
    "school_age_pop_mom_rate",
    "forecast_school_age_pop_0_19",
    "population_pressure_ratio",
]

BIRTH_MIGRATION_COLUMNS = [
    "total_fertility_rate",
    "tfr_yoy_change",
    "tfr_yoy_rate",
    "birth_count",
    "birth_count_yoy_rate",
    "in_migration_total",
    "out_migration_total",
    "net_migration_total",
    "in_migration_yoy_rate",
    "out_migration_yoy_rate",
    "net_migration_rate_proxy",
]

COMMERCIAL_COLUMNS = [
    "commercial_count",
    "education_business_count",
    "kids_business_count",
    "medical_business_count",
    "radius_0_5km_all_shops",
    "radius_0_5km_education_shops",
    "radius_0_5km_kids_shops",
    "radius_0_5km_medical_shops",
    "radius_1_0km_all_shops",
    "radius_1_0km_education_shops",
    "radius_1_0km_kids_shops",
    "radius_1_0km_medical_shops",
    "radius_2_0km_all_shops",
    "radius_2_0km_education_shops",
    "radius_2_0km_kids_shops",
    "radius_2_0km_medical_shops",
    "commercial_per_1000_child",
    "education_per_1000_child",
    "kids_per_1000_child",
]

SPATIAL_RISK_COLUMNS = [
    "nearest_same_level_school_km",
    "same_level_school_count_5km",
    "school_isolation_score",
    "sgg_commercial_vulnerability_score",
    "commercial_vulnerability_source",
    "commercial_vulnerability_score",
    "regional_decline_risk_score",
]

OBJECTIVE_EDSS_COLUMNS = [
    "objective_closure_probability",
    "objective_closure_score",
    "objective_closure_percentile",
    "objective_model_threshold",
    "objective_threshold_hit",
    "objective_risk_grade",
    "objective_model_note",
]

SCENARIO_COLUMNS = [
    "forecast_year",
    "school_level_sensitivity",
    "forecast_student_count",
    "low_student_threshold",
    "pred_low_student_flag",
    "long_term_decline_flag",
    "severe_decline_flag",
    "replacement_near_flag",
    "objective_top10_flag",
    "risk_score",
    "differentiated_risk_score",
    "risk_category",
    "risk_category_ko",
    "special_school_review_flag",
    "student_data_issue_flag",
]


COLUMN_GROUPS = {
    "school_identity": CORE_ID_COLUMNS,
    "current_school_status": CURRENT_SCHOOL_COLUMNS,
    "regional_population_forecast": REGIONAL_POPULATION_COLUMNS,
    "birth_migration": BIRTH_MIGRATION_COLUMNS,
    "commercial_context": COMMERCIAL_COLUMNS,
    "spatial_policy_risk": SPATIAL_RISK_COLUMNS,
    "edss_auxiliary_score": OBJECTIVE_EDSS_COLUMNS,
    "future_scenario_target": SCENARIO_COLUMNS,
}


def select_existing_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in frame.columns]


def build_column_dictionary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    used = set()
    for group_name, columns in COLUMN_GROUPS.items():
        for col in columns:
            if col in frame.columns:
                rows.append({"column": col, "feature_group": group_name, "available": True})
                used.add(col)
            else:
                rows.append({"column": col, "feature_group": group_name, "available": False})
    for col in frame.columns:
        if col not in used:
            rows.append({"column": col, "feature_group": "other_or_pipeline_metadata", "available": True})
    return pd.DataFrame(rows)


def add_source_flags(frame: pd.DataFrame) -> pd.DataFrame:
    legacy_cols = [
        "final_supervised_closure_probability",
        "final_supervised_closure_percentile",
    ]
    df = frame.drop(columns=[col for col in legacy_cols if col in frame.columns]).copy()
    df["source_school_name"] = "eduinfo_current_schools.schlNm"
    df["source_current_school_status"] = "eduinfo_current_school_detail_national"
    df["source_edss_auxiliary"] = "edss_national_school_panel_2009_2023_proxy_model"
    df["source_population_birth_migration"] = "kosis_processed_sgg_features"
    df["source_commercial"] = "small_shop_and_school_radius_features"
    df["source_spatial_risk"] = "current_school_coordinates_derived_features"
    return df


def build_master_current() -> pd.DataFrame:
    current = pd.read_csv(PROCESSED / "final_national_current_school_features.csv", low_memory=False)
    return add_source_flags(current)


def build_master_scenario() -> pd.DataFrame:
    scenario = pd.read_csv(PROCESSED / "final_national_school_scenario_2026_2040.csv", low_memory=False)
    return add_source_flags(scenario)


def main() -> int:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    current = build_master_current()
    scenario = build_master_scenario()

    current.to_csv(PROCESSED / "modeling_master_current_school_features.csv", index=False, encoding="utf-8-sig")
    scenario.to_csv(PROCESSED / "modeling_master_school_scenario_2026_2040.csv", index=False, encoding="utf-8-sig")

    dictionary = build_column_dictionary(scenario)
    dictionary.to_csv(REPORTS / "modeling_master_column_dictionary.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(
        [
            {
                "dataset": "modeling_master_current_school_features.csv",
                "rows": len(current),
                "columns": len(current.columns),
                "description": "One row per current school. Use for current-feature checks and feature engineering.",
            },
            {
                "dataset": "modeling_master_school_scenario_2026_2040.csv",
                "rows": len(scenario),
                "columns": len(scenario.columns),
                "description": "One row per school-year. Use for final scenario modeling and visualization.",
            },
            {
                "dataset": "modeling_master_column_dictionary.csv",
                "rows": len(dictionary),
                "columns": len(dictionary.columns),
                "description": "Feature-group dictionary for the modeling master dataset.",
            },
        ]
    )
    summary.to_csv(REPORTS / "modeling_master_dataset_summary.csv", index=False, encoding="utf-8-sig")

    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
