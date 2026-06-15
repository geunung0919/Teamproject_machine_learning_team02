from __future__ import annotations

# R1 core features (basic school student count, administrative, demographic KOSIS proxies)
R1_BASE_COLUMNS = [
    "student_count", "student_count_lag_1", "student_count_lag_2", "student_count_lag_3",
    "student_delta_lag_1", "student_growth_lag_1", "student_rolling_mean_3", 
    "student_rolling_delta_mean_3", "student_trend_slope_3", "metro_flag", 
    "class_count", "teacher_count", "students_per_class", "students_per_teacher",
    "land_area", "school_age_population_0_19", "age_0_4_pop", "age_5_9_pop", 
    "age_10_14_pop", "age_15_19_pop", "school_age_population_delta_1y", 
    "school_age_population_growth_1y", "birth_count", "total_fertility_rate", 
    "birth_count_yoy_change", "birth_count_yoy_rate", "tfr_yoy_change", "tfr_yoy_rate",
    "net_migration_total", "in_migration_total", "out_migration_total", 
    "net_migration_yoy_change", "sido", "sgg", "school_level", "branch_type",
    "foundation_type", "region_group", "size_bucket", "student_size_bin"
]

# R2 isolation specific features
R2_ISO_COLUMNS = [
    "nearest_same_level_distance_km", "second_nearest_same_level_distance_km",
    "same_level_school_count_within_3km", "same_level_school_count_within_5km",
    "same_level_school_count_within_10km", "no_same_level_school_within_5km_flag", 
    "isolation_score"
]

# R3 grade/class flow features
R3_GRADE_FLOW_COLUMNS = [
    "grade1_student_count", "grade2_student_count", "grade3_student_count",
    "grade4_student_count", "grade5_student_count", "grade6_student_count",
    "grade_student_sum", "grade1_share", "grade2_share", "grade3_share", 
    "grade4_share", "grade5_share", "grade6_share", "grade1_class_count", 
    "grade2_class_count", "grade3_class_count", "grade4_class_count", 
    "grade5_class_count", "grade6_class_count", "grade_class_sum", 
    "entrants_total", "graduates_total", "transfer_in", "transfer_out", 
    "lower_grade_student_count", "upper_grade_student_count", 
    "graduating_grade_student_count", "grade_imbalance_range", "grade_imbalance_std"
]

# R4 segment indices
R4_SEGMENT_COLUMNS = [
    "level_size_segment", "school_level_x_size_bucket", 
    "region_group_x_school_level", "sido_x_school_level"
]

# Target variables (excluded from features)
TARGET_COLUMNS = [
    "target_year_1yr", "target_student_count_1yr", "target_delta_1yr", "target_available_1yr",
    "target_year_2yr", "target_student_count_2yr", "target_delta_2yr", "target_available_2yr",
    "target_year_3yr", "target_student_count_3yr", "target_delta_3yr", "target_available_3yr",
    "target_year_4yr", "target_student_count_4yr", "target_delta_4yr", "target_available_4yr",
    "target_year_5yr", "target_student_count_5yr", "target_delta_5yr", "target_available_5yr",
]

# Standard metadata exclusions
EXCLUDE_BASE = {
    "school_key", "school_name", "school_name_norm", "address", "source_file", "survey_date",
    "exclusion_reason", "quality_note", "patch_exclusion_reason", "status",
    "r1_model_eligible_patched", "r2_model_eligible_patched", "r3_model_eligible_patched",
    "standard_model_eligible", "scenario_base_eligible", "exclude_p1_row_level", 
    "exclude_p1_school_level", "event_flags", "coordinate_source", 
    "coordinate_invalid_reason", "isolation_score_version", "grade_invalid_reason",
    "school_status_2025", "coordinate_quality_flag", "candidate_name", 
    "feature_family", "forecasting_strategy", "target_type", "model",
    "base_year", "target_year", "horizon", "base_student_count", 
    "actual_student_count", "predicted_student_count", "abs_error"
}
