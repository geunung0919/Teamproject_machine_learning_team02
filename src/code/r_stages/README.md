# V5 R0 to R6 Training and Evaluation Stages

This directory contains individual wrappers for running modeling experiments across R-stages, plus the selection script that evaluates recursive vs multi-output models.

## Stages Mapping

- `r0_baseline`: Persistence baseline forecasting.
- `r1_school_features`: School-level baseline features (no isolation, no grade flow, no cohort).
- `r2_isolation_features`: School-level features + spatial haversine isolation indices.
- `r3_grade_flow_features`: Isolation features + grade-level and class-level flow counts.
- `r4_segment_models`: Segment-aware models (metro/province, school size splits).
- `r5_cohort_proxy`: R3 features + demographic aggregate trend proxies.
- `r6_actual_cohort`: R3 features + actual cohort demographic pressure proxy features.
- `final_r3_r6_selection`: Evaluates validation metrics and path stability to select the best forecasting candidate.


