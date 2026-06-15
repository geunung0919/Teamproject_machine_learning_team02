# R6 - Actual Cohort

- **Purpose**: Evaluates actual cohort-pressure demographics from municipal birth and population records.
- **Input**: Model views combined with KOSIS birth, fertility, and age-specific demographics growth and pressure ratios.
- **Output**: Metrics comparison long formats, coefficients, and predictions.
- **Feature Group**: All R3 features + elementary/middle/high cohort pressure ratios and demographics growth rates.
- **Model Group**: LinearRegression, Ridge, RandomForest, HistGradientBoosting.
- **Relationship to Other Stages**: Final modeling stage. Directly compared to R3 in model selection to determine scenario path stability.


