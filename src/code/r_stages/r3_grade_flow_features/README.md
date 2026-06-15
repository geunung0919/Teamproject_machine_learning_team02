# R3 - R2 + Grade/Class Flow Features

- **Purpose**: Evaluates grade-level cohort distributions and entrants/graduates flow characteristics.
- **Input**: Model views (`r3_grade_flow_1yr.csv`, `r3_grade_flow_3yr.csv`).
- **Output**: Core validation metrics and predictions.
- **Feature Group**: All R2 features + grade student counts, grade shares, class counts, lower/upper school segments, entrants, graduates, and transfer flow features.
- **Model Group**: LinearRegression, Ridge, RandomForest, HistGradientBoosting.
- **Relationship to Other Stages**: Extends R2. R3 serves as the final selected stable dataset layout for web dashboard packaging due to cohort stability.


