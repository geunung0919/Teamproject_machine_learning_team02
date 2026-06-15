# R1 - Base School Features

- **Purpose**: Evaluates ML model accuracy using only base school administrative features.
- **Input**: Model views (`r1_basic_1yr.csv`, `r1_basic_3yr.csv`).
- **Output**: Model coefficients, predictions, and validation metrics.
- **Feature Group**: Base school student count history, class and teacher counts, administrative descriptors. No isolation, no grade flow, no cohort features.
- **Model Group**: LinearRegression, Ridge, RandomForest, HistGradientBoosting.
- **Relationship to Other Stages**: Provides the primary machine learning baseline. Subsequent R-stages add distinct feature groups to gauge incremental improvements.


