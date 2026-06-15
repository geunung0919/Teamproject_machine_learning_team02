# R2 - R1 + Isolation Features

- **Purpose**: Evaluates spatial isolation and school density signals on student delta forecasting.
- **Input**: Model views (`r2_isolation_1yr.csv`, `r2_isolation_3yr.csv`).
- **Output**: Evaluation metrics, predictions, and model fits.
- **Feature Group**: All R1 features + haversine distance features (nearest same-level school, count within 3km/5km/10km, isolation score).
- **Model Group**: LinearRegression, Ridge, RandomForest, HistGradientBoosting.
- **Relationship to Other Stages**: Builds directly on R1, measuring isolation ablation gains.


