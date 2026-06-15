# R4 - Segment Models

- **Purpose**: Assesses forecasting performance when splitting training and evaluation by region and school size segments.
- **Input**: Model views (`r4_region_group_1yr.csv`, `r4_region_group_3yr.csv`, `r4_size_bucket_1yr.csv`, `r4_size_bucket_3yr.csv`).
- **Output**: Segmented validation metrics (MAE, RMSE, WAPE).
- **Feature Group**: Segment indices (metro vs province, capital area, and size groupings).
- **Model Group**: Segment-mean baseline delta models.
- **Relationship to Other Stages**: Provides regional and scale accuracy comparisons against pooled models.


