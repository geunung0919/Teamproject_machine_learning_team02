# System Architecture

The final v5 project has four broad layers:

1. Data collection and cleaning from public data sources.
2. R-stage feature engineering and model evaluation.
3. 2026-2030 scenario generation.
4. Static web dashboard export.

## High-level flow

```text
public data sources
  -> raw/internal data preparation
  -> R-stage feature tables
  -> R3-based multi-output model evaluation
  -> scenario generation
  -> public/data static JSON/CSV
  -> web/index.html
```

## Main source folders

- `src/api/`: API collection scripts.
- `src/common/`: shared utilities and project paths.
- `src/features/`: feature engineering.
- `src/code/r_stages/`: stage-specific modeling/evaluation scripts.
- `src/code/scenario/`: scenario and web data export scripts.
- `src/pipeline/`: top-level pipeline wrappers.

## Public release boundary

This GitHub package includes final source code and final artifacts, but excludes raw data, intermediate model views, and heavy experiment outputs.


