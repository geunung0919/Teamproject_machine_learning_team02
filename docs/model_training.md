# Model Training Summary

This document summarizes the final v5 model training design. The public repository is a final-result package, so the raw and intermediate training datasets are not included.

## Final model family

- Main selected approach: R3-based multi-output regression.
- Core estimator family: `HistGradientBoostingRegressor`.
- Forecast horizon: 2026-2030 scenario period.
- Evaluation artifacts: `outputs/metrics/v5_final_r3_audit_and_web_filters/`.

## Actual public entry points

The package contains these pipeline entry points:

```bash
python src/pipeline/run_v5_full_pipeline.py --stage all --horizon 1
python src/pipeline/run_v5_model_only.py --horizon 1
python src/pipeline/run_v5_web_export_only.py
```

These scripts require internal raw/intermediate data that is not included in this public release.

## Final result viewing

For evaluation, open:

```text
web/index.html
```

No npm install or Vite rebuild is required.


