# Source Code

This folder contains the final v5 analysis and scenario-generation source code.

## Main folders

- `api/`: public-data and API collection scripts.
- `common/`: shared paths, IO, metrics, modeling, reporting, and validation helpers.
- `features/`: feature construction logic.
- `code/r_stages/`: R-stage model/evaluation scripts.
- `code/scenario/`: scenario generation and web data export scripts.
- `pipeline/`: high-level v5 pipeline entry points.

## Main entry points

```bash
python src/pipeline/run_v5_full_pipeline.py --stage all --horizon 1
python src/pipeline/run_v5_model_only.py --horizon 1
python src/pipeline/run_v5_web_export_only.py
```

The public release does not include raw/intermediate data required for full retraining.


