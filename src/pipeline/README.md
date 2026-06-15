# V5 Production Pipeline Orchestration

This package provides executable orchestration entrypoints for running model steps, model training, and exporting web assets.

## Orchestrators

- `run_v5_full_pipeline.py`: Coordinates dataset patch, R0-R6 models, model selection, scenario generation, and web exports.
- `run_v5_model_only.py`: Runs baseline and R1-R6 modeling experiments.
- `run_v5_web_export_only.py`: Generates the scenario projections and web packages.


