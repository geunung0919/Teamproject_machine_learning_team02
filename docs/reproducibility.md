# Reproducibility Scope

This v5 GitHub package is a final-result release for evaluation. It is not a full raw-data retraining repository.

## What is reproducible from this repository

- Inspect final source code under `src/`.
- Inspect final model/evaluation documents under `docs/`.
- Inspect final metrics under `outputs/metrics/`.
- Inspect final figures under `outputs/figures/`.
- Open the final static web result at `web/index.html`.

## Web result

No npm or Vite build is required.

```text
web/index.html
```

Open the file in a browser to view the final dashboard.

## Python environment

```bash
pip install -r requirements.txt
```

Known public entry points:

```bash
python src/pipeline/run_v5_full_pipeline.py --stage all --horizon 1
python src/pipeline/run_v5_model_only.py --horizon 1
python src/pipeline/run_v5_web_export_only.py
```

These scripts require internal raw/intermediate data that is not included in this public release.

## Data limitation

Raw data and intermediate training data are excluded from GitHub because of size and public redistribution constraints.


