from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


FINAL_STEPS = [
    ("Train EDSS proxy model", "src/model_edss_closure_risk_national.py"),
    ("Build school-radius commercial features", "src/build_school_radius_commercial_features.py"),
    ("Build final national regression and risk scenario", "src/final_national_training_pipeline.py"),
    ("Apply supervised auxiliary percentile", "src/final_supervised_training.py"),
    ("Analyze risk threshold sensitivity", "src/analyze_risk_threshold_sensitivity.py"),
    ("Build school-level cohort scenario", "src/build_school_level_cohort_scenario.py"),
    ("Apply cohort baseline to final risk scenario", "src/apply_cohort_scenario_to_risk.py"),
    ("Generate model vs cohort comparison chart", "src/generate_comparison_chart.py"),
    ("Generate base vs tuned model comparison visuals", "src/generate_model_comparison_visuals.py"),
    ("Extract feature importance", "src/extract_model_feature_importance.py"),
    ("Analyze fertility pathway", "src/fertility_pathway_analysis.py"),
    ("Build final interactive HTML map", "src/build_final_interactive_school_risk_map.py"),
]

COLLECT_STEPS = [
    ("Collect KOSIS population data", "src/collect_national_kosis_population.py"),
    ("Collect KOSIS birth and migration data", "src/collect_national_kosis_birth_migration.py"),
    ("Collect KOSIS fertility data", "src/collect_national_kosis_fertility.py"),
    ("Collect and summarize national commercial data", "src/collect_national_small_shop.py"),
    ("Collect EduInfo school and closure data", "src/collect_eduinfo_data.py"),
]


def run_step(title: str, script: str) -> None:
    path = ROOT / script
    if not path.exists():
        raise FileNotFoundError(f"Required script is missing: {script}")
    print(f"\n=== {title}: {script} ===", flush=True)
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
    subprocess.run([sys.executable, str(path)], cwd=ROOT, check=True, env=env)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the final school-age population and school risk pipeline.")
    parser.add_argument(
        "--with-collect",
        action="store_true",
        help="Re-collect API source data before running the final modeling pipeline.",
    )
    args = parser.parse_args()

    steps = [*COLLECT_STEPS, *FINAL_STEPS] if args.with_collect else FINAL_STEPS
    for title, script in steps:
        run_step(title, script)

    print("\nDone: outputs/maps/final_national_interactive_school_risk_scenario.html", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
