from __future__ import annotations

import argparse
from pathlib import Path

# Paths & Helpers
from src.common.paths import (
    CLEAN_PATCH_DIR, POLICY_COMP_DIR, RECURSIVE_FORECAST_DIR, 
    WEB_PACKAGE_DIR, PUBLIC_ASSETS_DIR, RAW_DIR
)

# Feature policies & stages
from src.code.r_stages.r0_baseline.run_stage import run_r0
from src.code.r_stages.r1_school_features.run_stage import run_r1
from src.code.r_stages.r2_isolation_features.run_stage import run_r2
from src.code.r_stages.r3_grade_flow_features.run_stage import run_r3
from src.code.r_stages.r4_segment_models.run_stage import run_r4
from src.code.r_stages.r5_cohort_proxy.run_stage import run_r5
from src.code.r_stages.r6_actual_cohort.run_stage import run_r6
from src.code.r_stages.final_r3_r6_selection.run_selection import run_selection
from src.code.scenario.generate_final_scenario import generate_scenario
from src.code.scenario.export_web_package import run_export

def run_pipeline(stage_arg: str, horizon: int) -> None:
    """Run specified pipeline phases or 'all' sequentially."""
    stage = stage_arg.lower()
    
    print(f"Starting V5 Reorganized Pipeline: stage={stage_arg} horizon={horizon}yr")
    
    # Standard input & output directories
    patch_view_r0 = CLEAN_PATCH_DIR / "model_views" / f"r0_baseline_{horizon}yr.csv"
    if not patch_view_r0.exists():
        patch_view_r0 = CLEAN_PATCH_DIR / "model_views" / "r0_baseline_1yr.csv"
        
    patch_view_r1 = CLEAN_PATCH_DIR / "model_views" / f"r1_basic_{horizon}yr.csv"
    if not patch_view_r1.exists():
        patch_view_r1 = CLEAN_PATCH_DIR / "model_views" / "r1_basic_1yr.csv"
        
    patch_view_r2 = CLEAN_PATCH_DIR / "model_views" / f"r2_isolation_{horizon}yr.csv"
    if not patch_view_r2.exists():
        patch_view_r2 = CLEAN_PATCH_DIR / "model_views" / "r2_isolation_1yr.csv"
        
    patch_view_r3 = CLEAN_PATCH_DIR / "model_views" / f"r3_grade_flow_{horizon}yr.csv"
    if not patch_view_r3.exists():
        patch_view_r3 = CLEAN_PATCH_DIR / "model_views" / "r3_grade_flow_1yr.csv"
        
    patch_view_r4 = CLEAN_PATCH_DIR / "model_views" / f"r4_region_group_{horizon}yr.csv"
    if not patch_view_r4.exists():
        patch_view_r4 = CLEAN_PATCH_DIR / "model_views" / "r4_region_group_1yr.csv"
        
    policy_results = POLICY_COMP_DIR / "results"
    
    # 1. R0 Baseline
    if stage in ["all", "r0"]:
        print("\n--- Running Stage R0 Baseline ---")
        run_r0(patch_view_r0, policy_results, horizon)
        
    # 2. R1 School Features
    if stage in ["all", "r1"]:
        print("\n--- Running Stage R1 Base School Features ---")
        run_r1(patch_view_r1, policy_results, horizon)
        
    # 3. R2 Isolation Features
    if stage in ["all", "r2"]:
        print("\n--- Running Stage R2 Isolation Features ---")
        run_r2(patch_view_r2, policy_results, horizon)
        
    # 4. R3 Grade Flow Features
    if stage in ["all", "r3"]:
        print("\n--- Running Stage R3 Grade Flow Features ---")
        run_r3(patch_view_r3, policy_results, horizon)
        
    # 5. R4 Segment Models
    if stage in ["all", "r4"]:
        print("\n--- Running Stage R4 Segment Models ---")
        run_r4(patch_view_r4, policy_results, "region_group", horizon)
        
    # 6. R5 Cohort Proxy
    if stage in ["all", "r5"]:
        print("\n--- Running Stage R5 Cohort Proxy Features ---")
        run_r5(patch_view_r3, policy_results, horizon)
        
    # 7. R6 Actual Cohort
    if stage in ["all", "r6"]:
        print("\n--- Running Stage R6 Actual Cohort Demographics ---")
        run_r6(patch_view_r3, policy_results, horizon)
        
    # 8. Model Selection check (R3 vs R6)
    if stage in ["all", "selection"]:
        print("\n--- Running Stage R3/R6 Model Selection check ---")
        run_selection(RECURSIVE_FORECAST_DIR, policy_results)
        
    # 9. Scenario Generation
    if stage in ["all", "scenario"]:
        print("\n--- Running Scenario Generation ---")
        excluded_path = RAW_DIR.parent / "v5_p1_excluded_school_list_v1" / "p1_excluded_schools_2173.csv"
        scen_out = POLICY_COMP_DIR.parent / "v5_final_2026_2030_scenario_generation_v1"
        generate_scenario(CLEAN_PATCH_DIR, POLICY_COMP_DIR, excluded_path, scen_out)
        
    # 10. Web Package Export
    if stage in ["all", "web"]:
        print("\n--- Running Web Package Export ---")
        run_export(RECURSIVE_FORECAST_DIR, CLEAN_PATCH_DIR, WEB_PACKAGE_DIR, PUBLIC_ASSETS_DIR)
        
    print("\nPipeline execution complete.")

def main() -> None:
    parser = argparse.ArgumentParser(description="V5 Full Pipeline Orchestration")
    parser.add_argument("--stage", type=str, default="all", help="Stage to execute: all, r0, r1, r2, r3, r4, r5, r6, selection, scenario, web")
    parser.add_argument("--horizon", type=int, default=1, help="Horizon projection length (1 to 5 yr)")
    args = parser.parse_args()
    
    run_pipeline(args.stage, args.horizon)

if __name__ == "__main__":
    main()
