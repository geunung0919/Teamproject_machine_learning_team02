from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from src.common.paths import RAW_DIR
from src.common.io import read_csv, write_csv

def extract_event_layer(excluded_path: Path, output_dir: Path) -> pd.DataFrame:
    """Isolate event-excluded anomalous schools for separate visual layer mapping."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not excluded_path.exists():
        print(f"Excluded schools file not found at: {excluded_path}")
        return pd.DataFrame()
        
    excluded = read_csv(excluded_path)
    
    cols = [
        "school_key", "school_name", "sido", "sgg", "school_level", 
        "latest_student_count", "event_layer_category", "event_flags", 
        "exclusion_reason", "recommended_action"
    ]
    
    keep = [c for c in cols if c in excluded.columns]
    event_layer = excluded[keep].copy()
    
    write_csv(event_layer, output_dir / "event_excluded_school_layer.csv")
    return event_layer

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Event Excluded School Layer")
    parser.add_argument("--excluded-schools", type=str, default=str(RAW_DIR.parent / "v5_p1_excluded_school_list_v1" / "p1_excluded_schools_2173.csv"))
    parser.add_argument("--output-dir", type=str, default=str(RAW_DIR.parent / "v5_final_2026_2030_scenario_generation_v1"))
    args = parser.parse_args()
    
    extract_event_layer(Path(args.excluded_schools), Path(args.output_dir))
    print("Event excluded school layer isolated.")

if __name__ == "__main__":
    main()
