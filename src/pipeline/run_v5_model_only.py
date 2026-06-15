from __future__ import annotations

import argparse
from src.pipeline.run_v5_full_pipeline import run_pipeline

def main() -> None:
    parser = argparse.ArgumentParser(description="V5 Models-Only Pipeline Execution")
    parser.add_argument("--horizon", type=int, default=1, help="Horizon projection length (1 to 5 yr)")
    args = parser.parse_args()
    
    # Run modeling experiments sequentially
    for stg in ["r0", "r1", "r2", "r3", "r4", "r5", "r6"]:
        run_pipeline(stg, args.horizon)
    print("V5 Models-only pipeline stages complete.")

if __name__ == "__main__":
    main()
