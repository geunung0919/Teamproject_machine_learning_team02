from __future__ import annotations

from src.pipeline.run_v5_full_pipeline import run_pipeline

def main() -> None:
    # Run only selection, scenario generation and web packaging
    for stg in ["selection", "scenario", "web"]:
        run_pipeline(stg, 1)
    print("V5 Web export only pipeline stages complete.")

if __name__ == "__main__":
    main()
