from __future__ import annotations

from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from project_config import assign_policy_risk_label, compute_policy_risk_score


ROOT = SRC.parent
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "outputs" / "reports"


def relabel_with_thresholds(frame: pd.DataFrame, isolation_threshold: float, context_threshold: float) -> pd.DataFrame:
    df = frame.copy()
    df["isolation_high_flag"] = (df["school_isolation_score"] >= isolation_threshold).astype(int)
    df["commercial_vulnerable_flag"] = (df["commercial_vulnerability_score"] >= context_threshold).astype(int)
    df["regional_decline_high_flag"] = (df["regional_decline_risk_score"] >= context_threshold).astype(int)
    df["risk_score"] = compute_policy_risk_score(df)
    df["risk_label"] = df.apply(assign_policy_risk_label, axis=1)
    return df


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    scenario = pd.read_csv(PROCESSED / "final_national_school_scenario_2026_2040.csv", low_memory=False)
    base_years = [2029, 2035, 2040]
    rows = []
    for year in base_years:
        base = scenario[scenario["forecast_year"].eq(year)].copy()
        for isolation_threshold in [60, 70, 80]:
            for context_threshold in [60, 70, 80]:
                relabeled = relabel_with_thresholds(base, isolation_threshold, context_threshold)
                counts = relabeled["risk_label"].value_counts().to_dict()
                rows.append(
                    {
                        "forecast_year": year,
                        "isolation_threshold": isolation_threshold,
                        "commercial_regional_threshold": context_threshold,
                        "total_schools": len(relabeled),
                        "consolidation_high_risk": counts.get("consolidation_high_risk", 0),
                        "education_gap_high_risk": counts.get("education_gap_high_risk", 0),
                        "high_risk_review": counts.get("high_risk_review", 0),
                        "mid_risk": counts.get("mid_risk", 0),
                        "low_risk": counts.get("low_risk", 0),
                        "special_school_review": counts.get("special_school_review", 0),
                        "data_check_needed": counts.get("data_check_needed", 0),
                    }
                )
    out = pd.DataFrame(rows)
    out["high_risk_total"] = (
        out["consolidation_high_risk"] + out["education_gap_high_risk"] + out["high_risk_review"]
    )
    out.to_csv(REPORTS / "risk_threshold_sensitivity.csv", index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))
    print("saved:", REPORTS / "risk_threshold_sensitivity.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

