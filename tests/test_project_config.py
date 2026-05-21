from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from project_config import assign_policy_risk_label, compute_policy_risk_score


def make_policy_frame(**overrides: object) -> pd.DataFrame:
    base = {
        "student_count_2025": 80,
        "school_level": "초등학교",
        "pred_low_student_flag": 1,
        "long_term_decline_flag": 0,
        "severe_decline_flag": 0,
        "isolation_high_flag": 0,
        "commercial_vulnerable_flag": 0,
        "regional_decline_high_flag": 0,
        "objective_top10_flag": 0,
        "replacement_near_flag": 0,
    }
    base.update(overrides)
    return pd.DataFrame([base])


def with_score(df: pd.DataFrame) -> pd.Series:
    scored = df.copy()
    scored["risk_score"] = compute_policy_risk_score(scored)
    return scored.iloc[0]


def test_zero_student_count_requires_data_check() -> None:
    row = with_score(make_policy_frame(student_count_2025=0, risk_score=100))
    assert assign_policy_risk_label(row) == "data_check_needed"


def test_isolated_low_student_school_becomes_education_gap_high_risk() -> None:
    row = with_score(make_policy_frame(isolation_high_flag=1, replacement_near_flag=0))
    assert row["risk_score"] >= 35
    assert assign_policy_risk_label(row) == "education_gap_high_risk"


def test_near_replacement_school_becomes_consolidation_high_risk() -> None:
    row = with_score(make_policy_frame(isolation_high_flag=0, replacement_near_flag=1))
    assert row["risk_score"] >= 35
    assert assign_policy_risk_label(row) == "consolidation_high_risk"
