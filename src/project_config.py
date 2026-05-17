from __future__ import annotations

import pandas as pd


YEARS = list(range(2026, 2041))

RISK_LABEL_KO = {
    "low_risk": "저위험",
    "mid_risk": "중위험",
    "high_risk_review": "고위험 검토",
    "consolidation_high_risk": "통폐합 가능 고위험",
    "education_gap_high_risk": "교육공백 우려 고위험",
    "special_school_review": "특수/각종학교 별도검토",
    "data_check_needed": "학생수 데이터 확인필요",
}

RISK_COLORS = {
    "low_risk": "#2ca25f",
    "mid_risk": "#fdae6b",
    "high_risk_review": "#fb6a4a",
    "consolidation_high_risk": "#de2d26",
    "education_gap_high_risk": "#756bb1",
    "special_school_review": "#3182bd",
    "data_check_needed": "#737373",
}

RISK_ORDER = [
    "consolidation_high_risk",
    "education_gap_high_risk",
    "high_risk_review",
    "mid_risk",
    "low_risk",
    "special_school_review",
    "data_check_needed",
]

SIDO_COORD_BOUNDS = {
    "서울": (37.40, 37.75, 126.75, 127.25),
    "부산": (35.00, 35.40, 128.75, 129.35),
    "대구": (35.75, 36.05, 128.35, 128.80),
    "인천": (37.00, 38.00, 124.50, 127.05),
    "광주": (35.00, 35.30, 126.65, 127.05),
    "대전": (36.15, 36.55, 127.20, 127.60),
    "울산": (35.30, 35.75, 129.00, 129.50),
    "세종": (36.35, 36.75, 127.10, 127.45),
    "경기": (36.85, 38.35, 126.45, 127.95),
    "강원": (37.00, 38.65, 127.00, 129.40),
    "충북": (36.00, 37.35, 127.30, 128.75),
    "충남": (35.95, 37.10, 125.90, 127.65),
    "전북": (35.25, 36.20, 126.30, 127.90),
    "전남": (33.85, 35.55, 125.00, 127.85),
    "경북": (35.55, 37.65, 128.00, 131.00),
    "경남": (34.55, 35.95, 127.55, 129.40),
    "제주": (33.05, 34.10, 126.00, 126.95),
}

GENERAL_SCHOOL_LEVELS = {"초등학교", "중학교", "고등학교"}

RISK_SCORE_WEIGHTS = {
    "low_student": 28,
    "long_decline": 12,
    "severe_decline": 12,
    "isolation": 13,
    "commercial": 10,
    "regional": 10,
    "edss": 15,
    "replacement": 8,
}


def normalize_school_level(value: object) -> str:
    text = str(value)
    if "초등" in text:
        return "초등학교"
    if "중학교" in text or text == "중학":
        return "중학교"
    if "고등" in text:
        return "고등학교"
    if "특수" in text or "맹아" in text or "농아" in text:
        return "특수학교"
    if "각종" in text:
        return "각종학교"
    return "기타학교"


def valid_sido_coord_mask(df: pd.DataFrame) -> pd.Series:
    lat = pd.to_numeric(df["lttud"], errors="coerce")
    lon = pd.to_numeric(df["lgtud"], errors="coerce")
    valid = lat.notna() & lon.notna()
    in_any_korea_box = lat.between(33.0, 38.7) & lon.between(124.0, 131.5)

    sido_valid = pd.Series(False, index=df.index)
    for sido, (lat_min, lat_max, lon_min, lon_max) in SIDO_COORD_BOUNDS.items():
        mask = df["requested_sido_name"].eq(sido)
        sido_valid.loc[mask] = lat.loc[mask].between(lat_min, lat_max) & lon.loc[mask].between(lon_min, lon_max)

    return valid & in_any_korea_box & sido_valid


def coordinate_issue_reason(row: pd.Series) -> str:
    lat = pd.to_numeric(pd.Series([row.get("lttud")]), errors="coerce").iloc[0]
    lon = pd.to_numeric(pd.Series([row.get("lgtud")]), errors="coerce").iloc[0]
    if pd.isna(lat) or pd.isna(lon):
        return "좌표 누락"
    if not (33.0 <= lat <= 38.7 and 124.0 <= lon <= 131.5):
        return "한국 좌표 범위 밖"
    bounds = SIDO_COORD_BOUNDS.get(str(row.get("requested_sido_name")))
    if bounds is None:
        return "시도명 미매칭"
    lat_min, lat_max, lon_min, lon_max = bounds
    if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
        return "시도 bounds 불일치"
    return "정상"


def compute_policy_risk_score(df: pd.DataFrame) -> pd.Series:
    weights = RISK_SCORE_WEIGHTS
    return (
        df["pred_low_student_flag"] * weights["low_student"]
        + df["long_term_decline_flag"] * weights["long_decline"]
        + df["severe_decline_flag"] * weights["severe_decline"]
        + df["isolation_high_flag"] * weights["isolation"]
        + df["commercial_vulnerable_flag"] * weights["commercial"]
        + df["regional_decline_high_flag"] * weights["regional"]
        + df["objective_top10_flag"] * weights["edss"]
        + df["replacement_near_flag"] * weights["replacement"]
    )


def assign_policy_risk_label(row: pd.Series) -> str:
    if pd.isna(row["student_count_2025"]) or row["student_count_2025"] <= 0:
        return "data_check_needed"
    if row["school_level"] not in GENERAL_SCHOOL_LEVELS:
        return "special_school_review"
    if row["risk_score"] < 35:
        return "low_risk"
    if row["pred_low_student_flag"] and row["isolation_high_flag"] and not row["replacement_near_flag"]:
        return "education_gap_high_risk"
    if row["pred_low_student_flag"] and row["replacement_near_flag"]:
        return "consolidation_high_risk"
    if row["risk_score"] >= 60:
        return "high_risk_review"
    return "mid_risk"
