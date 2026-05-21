from __future__ import annotations

from pathlib import Path
import re
import sys

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


ROOT = SRC.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "outputs" / "reports"

EARTH_RADIUS_KM = 6371.0088
GENERAL_LEVELS = {"초등학교", "중학교", "고등학교"}


def normalize_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.replace("（", "(").replace("）", ")")
    text = re.sub(r"\s+", "", text)
    return text.strip()


def normalize_sido(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    mapping = {
        "서울특별시": "서울",
        "부산광역시": "부산",
        "대구광역시": "대구",
        "인천광역시": "인천",
        "광주광역시": "광주",
        "대전광역시": "대전",
        "울산광역시": "울산",
        "세종특별자치시": "세종",
        "경기도": "경기",
        "강원특별자치도": "강원",
        "강원도": "강원",
        "충청북도": "충북",
        "충청남도": "충남",
        "전북특별자치도": "전북",
        "전라북도": "전북",
        "전라남도": "전남",
        "경상북도": "경북",
        "경상남도": "경남",
        "제주특별자치도": "제주",
    }
    return mapping.get(text, text)


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")


def load_long_schooldata() -> pd.DataFrame:
    path = PROCESSED / "school_data_2008_2025_clean.csv"
    usecols = [
        "데이터_연도",
        "조사기준일",
        "시도",
        "행정구",
        "학교급",
        "학교명",
        "본분교",
        "설립",
        "상태",
        "주소",
        "학급수_계",
        "학생수_총계_계",
        "교원수_총계_계",
        "교원1인당 학생수",
        "교지면적",
        "학생_1인당_교지면적",
    ]
    df = pd.read_csv(path, usecols=lambda c: c in usecols, low_memory=False)
    df = df[df["학교급"].isin(GENERAL_LEVELS)].copy()
    df["year"] = pd.to_numeric(df["데이터_연도"], errors="coerce").astype("Int64")
    df["sido_name"] = df["시도"].map(normalize_sido)
    df["school_name_norm"] = df["학교명"].map(normalize_name)
    df["school_key"] = df["sido_name"] + "|" + df["school_name_norm"]
    df["student_count"] = clean_numeric(df["학생수_총계_계"])
    df["class_count"] = clean_numeric(df["학급수_계"])
    df["teacher_count"] = clean_numeric(df["교원수_총계_계"])
    df["students_per_teacher"] = clean_numeric(df["교원1인당 학생수"])
    df["land_area"] = clean_numeric(df["교지면적"])
    df["land_area_per_student"] = clean_numeric(df["학생_1인당_교지면적"])
    df["students_per_class"] = df["student_count"] / df["class_count"].replace(0, np.nan)
    df["is_closed_status"] = df["상태"].astype(str).str.contains("폐", na=False).astype(int)
    df["is_suspended_status"] = df["상태"].astype(str).str.contains("휴", na=False).astype(int)
    df["is_active_status"] = df["상태"].astype(str).str.contains("기존|신설", regex=True, na=False).astype(int)
    return df


def load_current_coordinate_reference() -> pd.DataFrame:
    raw_path = RAW / "eduinfo_current_schools_national.csv"
    archive_path = ROOT / "archive" / "schooldata" / "raw" / "eduinfo_current_schools_national.csv"
    source_path = raw_path if raw_path.exists() else archive_path
    current = pd.read_csv(source_path, low_memory=False)
    current["sido_name"] = current["requested_sido_name"].map(normalize_sido)
    current["school_name_norm"] = current["schlNm"].map(normalize_name)
    current["school_key"] = current["sido_name"] + "|" + current["school_name_norm"]
    current["lttud"] = pd.to_numeric(current["lttud"], errors="coerce")
    current["lgtud"] = pd.to_numeric(current["lgtud"], errors="coerce")
    keep = [
        "school_key",
        "schlCd",
        "schlNm",
        "schlKndCd",
        "sggCd",
        "schulRdnma",
        "openDate",
        "lttud",
        "lgtud",
    ]
    return current.drop_duplicates("school_key")[keep]


def add_temporal_targets(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.sort_values(["school_key", "year"]).copy()
    df["student_growth_1yr"] = df.groupby("school_key")["student_count"].pct_change()
    df["student_diff_1yr"] = df.groupby("school_key")["student_count"].diff()
    df["next_status"] = df.groupby("school_key")["상태"].shift(-1)
    df["next_year"] = df.groupby("school_key")["year"].shift(-1)
    df["missing_next_year"] = (df["next_year"] != df["year"] + 1).astype(float)
    df.loc[df["year"].eq(df["year"].max()), "missing_next_year"] = np.nan
    df["closed_next_year_label"] = (
        df["next_status"].astype(str).str.contains("폐", na=False) | df["missing_next_year"].eq(1)
    ).astype(float)
    df.loc[df["year"].eq(df["year"].max()), "closed_next_year_label"] = np.nan
    return df


def compute_isolation_features(current_like: pd.DataFrame) -> pd.DataFrame:
    cols = ["school_key", "학교명", "학교급", "sido_name", "lttud", "lgtud"]
    base = current_like.dropna(subset=["lttud", "lgtud"]).drop_duplicates("school_key")[cols].copy()
    rows = []
    for level, group in base.groupby("학교급"):
        group = group.reset_index(drop=True)
        coords = np.deg2rad(group[["lttud", "lgtud"]].to_numpy(float))
        tree = BallTree(coords, metric="haversine")
        nearest_dist = np.full(len(group), np.nan)
        same_5km = np.zeros(len(group), dtype=int)
        if len(group) > 1:
            dist, _ = tree.query(coords, k=2)
            nearest_dist = dist[:, 1] * EARTH_RADIUS_KM
        for idxs_idx, idxs in enumerate(tree.query_radius(coords, r=5 / EARTH_RADIUS_KM)):
            same_5km[idxs_idx] = max(len(idxs) - 1, 0)
        out = group[["school_key"]].copy()
        out["nearest_same_level_school_km"] = nearest_dist
        out["same_level_school_count_5km"] = same_5km
        out["school_isolation_score"] = (
            np.nan_to_num(nearest_dist, nan=20).clip(0, 20) / 20 * 70
            + (1 - np.minimum(same_5km, 10) / 10) * 30
        ).round(1)
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["school_key"])


def load_radius_features() -> pd.DataFrame:
    path = PROCESSED / "school_radius_commercial_features.csv"
    if not path.exists():
        return pd.DataFrame(columns=["schlCd"])
    radius = pd.read_csv(path, low_memory=False)
    keep = [
        "schlCd",
        "radius_0_5km_all_shops",
        "radius_0_5km_education_shops",
        "radius_0_5km_kids_shops",
        "radius_1_0km_all_shops",
        "radius_1_0km_education_shops",
        "radius_1_0km_kids_shops",
        "radius_2_0km_all_shops",
        "radius_2_0km_education_shops",
        "radius_2_0km_kids_shops",
    ]
    return radius[[c for c in keep if c in radius.columns]].drop_duplicates("schlCd")


def main() -> int:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    panel = load_long_schooldata()
    coords = load_current_coordinate_reference()
    merged = panel.merge(coords, on="school_key", how="left", suffixes=("", "_eduinfo"))
    isolation = compute_isolation_features(merged[merged["year"].eq(2025)])
    radius = load_radius_features()
    merged = merged.merge(isolation, on="school_key", how="left")
    if not radius.empty and "schlCd" in merged.columns:
        merged = merged.merge(radius, on="schlCd", how="left")
    merged = add_temporal_targets(merged)

    output = PROCESSED / "schooldata_modeling_panel_2008_2025.csv"
    merged.to_csv(output, index=False, encoding="utf-8-sig")

    report = pd.DataFrame(
        [
            {
                "dataset": output.name,
                "rows": len(merged),
                "years": f"{int(merged['year'].min())}-{int(merged['year'].max())}",
                "coordinate_match_rate": merged["lttud"].notna().mean(),
                "active_2025_coordinate_match_rate": merged[
                    merged["year"].eq(2025) & merged["is_active_status"].eq(1)
                ]["lttud"].notna().mean(),
                "closed_next_year_positive_rate": merged["closed_next_year_label"].mean(),
            }
        ]
    )
    report.to_csv(REPORTS / "schooldata_modeling_panel_report.csv", index=False, encoding="utf-8-sig")
    print("saved:", output)
    print(report.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
