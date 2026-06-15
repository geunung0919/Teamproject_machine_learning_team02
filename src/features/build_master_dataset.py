from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from src.common.paths import (
    RAW_DIR, CLEAN_BUILD_DIR, CLEAN_PATCH_DIR, rel
)
from src.common.io import read_csv, write_csv

YEARS = list(range(2012, 2026))
TARGET_LEVELS = {"초등학교", "중학교", "고등학교"}
METRO_SIDO = {"서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종"}

SIDO_BY_CODE = {
    11: "서울", 21: "부산", 22: "대구", 23: "인천", 24: "광주", 25: "대전", 26: "울산",
    29: "세종", 31: "경기", 32: "강원", 33: "충북", 34: "충남", 35: "전북", 36: "전남",
    37: "경북", 38: "경남", 39: "제주",
}
SIDO_ALIASES = {
    "서울특별시": "서울", "서울": "서울", "부산광역시": "부산", "부산": "부산", "대구광역시": "대구", "대구": "대구",
    "인천광역시": "인천", "인천": "인천", "광주광역시": "광주", "광주": "광주", "대전광역시": "대전", "대전": "대전",
    "울산광역시": "울산", "울산": "울산", "세종특별자치시": "세종", "세종": "세종", "경기도": "경기", "경기": "경기",
    "강원도": "강원", "강원특별자치도": "강원", "강원": "강원", "충청북도": "충북", "충북": "충북",
    "충청남도": "충남", "충남": "충남", "전라북도": "전북", "전북특별자치도": "전북", "전북": "전북",
    "전라남도": "전남", "전남": "전남", "경상북도": "경북", "경북": "경북", "경상남도": "경남", "경남": "경남",
    "제주특별자치도": "제주", "제주도": "제주", "제주": "제주",
}

def norm_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    return "".join(str(x).split())

def norm_sido(x: Any) -> str:
    s = "" if pd.isna(x) else str(x).strip()
    return SIDO_ALIASES.get(s, s)

def stable_school_key(row: pd.Series) -> str:
    return "|".join([
        norm_sido(row.get("sido", "")),
        str(row.get("sgg", "")).strip(),
        str(row.get("school_level", "")).strip(),
        norm_text(row.get("school_name_raw", row.get("school_name_norm", row.get("school_name", "")))),
        str(row.get("branch_type", "")).strip(),
    ])

def size_bucket(n: Any) -> str:
    if pd.isna(n):
        return "missing"
    n = float(n)
    if n == 0:
        return "0"
    if n <= 30:
        return "1_30"
    if n <= 60:
        return "31_60"
    if n <= 100:
        return "61_100"
    if n <= 300:
        return "101_300"
    if n <= 600:
        return "301_600"
    if n <= 1000:
        return "601_1000"
    return "1000_plus"

def region_group(sido: str) -> str:
    if sido in METRO_SIDO:
        return "metro"
    if sido == "경기":
        return "capital_area"
    return "province"

def safe_growth(curr: Any, prev: Any) -> float:
    if pd.isna(curr) or pd.isna(prev) or prev == 0:
        return np.nan
    return (float(curr) - float(prev)) / abs(float(prev))

def build_school_panel(school: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    src = school[school["year"].isin(YEARS)].copy()
    src["school_level"] = src["school_level"].astype(str).str.strip()
    src["sido"] = src["sido"].map(norm_sido)
    src["is_target_level"] = src["school_level"].isin(TARGET_LEVELS)
    filter_audit = src.groupby(["year", "school_level", "is_target_level"]).size().reset_index(name="row_count")
    
    panel = src[src["is_target_level"]].copy()
    panel["school_key"] = panel.apply(stable_school_key, axis=1)
    panel["school_name"] = panel["school_name_raw"]
    panel["address"] = panel["address_raw"]
    panel["students_per_class"] = panel["students_per_class_calc"]
    panel["students_per_teacher"] = panel["students_per_teacher_calc"]
    
    keep = [
        "school_key", "year", "survey_date", "sido", "sgg", "school_level", "school_name", "school_name_norm",
        "branch_type", "foundation_type", "status", "address", "student_count", "class_count", "teacher_count",
        "students_per_class", "students_per_teacher", "land_area", "land_area_per_student",
    ]
    panel = panel[keep].copy()
    panel = panel.sort_values(["school_key", "year", "survey_date"]).drop_duplicates(["school_key", "year"], keep="last")
    
    for col in ["student_count", "class_count", "teacher_count", "students_per_class", "students_per_teacher", "land_area", "land_area_per_student"]:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")
        
    g = panel.groupby("school_key", group_keys=False)
    for lag in [1, 2, 3]:
        panel[f"student_count_lag_{lag}"] = g["student_count"].shift(lag)
    panel["student_delta_lag_1"] = panel["student_count"] - panel["student_count_lag_1"]
    panel["student_growth_lag_1"] = [safe_growth(c, p) for c, p in zip(panel["student_count"], panel["student_count_lag_1"])]
    panel["student_rolling_mean_3"] = g["student_count"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    panel["student_rolling_delta_mean_3"] = g["student_count"].transform(lambda s: s.diff().shift(1).rolling(3, min_periods=1).mean())
    panel["student_trend_slope_3"] = g["student_count"].transform(lambda s: s.shift(1).rolling(3).apply(lambda x: float(np.polyfit(np.arange(3), x, 1)[0]), raw=False))
    panel["student_rolling_mean_5"] = g["student_count"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    panel["student_trend_slope_5"] = g["student_count"].transform(lambda s: s.shift(1).rolling(5).apply(lambda x: float(np.polyfit(np.arange(5), x, 1)[0]), raw=False))
    
    panel["size_bucket"] = panel["student_count"].map(size_bucket)
    panel["student_size_bin"] = panel["size_bucket"]
    panel["metro_flag"] = panel["sido"].isin(METRO_SIDO)
    panel["region_group"] = panel["sido"].map(region_group)
    panel["level_size_segment"] = panel["school_level"].astype(str) + "|" + panel["size_bucket"].astype(str)
    
    return panel, filter_audit

def build_school_master(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dup = panel[panel.duplicated(["school_key", "year"], keep=False)].copy()
    rows = []
    for key, g in panel.sort_values("year").groupby("school_key"):
        latest = g.iloc[-1]
        names = g["school_name"].dropna().astype(str).unique().tolist()
        locs = g[["sido", "sgg", "school_level", "branch_type"]].drop_duplicates()
        entity_needed = len(names) > 1 or len(locs) > 1
        rows.append({
            "school_key": key,
            "school_name_canonical": names[-1] if names else latest["school_name"],
            "school_name_latest": latest["school_name"],
            "sido_latest": latest["sido"],
            "sgg_latest": latest["sgg"],
            "school_level": latest["school_level"],
            "branch_type_latest": latest["branch_type"],
            "foundation_type_latest": latest["foundation_type"],
            "first_observed_year": int(g["year"].min()),
            "last_observed_year": int(g["year"].max()),
            "years_observed_count": int(g["year"].nunique()),
            "entity_status": "entity_resolution_needed" if entity_needed else "stable_candidate",
            "entity_resolution_confidence": "medium" if entity_needed else "high",
            "entity_resolution_note": "name/location/branch changed within school_key" if entity_needed else "",
        })
    master = pd.DataFrame(rows)
    master_audit = pd.DataFrame({
        "metric": ["school_master_rows", "entity_resolution_needed_count", "duplicate_school_key_year_rows"],
        "value": [len(master), int((master["entity_status"] == "entity_resolution_needed").sum()), len(dup)],
    })
    return master, master_audit, dup

def build_corrected_demographics(
    panel: pd.DataFrame, pop: pd.DataFrame, birth: pd.DataFrame, mig: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pop = pop.copy()
    pop["year"] = (pd.to_numeric(pop["period"], errors="coerce") // 100).astype("Int64")
    pop["sgg_code_str"] = pop["sgg_code"].astype(str).str.zfill(5)
    pop["sido"] = pop["sgg_code_str"].str[:2].map(SIDO_BY_CODE)
    pop["sgg"] = pop["sgg_name"].astype(str)
    pop = pop[pop["year"].isin(YEARS)]
    
    pop_piv = pop.pivot_table(index=["year", "sido", "sgg", "sgg_code"], columns="age_group", values="population", aggfunc="first").reset_index()
    pop_piv.columns.name = None
    pop_piv = pop_piv.rename(columns={"0 - 4세": "age_0_4_pop", "5 - 9세": "age_5_9_pop", "10 - 14세": "age_10_14_pop", "15 - 19세": "age_15_19_pop"})
    
    for c in ["age_0_4_pop", "age_5_9_pop", "age_10_14_pop", "age_15_19_pop"]:
        if c not in pop_piv:
            pop_piv[c] = np.nan
    pop_piv["school_age_population_0_19"] = pop_piv[["age_0_4_pop", "age_5_9_pop", "age_10_14_pop", "age_15_19_pop"]].sum(axis=1, min_count=1)

    birth = birth.copy()
    birth["sido"] = birth["sido"].map(norm_sido)
    birth["sgg"] = birth["sgg"].astype(str)
    birth = birth[birth["year"].isin(YEARS)]
    birth = birth.groupby(["year", "sido", "sgg"], as_index=False).agg(
        region_code=("region_code", "first"),
        birth_count=("birth_count", lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan),
        total_fertility_rate=("total_fertility_rate", lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan),
    )

    mig = mig.copy()
    mig = mig[(mig["year"].isin(YEARS)) & (mig["region_level"].eq("sgg"))].copy()
    mig["code_str"] = mig["region_code"].astype(str).str.zfill(5)
    mig["sido"] = mig["code_str"].str[:2].map(SIDO_BY_CODE)
    mig["sgg"] = mig["region_name"].astype(str)
    
    mig_piv = mig.pivot_table(index=["year", "sido", "sgg"], columns="item_name", values="value", aggfunc="first").reset_index()
    mig_piv.columns.name = None
    mig_piv = mig_piv.rename(columns={"순이동": "net_migration_total", "총전입": "in_migration_total", "총전출": "out_migration_total"})

    demo = pop_piv.merge(
        birth[["year", "sido", "sgg", "region_code", "birth_count", "total_fertility_rate"]],
        on=["year", "sido", "sgg"],
        how="left",
    ).merge(
        mig_piv[["year", "sido", "sgg", "net_migration_total", "in_migration_total", "out_migration_total"]],
        on=["year", "sido", "sgg"],
        how="left",
    )
    
    demo = demo.sort_values(["sido", "sgg", "year"])
    for c in ["school_age_population_0_19", "birth_count", "total_fertility_rate", "net_migration_total"]:
        demo[c] = pd.to_numeric(demo[c], errors="coerce")
        
    g = demo.groupby(["sido", "sgg"], group_keys=False)
    demo["school_age_population_delta_1y"] = g["school_age_population_0_19"].diff()
    demo["school_age_population_growth_1y"] = g["school_age_population_0_19"].pct_change(fill_method=None)
    demo["birth_count_yoy_change"] = g["birth_count"].diff()
    demo["birth_count_yoy_rate"] = g["birth_count"].pct_change(fill_method=None)
    demo["tfr_yoy_change"] = g["total_fertility_rate"].diff()
    demo["tfr_yoy_rate"] = g["total_fertility_rate"].pct_change(fill_method=None)
    demo["net_migration_yoy_change"] = g["net_migration_total"].diff()
    
    cols = [
        "year", "sido", "sgg", "sgg_code", "age_0_4_pop", "age_5_9_pop", "age_10_14_pop", "age_15_19_pop",
        "school_age_population_0_19", "school_age_population_delta_1y", "school_age_population_growth_1y",
        "birth_count", "total_fertility_rate", "birth_count_yoy_change", "birth_count_yoy_rate", "tfr_yoy_change",
        "tfr_yoy_rate", "net_migration_total", "in_migration_total", "out_migration_total", "net_migration_yoy_change",
    ]
    demo = demo[cols]

    panel_regions = panel[["year", "sido", "sgg"]].drop_duplicates()
    sanity_rows = []
    join_rows = []
    unmatched_rows = []
    joined = panel.merge(demo, on=["year", "sido", "sgg"], how="left")
    
    for y in YEARS:
        d = demo[demo["year"].eq(y)]
        pr = panel_regions[panel_regions["year"].eq(y)]
        school_y = panel[panel["year"].eq(y)]
        joined_y = joined[joined["year"].eq(y)]
        sanity_rows.append({
            "year": y,
            "sgg_demo_rows": len(d),
            "unique_sido_count": d["sido"].nunique(),
            "unique_sgg_count": d[["sido", "sgg"]].drop_duplicates().shape[0],
            "school_panel_sgg_count": pr[["sido", "sgg"]].drop_duplicates().shape[0],
            "school_age_nonnull_rate": d["school_age_population_0_19"].notna().mean(),
            "birth_nonnull_rate": d["birth_count"].notna().mean(),
            "migration_nonnull_rate": d["net_migration_total"].notna().mean(),
        })
        non_school = joined_y["school_age_population_0_19"].notna()
        birth_non = joined_y["birth_count"].notna()
        mig_non = joined_y["net_migration_total"].notna()
        unmatched_subset = joined_y[~(non_school & birth_non & mig_non)]
        
        join_rows.append({
            "year": y,
            "school_rows": len(school_y),
            "joined_rows": int(non_school.sum()),
            "school_age_population_nonnull_rate": non_school.mean(),
            "birth_count_nonnull_rate": birth_non.mean(),
            "migration_nonnull_rate": mig_non.mean(),
            "unmatched_school_region_count": unmatched_subset[["sido", "sgg"]].drop_duplicates().shape[0],
            "unmatched_school_rows": len(unmatched_subset),
        })
        if len(unmatched_subset):
            grp = unmatched_subset.groupby(["year", "sido", "sgg"]).agg(
                school_rows=("school_key", "size"),
                missing_school_age_population=("school_age_population_0_19", lambda s: s.isna().any()),
                missing_birth_fertility=("birth_count", lambda s: s.isna().any()),
                missing_migration=("net_migration_total", lambda s: s.isna().any()),
            ).reset_index()
            grp["recommended_fix"] = "check_region_alias_or_source_missing"
            unmatched_rows.append(grp)
            
    unmatched_df = pd.concat(unmatched_rows, ignore_index=True) if unmatched_rows else pd.DataFrame(columns=["year", "sido", "sgg", "school_rows", "missing_school_age_population", "missing_birth_fertility", "missing_migration", "recommended_fix"])
    return demo, pd.DataFrame(sanity_rows), pd.DataFrame(join_rows), unmatched_df

def build_isolation_features(panel: pd.DataFrame, geo: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    g = geo.rename(columns={"시도": "sido", "행정구": "sgg", "학교급": "school_level", "학교명": "school_name", "lttud": "latitude", "lgtud": "longitude"}).copy()
    g = g[g["year"].isin(YEARS)]
    
    p = panel[["school_key", "year", "sido", "sgg", "school_level", "school_name"]].copy()
    p["geo_join_key"] = p["year"].astype(str) + "|" + p["sido"].astype(str) + "|" + p["sgg"].astype(str) + "|" + p["school_level"].astype(str) + "|" + p["school_name"].map(norm_text)
    g["geo_join_key"] = g["year"].astype(str) + "|" + g["sido"].map(norm_sido).astype(str) + "|" + g["sgg"].astype(str) + "|" + g["school_level"].astype(str) + "|" + g["school_name"].map(norm_text)
    
    geo_small = g[["geo_join_key", "latitude", "longitude", "coordinate_source"]].drop_duplicates("geo_join_key")
    iso = p.merge(geo_small, on="geo_join_key", how="left")
    iso["latitude"] = pd.to_numeric(iso["latitude"], errors="coerce")
    iso["longitude"] = pd.to_numeric(iso["longitude"], errors="coerce")
    
    valid = iso["latitude"].between(33, 39) & iso["longitude"].between(124, 132)
    iso["coordinate_valid"] = valid
    iso["coordinate_invalid_reason"] = np.where(iso["latitude"].isna() | iso["longitude"].isna(), "missing", np.where(~valid, "outside_korea_bbox", ""))
    
    for c in ["nearest_same_level_distance_km", "second_nearest_same_level_distance_km", "same_level_school_count_within_3km", "same_level_school_count_within_5km", "same_level_school_count_within_10km", "isolation_score"]:
        iso[c] = np.nan
    iso["no_same_level_school_within_5km_flag"] = pd.Series(pd.NA, index=iso.index, dtype="object")
    
    for (year, level), idx in iso[iso["coordinate_valid"]].groupby(["year", "school_level"]).groups.items():
        idx_list = list(idx)
        coords = np.radians(iso.loc[idx_list, ["latitude", "longitude"]].to_numpy())
        if len(coords) < 2:
            continue
        tree = BallTree(coords, metric="haversine")
        dists, _ = tree.query(coords, k=min(3, len(coords)))
        radius = 6371.0088
        nearest = dists[:, 1] * radius
        second = dists[:, 2] * radius if dists.shape[1] > 2 else np.nan
        cnt3 = tree.query_radius(coords, r=3 / radius, count_only=True) - 1
        cnt5 = tree.query_radius(coords, r=5 / radius, count_only=True) - 1
        cnt10 = tree.query_radius(coords, r=10 / radius, count_only=True) - 1
        
        iso.loc[idx_list, "nearest_same_level_distance_km"] = nearest
        iso.loc[idx_list, "second_nearest_same_level_distance_km"] = second
        iso.loc[idx_list, "same_level_school_count_within_3km"] = cnt3
        iso.loc[idx_list, "same_level_school_count_within_5km"] = cnt5
        iso.loc[idx_list, "same_level_school_count_within_10km"] = cnt10
        iso.loc[idx_list, "no_same_level_school_within_5km_flag"] = cnt5 == 0
        iso.loc[idx_list, "isolation_score"] = nearest / (1 + cnt5)
        
    iso["isolation_score_version"] = "v5_haversine_nearest_div_1_plus_5km_count"
    audit = iso.groupby("year").agg(
        rows=("school_key", "size"),
        coordinate_valid_rate=("coordinate_valid", "mean"),
        isolation_nonnull_rate=("isolation_score", lambda s: s.notna().mean()),
    ).reset_index()
    coord_audit = iso.groupby(["year", "coordinate_invalid_reason"]).size().reset_index(name="row_count")
    
    cols = ["school_key", "year", "latitude", "longitude", "coordinate_source", "coordinate_valid", "coordinate_invalid_reason", "nearest_same_level_distance_km", "second_nearest_same_level_distance_km", "same_level_school_count_within_3km", "same_level_school_count_within_5km", "same_level_school_count_within_10km", "no_same_level_school_within_5km_flag", "isolation_score", "isolation_score_version"]
    return iso[cols], coord_audit, audit

def build_grade_flow_features(school: pd.DataFrame, panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    src = school[(school["year"].isin(YEARS)) & (school["school_level"].isin(TARGET_LEVELS))].copy()
    src["sido"] = src["sido"].map(norm_sido)
    src["school_key"] = src.apply(stable_school_key, axis=1)
    src = src.sort_values(["school_key", "year", "survey_date"]).drop_duplicates(["school_key", "year"], keep="last")
    
    cols = ["school_key", "year"] + [f"grade{i}_student_count" for i in range(1, 7)] + [f"grade{i}_class_count" for i in range(1, 7)] + ["grade_student_sum", "grade_class_sum", "entrants_total", "graduates_total", "transfer_in", "transfer_out", "grade_student_sum_diff", "grade_class_sum_diff"]
    gf = src[[c for c in cols if c in src.columns]].copy()
    gf = gf.merge(panel[["school_key", "year", "school_level", "student_count", "class_count"]], on=["school_key", "year"], how="left")
    
    grade_cols = [f"grade{i}_student_count" for i in range(1, 7)]
    class_cols = [f"grade{i}_class_count" for i in range(1, 7)]
    for c in grade_cols + class_cols + ["entrants_total", "graduates_total", "transfer_in", "transfer_out"]:
        if c in gf.columns:
            gf[c] = pd.to_numeric(gf[c], errors="coerce")
            
    gf["grade_student_sum"] = gf[[c for c in grade_cols if c in gf.columns]].sum(axis=1, min_count=1)
    gf["grade_class_sum"] = gf[[c for c in class_cols if c in gf.columns]].sum(axis=1, min_count=1)
    
    for i in range(1, 7):
        if f"grade{i}_student_count" in gf.columns:
            gf[f"grade{i}_share"] = gf[f"grade{i}_student_count"] / gf["grade_student_sum"].replace({0: np.nan})
            
    gf["lower_grade_student_count"] = gf[["grade1_student_count", "grade2_student_count", "grade3_student_count"]].sum(axis=1, min_count=1)
    gf["upper_grade_student_count"] = gf[["grade4_student_count", "grade5_student_count", "grade6_student_count"]].sum(axis=1, min_count=1)
    gf["graduating_grade_student_count"] = np.where(gf["school_level"].eq("초등학교"), gf["grade6_student_count"], gf["grade3_student_count"])
    
    gf["grade_imbalance_range"] = gf[[c for c in grade_cols if c in gf.columns]].max(axis=1) - gf[[c for c in grade_cols if c in gf.columns]].min(axis=1)
    gf["grade_imbalance_std"] = gf[[c for c in grade_cols if c in gf.columns]].std(axis=1)
    
    gf["student_diff"] = gf["student_count"] - gf["grade_student_sum"]
    gf["class_diff"] = gf["class_count"] - gf["grade_class_sum"]
    gf["grade_data_valid"] = gf["student_diff"].abs().fillna(999999) <= 3
    gf["grade_invalid_reason"] = np.where(gf["grade_data_valid"], "", "grade_sum_mismatch")
    
    grade_mis = gf[gf["student_diff"].abs().fillna(0) > 3][["school_key", "year", "school_level", "student_count", "grade_student_sum", "student_diff"]].copy()
    class_mis = gf[gf["class_diff"].abs().fillna(0) > 1][["school_key", "year", "school_level", "class_count", "grade_class_sum", "class_diff"]].copy()
    
    out_cols = ["school_key", "year"] + [c for c in grade_cols if c in gf.columns] + ["grade_student_sum"] + [f"grade{i}_share" for i in range(1, 7) if f"grade{i}_share" in gf.columns] + [c for c in class_cols if c in gf.columns] + ["grade_class_sum", "entrants_total", "graduates_total", "transfer_in", "transfer_out", "lower_grade_student_count", "upper_grade_student_count", "graduating_grade_student_count", "grade_imbalance_range", "grade_imbalance_std", "grade_data_valid", "grade_invalid_reason"]
    return gf[out_cols], grade_mis, class_mis

def build_targets(panel: pd.DataFrame) -> pd.DataFrame:
    base = panel[["school_key", "year", "student_count"]].rename(columns={"year": "base_year", "student_count": "base_student_count"})
    out = base.copy()
    for horizon in [1, 2, 3, 4, 5]:
        target = panel[["school_key", "year", "student_count"]].rename(columns={"year": f"target_year_{horizon}yr", "student_count": f"target_student_count_{horizon}yr"})
        out[f"target_year_{horizon}yr"] = out["base_year"] + horizon
        out = out.merge(target, on=["school_key", f"target_year_{horizon}yr"], how="left")
        out[f"target_delta_{horizon}yr"] = out[f"target_student_count_{horizon}yr"] - out["base_student_count"]
        out[f"target_available_{horizon}yr"] = out[f"target_student_count_{horizon}yr"].notna()
    out = out.drop(columns=["base_student_count"])
    return out

def anomaly_flags(panel: pd.DataFrame, iso: pd.DataFrame, grade_mis: pd.DataFrame, class_mis: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    flags = panel[["school_key", "year", "student_count"]].copy()
    flags["standard_model_eligible"] = True
    flags["scenario_base_eligible"] = flags["year"].eq(2025)
    for c in ["event_school_candidate", "critical_student_count_anomaly", "adjacent_year_anomaly", "multi_year_pattern_anomaly", "zero_student_flag", "drop_to_zero_flag", "jump_from_zero_flag", "temporary_zero_gap_flag", "multi_year_zero_then_jump_flag", "persistent_level_shift_flag", "coordinate_missing_flag", "coordinate_outlier_flag", "grade_sum_mismatch_flag", "class_sum_mismatch_flag", "entity_resolution_needed"]:
        flags[c] = False
    flags["exclusion_reason"] = ""
    flags["quality_note"] = ""
    
    anomalies = []
    patterns = []
    
    for key, g in panel.sort_values("year").groupby("school_key"):
        vals = g[["year", "student_count"]].set_index("year")["student_count"].to_dict()
        years = sorted(vals)
        seq = [vals[y] for y in years]
        
        for y in years[1:]:
            a, b = vals.get(y - 1, np.nan), vals.get(y, np.nan)
            if pd.isna(a) or pd.isna(b):
                continue
            types = []
            if a >= 50 and b == 0:
                types.append("drop_to_zero")
            if a == 0 and b >= 300:
                types.append("jump_from_zero")
            if abs(b - a) >= 200:
                types.append("large_abs_jump")
            if a >= 30 and abs(safe_growth(b, a)) >= 0.5:
                types.append("large_pct_jump")
            if types:
                anomalies.append({
                    "school_key": key, "prev_year": y - 1, "year": y, "student_count_prev": a, 
                    "student_count_current": b, "delta": b - a, "pct_change": safe_growth(b, a), 
                    "anomaly_type": ";".join(types), "severity": "critical" if any(t in types for t in ["drop_to_zero", "jump_from_zero"]) else "high"
                })
                idx = (flags["school_key"].eq(key) & flags["year"].eq(y))
                flags.loc[idx, "adjacent_year_anomaly"] = True
                flags.loc[idx, "critical_student_count_anomaly"] = any(t in types for t in ["drop_to_zero", "jump_from_zero"])
                flags.loc[idx, "drop_to_zero_flag"] = "drop_to_zero" in types
                flags.loc[idx, "jump_from_zero_flag"] = "jump_from_zero" in types
                
        for i in range(2, len(seq)):
            if seq[i - 2] == 0 and seq[i - 1] == 0 and seq[i] >= 300:
                patterns.append({
                    "school_key": key, "pattern_type": "multi_year_zero_then_jump", 
                    "start_year": years[i - 2], "end_year": years[i], 
                    "student_count_sequence": ",".join(map(lambda x: "" if pd.isna(x) else str(int(x)), seq))
                })
                flags.loc[flags["school_key"].eq(key), "multi_year_zero_then_jump_flag"] = True
                
        for i in range(1, len(seq) - 1):
            if seq[i] == 0 and seq[i - 1] >= 50 and seq[i + 1] >= 50:
                patterns.append({
                    "school_key": key, "pattern_type": "temporary_zero_gap", 
                    "start_year": years[i - 1], "end_year": years[i + 1], 
                    "student_count_sequence": ",".join(map(lambda x: "" if pd.isna(x) else str(int(x)), seq))
                })
                flags.loc[flags["school_key"].eq(key), "temporary_zero_gap_flag"] = True
                
    flags["zero_student_flag"] = flags["student_count"].eq(0)
    coord = iso[["school_key", "year", "coordinate_invalid_reason"]].copy()
    flags = flags.merge(coord, on=["school_key", "year"], how="left")
    flags["coordinate_missing_flag"] = flags["coordinate_invalid_reason"].eq("missing")
    flags["coordinate_outlier_flag"] = flags["coordinate_invalid_reason"].eq("outside_korea_bbox")
    flags = flags.drop(columns=["coordinate_invalid_reason"])
    
    gm = grade_mis[["school_key", "year"]].drop_duplicates()
    cm = class_mis[["school_key", "year"]].drop_duplicates()
    flags.loc[flags.set_index(["school_key", "year"]).index.isin(gm.set_index(["school_key", "year"]).index), "grade_sum_mismatch_flag"] = True
    flags.loc[flags.set_index(["school_key", "year"]).index.isin(cm.set_index(["school_key", "year"]).index), "class_sum_mismatch_flag"] = True
    
    anomaly_cols = ["critical_student_count_anomaly", "adjacent_year_anomaly", "multi_year_zero_then_jump_flag", "temporary_zero_gap_flag"]
    flags["event_school_candidate"] = flags[anomaly_cols].any(axis=1)
    flags.loc[flags["event_school_candidate"], "standard_model_eligible"] = False
    flags.loc[flags["event_school_candidate"], "scenario_base_eligible"] = False
    flags["exclusion_reason"] = np.where(flags["event_school_candidate"], "student_count_event_or_entity_anomaly", "")
    flags["quality_note"] = np.where(flags["coordinate_missing_flag"] | flags["coordinate_outlier_flag"], "coordinate_issue", "")
    
    suspicious = flags[flags["event_school_candidate"] | flags["coordinate_missing_flag"] | flags["coordinate_outlier_flag"] | flags["grade_sum_mismatch_flag"] | flags["class_sum_mismatch_flag"]].copy()
    suspicious["review_id"] = ["REV%06d" % (i + 1) for i in range(len(suspicious))]
    return flags.drop(columns=["student_count"]), pd.DataFrame(anomalies), pd.DataFrame(patterns), suspicious

def patch_flags(
    flags: pd.DataFrame, join_audit: pd.DataFrame, unmatched: pd.DataFrame, 
    gmis: pd.DataFrame, cmis: pd.DataFrame, panel: pd.DataFrame, demo: pd.DataFrame
) -> pd.DataFrame:
    out = flags.copy()
    join = panel[["school_key", "year", "sido", "sgg"]].merge(demo[["year", "sido", "sgg", "school_age_population_0_19", "birth_count", "total_fertility_rate", "net_migration_total"]], on=["year", "sido", "sgg"], how="left")
    out = out.merge(join[["school_key", "year", "school_age_population_0_19", "birth_count", "total_fertility_rate", "net_migration_total"]], on=["school_key", "year"], how="left")
    
    out["demographics_join_missing_flag"] = out["school_age_population_0_19"].isna()
    out["birth_fertility_missing_pair_flag"] = out["birth_count"].isna() | out["total_fertility_rate"].isna()
    out["migration_join_missing_flag"] = out["net_migration_total"].isna()
    out = out.drop(columns=["school_age_population_0_19", "birth_count", "total_fertility_rate", "net_migration_total"])
    
    gidx = set(map(tuple, gmis[["school_key", "year"]].drop_duplicates().values.tolist()))
    cidx = set(map(tuple, cmis[["school_key", "year"]].drop_duplicates().values.tolist()))
    idx = list(map(tuple, out[["school_key", "year"]].values.tolist()))
    
    out["grade_sum_mismatch_flag"] = [x in gidx for x in idx]
    out["class_sum_mismatch_flag"] = [x in cidx for x in idx]
    out["grade_flow_audit_valid"] = ~(out["grade_sum_mismatch_flag"] | out["class_sum_mismatch_flag"])
    out["clean_dataset_patch_issue_flag"] = out[["demographics_join_missing_flag", "birth_fertility_missing_pair_flag", "migration_join_missing_flag", "grade_sum_mismatch_flag", "class_sum_mismatch_flag"]].any(axis=1)
    
    out["r1_model_eligible_patched"] = out["standard_model_eligible"] & ~out["demographics_join_missing_flag"]
    out["r2_model_eligible_patched"] = out["r1_model_eligible_patched"] & ~out.get("coordinate_missing_flag", False) & ~out.get("coordinate_outlier_flag", False)
    out["r3_model_eligible_patched"] = out["r2_model_eligible_patched"] & out["grade_flow_audit_valid"]
    
    reasons = []
    for _, row in out.iterrows():
        r = []
        for c in ["demographics_join_missing_flag", "birth_fertility_missing_pair_flag", "migration_join_missing_flag", "grade_sum_mismatch_flag", "class_sum_mismatch_flag"]:
            if bool(row.get(c, False)):
                r.append(c)
        reasons.append(";".join(r))
    out["patch_exclusion_reason"] = reasons
    return out

def build_model_base_dataset(
    panel: pd.DataFrame, demo: pd.DataFrame, iso: pd.DataFrame, grade: pd.DataFrame, 
    targets: pd.DataFrame, flags: pd.DataFrame
) -> pd.DataFrame:
    df = panel.merge(demo, on=["year", "sido", "sgg"], how="left")
    df = df.merge(iso.drop(columns=["latitude", "longitude"], errors="ignore"), on=["school_key", "year"], how="left")
    df = df.merge(grade, on=["school_key", "year"], how="left")
    df = df.merge(targets, left_on=["school_key", "year"], right_on=["school_key", "base_year"], how="left").drop(columns=["base_year"])
    df = df.merge(flags, on=["school_key", "year"], how="left")
    return df
