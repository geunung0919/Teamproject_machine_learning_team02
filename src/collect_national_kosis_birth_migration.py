from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from config_loader import load_api_keys


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
KOSIS_URL = "https://kosis.kr/openapi/Param/statisticsParameterData.do"
ORG_ID = "101"

BIRTH_TABLE_ID = "DT_1B81A23"
MIGRATION_TABLE_ID = "DT_1B26007"


def get_json(params: dict[str, str], timeout: int = 180) -> Any:
    url = KOSIS_URL + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
    data = json.loads(body)
    if isinstance(data, dict) and data.get("err"):
        raise RuntimeError(f"KOSIS error: {data.get('err')} {data.get('errMsg')}")
    return data


def collect_birth(api_key: str, start: str = "2011", end: str = "2025") -> pd.DataFrame:
    params = {
        "method": "getList",
        "apiKey": api_key,
        "itmId": "ALL",
        "objL1": "ALL",
        "format": "json",
        "jsonVD": "Y",
        "prdSe": "Y",
        "startPrdDe": start,
        "endPrdDe": end,
        "orgId": ORG_ID,
        "tblId": BIRTH_TABLE_ID,
    }
    frame = pd.DataFrame(get_json(params))
    out = pd.DataFrame(
        {
            "year": pd.to_numeric(frame["PRD_DE"], errors="coerce").astype("Int64"),
            "region_code": frame["C1"].astype(str),
            "region_name": frame["C1_NM"],
            "item_code": frame["ITM_ID"],
            "item_name": frame["ITM_NM"],
            "value": pd.to_numeric(frame["DT"], errors="coerce"),
            "table_id": frame["TBL_ID"],
            "table_name": frame["TBL_NM"],
            "last_change_date": frame.get("LST_CHN_DE", ""),
        }
    )
    out["region_level"] = out["region_code"].str.len().map({2: "sido", 5: "sgg"}).fillna("other")
    return out.sort_values(["region_code", "year", "item_code"]).reset_index(drop=True)


def build_birth_features(raw: pd.DataFrame) -> pd.DataFrame:
    sgg = raw[raw["region_level"].eq("sgg")].copy()
    wide = (
        sgg.pivot_table(
            index=["region_code", "region_name", "year"],
            columns="item_code",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={"T1": "birth_count", "T2": "total_fertility_rate"})
    )
    wide["sido_code"] = wide["region_code"].astype(str).str[:2]
    wide = wide.sort_values(["region_code", "year"])
    wide["birth_count_yoy_change"] = wide.groupby("region_code")["birth_count"].diff()
    wide["birth_count_yoy_rate"] = wide.groupby("region_code")["birth_count"].pct_change()
    wide["tfr_yoy_change"] = wide.groupby("region_code")["total_fertility_rate"].diff()
    wide["tfr_yoy_rate"] = wide.groupby("region_code")["total_fertility_rate"].pct_change()
    return wide.reset_index(drop=True)


def collect_migration(api_key: str, start: str = "2011", end: str = "2025") -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for year in range(int(start), int(end) + 1):
        params = {
            "method": "getList",
            "apiKey": api_key,
            "itmId": "ALL",
            "objL1": "ALL",
            "objL2": "00",
            "format": "json",
            "jsonVD": "Y",
            "prdSe": "Y",
            "startPrdDe": str(year),
            "endPrdDe": str(year),
            "orgId": ORG_ID,
            "tblId": MIGRATION_TABLE_ID,
        }
        frame = pd.DataFrame(get_json(params))
        frames.append(frame)
        print(f"migration year collected: {year}, rows={len(frame)}")
        time.sleep(0.1)
    frame = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame(
        {
            "year": pd.to_numeric(frame["PRD_DE"], errors="coerce").astype("Int64"),
            "region_code": frame["C1"].astype(str),
            "region_name": frame["C1_NM"],
            "movement_scale_code": frame["C2"].astype(str),
            "movement_scale_name": frame["C2_NM"],
            "item_code": frame["ITM_ID"],
            "item_name": frame["ITM_NM"],
            "value": pd.to_numeric(frame["DT"], errors="coerce"),
            "unit": frame.get("UNIT_NM", ""),
            "table_id": frame["TBL_ID"],
            "table_name": frame["TBL_NM"],
            "last_change_date": frame.get("LST_CHN_DE", ""),
        }
    )
    out["region_level"] = out["region_code"].str.len().map({2: "sido", 5: "sgg"}).fillna("other")
    return out.sort_values(["region_code", "year", "item_code"]).reset_index(drop=True)


def build_migration_features(raw: pd.DataFrame) -> pd.DataFrame:
    sgg = raw[(raw["region_level"].eq("sgg")) & (raw["movement_scale_code"].eq("00"))].copy()
    wide = (
        sgg.pivot_table(
            index=["region_code", "region_name", "year"],
            columns="item_name",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename(
            columns={
                "총전입": "in_migration_total",
                "총전출": "out_migration_total",
                "순이동": "net_migration_total",
                "시도간전입": "inter_sido_in",
                "시도간전출": "inter_sido_out",
                "시도내-시군구간전입": "intra_sido_in",
                "시도내-시군구간전출": "intra_sido_out",
                "시도내-시군구내이동": "within_sgg_move",
            }
        )
    )
    wide["sido_code"] = wide["region_code"].astype(str).str[:2]
    for col in ["in_migration_total", "out_migration_total", "net_migration_total"]:
        if col not in wide.columns:
            wide[col] = pd.NA
    wide = wide.sort_values(["region_code", "year"])
    wide["net_migration_yoy_change"] = wide.groupby("region_code")["net_migration_total"].diff()
    wide["in_migration_yoy_rate"] = wide.groupby("region_code")["in_migration_total"].pct_change()
    wide["out_migration_yoy_rate"] = wide.groupby("region_code")["out_migration_total"].pct_change()
    return wide.reset_index(drop=True)


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    api_key = load_api_keys().get("KOSIS_API_KEY")
    if not api_key:
        raise RuntimeError("KOSIS_API_KEY is missing")

    birth = collect_birth(api_key)
    birth.to_csv(RAW / "national_kosis_birth_tfr_sgg.csv", index=False, encoding="utf-8-sig")
    birth_features = build_birth_features(birth)
    birth_features.to_csv(PROCESSED / "national_birth_features_sgg.csv", index=False, encoding="utf-8-sig")

    migration = collect_migration(api_key)
    migration.to_csv(RAW / "national_kosis_migration_sgg.csv", index=False, encoding="utf-8-sig")
    migration_features = build_migration_features(migration)
    migration_features.to_csv(PROCESSED / "national_migration_features_sgg.csv", index=False, encoding="utf-8-sig")

    print(f"birth raw rows: {len(birth)}")
    print(f"birth feature rows: {len(birth_features)}")
    print(f"migration raw rows: {len(migration)}")
    print(f"migration feature rows: {len(migration_features)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
