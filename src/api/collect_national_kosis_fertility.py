from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from typing import Any

import pandas as pd

from config_loader import load_api_keys


ROOT = SRC.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

KOSIS_URL = "https://kosis.kr/openapi/Param/statisticsParameterData.do"
ORG_ID = "101"
FERTILITY_TABLE_ID = "INH_1B81A17"
AGE_SPECIFIC_TABLE_ID = "DT_1B81A17"


def get_json(params: dict[str, str], timeout: int = 180) -> Any:
    url = KOSIS_URL + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
    data = json.loads(body)
    if isinstance(data, dict) and data.get("err"):
        raise RuntimeError(f"KOSIS error: {data.get('err')} {data.get('errMsg')}")
    return data


def collect_fertility(api_key: str, start: str = "2011", end: str = "2023") -> pd.DataFrame:
    params = {
        "method": "getList",
        "apiKey": api_key,
        "itmId": "T1",
        "objL1": "ALL",
        "format": "json",
        "jsonVD": "Y",
        "prdSe": "Y",
        "startPrdDe": start,
        "endPrdDe": end,
        "orgId": ORG_ID,
        "tblId": FERTILITY_TABLE_ID,
    }
    data = get_json(params)
    frame = pd.DataFrame(data)
    out = pd.DataFrame(
        {
            "year": pd.to_numeric(frame["PRD_DE"], errors="coerce").astype("Int64"),
            "region_code": frame["C1"].astype(str),
            "region_name": frame["C1_NM"],
            "total_fertility_rate": pd.to_numeric(frame["DT"], errors="coerce"),
            "table_id": frame["TBL_ID"],
            "table_name": frame["TBL_NM"],
            "last_change_date": frame.get("LST_CHN_DE", ""),
        }
    )
    out["region_code_len"] = out["region_code"].str.len()
    out["region_level"] = out["region_code_len"].map({2: "sido", 5: "sgg"}).fillna("other")
    return out.sort_values(["region_code", "year"]).reset_index(drop=True)


def collect_age_specific_fertility(api_key: str, start: str = "2011", end: str = "2023") -> pd.DataFrame:
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
        "tblId": AGE_SPECIFIC_TABLE_ID,
    }
    data = get_json(params)
    frame = pd.DataFrame(data)
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
    out["region_code_len"] = out["region_code"].str.len()
    out["region_level"] = out["region_code_len"].map({2: "sido", 5: "sgg"}).fillna("other")
    return out.sort_values(["region_code", "year", "item_code"]).reset_index(drop=True)


def build_fertility_features(fertility: pd.DataFrame) -> pd.DataFrame:
    sgg = fertility[fertility["region_level"].eq("sgg")].copy()
    sgg["sido_code"] = sgg["region_code"].str[:2]
    sgg = sgg.sort_values(["region_code", "year"])
    sgg["tfr_yoy_change"] = sgg.groupby("region_code")["total_fertility_rate"].diff()
    sgg["tfr_yoy_rate"] = sgg.groupby("region_code")["total_fertility_rate"].pct_change()
    latest = sgg.groupby("region_code")["year"].transform("max")
    sgg["is_latest_year"] = sgg["year"].eq(latest)
    return sgg.reset_index(drop=True)


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    api_key = load_api_keys().get("KOSIS_API_KEY")
    if not api_key:
        raise RuntimeError("KOSIS_API_KEY is missing")

    fertility = collect_fertility(api_key)
    fertility.to_csv(RAW / "national_kosis_total_fertility_rate.csv", index=False, encoding="utf-8-sig")

    age_specific = collect_age_specific_fertility(api_key)
    age_specific.to_csv(RAW / "national_kosis_age_specific_fertility_rate.csv", index=False, encoding="utf-8-sig")

    features = build_fertility_features(fertility)
    features.to_csv(PROCESSED / "national_fertility_features_sgg.csv", index=False, encoding="utf-8-sig")

    print(f"total fertility rows: {len(fertility)}")
    print(f"age specific fertility rows: {len(age_specific)}")
    print(f"sgg fertility feature rows: {len(features)}")
    print(features[["year", "region_code", "region_name", "total_fertility_rate"]].tail().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

