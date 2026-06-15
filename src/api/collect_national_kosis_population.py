from __future__ import annotations

import json
import time
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
TABLE_ID = "DT_1B04005N"
ITEM_ID = "T2"
AGE_GROUPS = {
    "5": "pop_0_4",
    "10": "pop_5_9",
    "15": "pop_10_14",
    "20": "pop_15_19",
}


def get_json(params: dict[str, str], timeout: int = 180) -> Any:
    url = KOSIS_URL + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
    data = json.loads(body)
    if isinstance(data, dict) and data.get("err"):
        raise RuntimeError(f"KOSIS error: {data.get('err')} {data.get('errMsg')}")
    return data


def discover_sgg_codes(api_key: str) -> pd.DataFrame:
    params = {
        "method": "getList",
        "apiKey": api_key,
        "itmId": ITEM_ID,
        "objL1": "ALL",
        "objL2": "5",
        "format": "json",
        "jsonVD": "Y",
        "prdSe": "M",
        "startPrdDe": "202512",
        "endPrdDe": "202512",
        "orgId": ORG_ID,
        "tblId": TABLE_ID,
    }
    data = get_json(params)
    frame = pd.DataFrame(data)
    codes = frame[["C1", "C1_NM"]].drop_duplicates().copy()
    codes["code_len"] = codes["C1"].astype(str).str.len()
    sgg = codes[codes["code_len"].eq(5)].copy()
    sgg = sgg.rename(columns={"C1": "sgg_code", "C1_NM": "sgg_name"}).drop(columns=["code_len"])
    sgg["sido_code"] = sgg["sgg_code"].astype(str).str[:2]
    return sgg.sort_values("sgg_code").reset_index(drop=True)


def collect_population(
    api_key: str,
    sgg_codes: list[str],
    start: str = "201201",
    end: str = "202512",
    chunk_size: int = 20,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for chunk_start in range(0, len(sgg_codes), chunk_size):
        chunk = sgg_codes[chunk_start : chunk_start + chunk_size]
        frames.append(collect_population_chunk(api_key, chunk, start, end))
        print(f"collected KOSIS sgg {chunk_start + len(chunk)} / {len(sgg_codes)}")
        time.sleep(0.2)
    return pd.concat(frames, ignore_index=True).sort_values(["sgg_code", "period", "age_group_code"])


def collect_population_chunk(api_key: str, sgg_codes: list[str], start: str, end: str) -> pd.DataFrame:
    params = {
        "method": "getList",
        "apiKey": api_key,
        "itmId": ITEM_ID,
        "objL1": " ".join(sgg_codes),
        "objL2": " ".join(AGE_GROUPS),
        "format": "json",
        "jsonVD": "Y",
        "prdSe": "M",
        "startPrdDe": start,
        "endPrdDe": end,
        "orgId": ORG_ID,
        "tblId": TABLE_ID,
    }
    rows = get_json(params, timeout=240)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("KOSIS returned no rows")
    out = pd.DataFrame(
        {
            "period": frame["PRD_DE"],
            "sgg_code": frame["C1"].astype(str),
            "sgg_name": frame["C1_NM"],
            "age_group_code": frame["C2"].astype(str),
            "age_group": frame["C2_NM"],
            "population": pd.to_numeric(frame["DT"], errors="coerce").fillna(0).astype(int),
        }
    )
    return out.sort_values(["sgg_code", "period", "age_group_code"]).reset_index(drop=True)


def build_features(raw: pd.DataFrame, codes: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        raw.pivot_table(
            index=["sgg_code", "sgg_name", "period"],
            columns="age_group_code",
            values="population",
            aggfunc="sum",
        )
        .reset_index()
        .rename(columns=AGE_GROUPS)
    )
    for col in AGE_GROUPS.values():
        if col not in pivot.columns:
            pivot[col] = 0
    pivot["school_age_pop_0_19"] = pivot[list(AGE_GROUPS.values())].sum(axis=1)
    pivot = pivot.merge(codes[["sgg_code", "sido_code"]], on="sgg_code", how="left")
    pivot["period_dt"] = pd.to_datetime(pivot["period"].astype(str), format="%Y%m")
    pivot["year"] = pivot["period_dt"].dt.year
    pivot["month"] = pivot["period_dt"].dt.month
    pivot = pivot.sort_values(["sgg_code", "period"])
    pivot["school_age_pop_mom_change"] = pivot.groupby("sgg_code")["school_age_pop_0_19"].diff()
    pivot["school_age_pop_mom_rate"] = pivot.groupby("sgg_code")["school_age_pop_0_19"].pct_change()
    return pivot.drop(columns=["period_dt"])


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    api_key = load_api_keys().get("KOSIS_API_KEY")
    if not api_key:
        raise RuntimeError("KOSIS_API_KEY is missing")

    codes = discover_sgg_codes(api_key)
    codes.to_csv(RAW / "national_kosis_sgg_codes.csv", index=False, encoding="utf-8-sig")
    print(f"sgg codes: {len(codes)}")

    raw = collect_population(api_key, codes["sgg_code"].astype(str).tolist())
    raw.to_csv(RAW / "national_kosis_school_age_population_sgg.csv", index=False, encoding="utf-8-sig")
    print(f"raw rows: {len(raw)}")
    time.sleep(0.1)

    features = build_features(raw, codes)
    features.to_csv(PROCESSED / "national_population_features_sgg.csv", index=False, encoding="utf-8-sig")
    print(f"feature rows: {len(features)}")
    print(features[["period", "sgg_code", "sgg_name", "school_age_pop_0_19"]].head().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

