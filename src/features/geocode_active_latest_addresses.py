from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import requests

from config_loader import load_api_keys


ROOT = SRC.parent
TEMP = ROOT / "data" / "temp"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "outputs" / "reports"

INPUT_PATH = TEMP / "active_unmatch_latest_addresses.csv"
RESULT_PATH = TEMP / "active_latest_geocode_results_vworld.csv"
PANEL_PATH = PROCESSED / "schooldata_modeling_panel_2008_2025_geocoded.csv"
APPLIED_PATH = PROCESSED / "schooldata_active_temp_geocode_results_applied.csv"
REPORT_PATH = REPORTS / "schooldata_active_temp_geocode_apply_report.csv"

COL_YEAR = "데이터_연도"
COL_SIDO = "시도"
COL_SGG = "행정구"
COL_LEVEL = "학교급"
COL_NAME = "학교명"
COL_STATUS = "상태"
COL_ADDRESS = "주소"


def valid_korea_coord(lat: float | None, lon: float | None) -> bool:
    return lat is not None and lon is not None and 33.0 <= lat <= 38.7 and 124.0 <= lon <= 131.5


def geocode_vworld(address: str, api_key: str, timeout: int = 10) -> dict[str, object]:
    url = "https://api.vworld.kr/req/address"
    for addr_type in ["road", "parcel"]:
        params = {
            "service": "address",
            "request": "getcoord",
            "version": "2.0",
            "crs": "epsg:4326",
            "address": address,
            "format": "json",
            "type": addr_type,
            "key": api_key,
        }
        try:
            response = requests.get(url, params=params, timeout=timeout)
            data = response.json()
        except Exception as exc:
            return {"status": "error", "source": f"vworld_{addr_type}", "error": str(exc)}
        status = str(data.get("response", {}).get("status", ""))
        if status.upper() == "OK":
            point = data["response"]["result"]["point"]
            lon = float(point["x"])
            lat = float(point["y"])
            if valid_korea_coord(lat, lon):
                return {
                    "status": "ok",
                    "source": f"vworld_{addr_type}",
                    "lat": lat,
                    "lon": lon,
                    "matched_address": address,
                }
    return {"status": "failed", "source": "", "lat": None, "lon": None, "matched_address": ""}


def load_targets() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    df = df[df["latest_address_candidate"].notna()].copy()
    df = df.sort_values(["school_key", "latest_year_for_school"]).drop_duplicates("school_key", keep="last")
    return df.reset_index(drop=True)


def load_existing() -> pd.DataFrame:
    if RESULT_PATH.exists():
        return pd.read_csv(RESULT_PATH, low_memory=False)
    return pd.DataFrame()


def run_geocode(args: argparse.Namespace) -> pd.DataFrame:
    keys = load_api_keys()
    api_key = keys.get("VWORLD_API_KEY", "")
    if not api_key:
        raise RuntimeError("VWORLD_API_KEY is missing.")

    targets = load_targets()
    existing = load_existing()
    done = set(existing["school_key"].astype(str)) if not existing.empty else set()
    new_targets = targets[~targets["school_key"].astype(str).isin(done)].copy()
    if args.offset:
        new_targets = new_targets.iloc[args.offset :].copy()
    if args.limit is not None:
        new_targets = new_targets.head(args.limit)

    rows = existing.to_dict("records") if not existing.empty else []
    print(f"targets unique school_key: {len(targets):,}")
    print(f"already cached: {len(done):,}")
    print(f"new requests this run: {len(new_targets):,}")

    for idx, row in enumerate(new_targets.itertuples(index=False), start=1):
        result = geocode_vworld(getattr(row, "latest_address_candidate"), api_key)
        rows.append(
            {
                "school_key": getattr(row, "school_key"),
                "school_name": getattr(row, COL_NAME),
                "status_label": getattr(row, COL_STATUS),
                "original_address": getattr(row, COL_ADDRESS),
                "latest_address_candidate": getattr(row, "latest_address_candidate"),
                "address_confidence": getattr(row, "address_confidence"),
                "needs_manual_check": getattr(row, "needs_manual_check"),
                "status": result.get("status"),
                "source": result.get("source"),
                "lat": result.get("lat"),
                "lon": result.get("lon"),
                "matched_address": result.get("matched_address"),
            }
        )
        if idx % 50 == 0:
            pd.DataFrame(rows).drop_duplicates("school_key", keep="last").to_csv(
                RESULT_PATH, index=False, encoding="utf-8-sig"
            )
            print(f"processed {idx:,}/{len(new_targets):,}", flush=True)
        time.sleep(args.sleep)

    result_df = pd.DataFrame(rows).drop_duplicates("school_key", keep="last")
    result_df.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")
    return result_df


def apply_results() -> pd.DataFrame:
    panel = pd.read_csv(PANEL_PATH, low_memory=False)
    panel["lttud"] = pd.to_numeric(panel["lttud"], errors="coerce")
    panel["lgtud"] = pd.to_numeric(panel["lgtud"], errors="coerce")
    before_missing = int((panel["lttud"].isna() | panel["lgtud"].isna()).sum())

    result_df = load_existing()
    ok = result_df[result_df["status"].eq("ok") & result_df["lat"].notna() & result_df["lon"].notna()].copy()
    ok = ok.drop_duplicates("school_key", keep="last")
    panel = panel.merge(
        ok[["school_key", "lat", "lon", "source", "matched_address"]],
        on="school_key",
        how="left",
        suffixes=("", "_active_temp"),
    )
    fill_mask = (panel["lttud"].isna() | panel["lgtud"].isna()) & panel["lat"].notna() & panel["lon"].notna()
    panel.loc[fill_mask, "lttud"] = panel.loc[fill_mask, "lat"]
    panel.loc[fill_mask, "lgtud"] = panel.loc[fill_mask, "lon"]
    panel.loc[fill_mask, "coordinate_source"] = "temp_active_" + panel.loc[fill_mask, "source"].astype(str)
    panel.loc[fill_mask, "matched_address"] = panel.loc[fill_mask, "matched_address"].astype(str)

    applied_cols = [
        COL_YEAR,
        COL_SIDO,
        COL_SGG,
        COL_LEVEL,
        COL_NAME,
        COL_STATUS,
        COL_ADDRESS,
        "school_key",
        "lttud",
        "lgtud",
        "coordinate_source",
        "matched_address",
    ]
    applied = panel.loc[fill_mask, [c for c in applied_cols if c in panel.columns]].copy()
    applied.to_csv(APPLIED_PATH, index=False, encoding="utf-8-sig")

    panel = panel.drop(columns=[c for c in ["lat", "lon", "source", "matched_address"] if c in panel.columns])
    panel.to_csv(PANEL_PATH, index=False, encoding="utf-8-sig")
    after_missing = int((panel["lttud"].isna() | panel["lgtud"].isna()).sum())

    report = pd.DataFrame(
        [
            {
                "panel_rows": len(panel),
                "missing_before": before_missing,
                "missing_after": after_missing,
                "filled_rows": int(fill_mask.sum()),
                "filled_unique_school_keys": int(panel.loc[fill_mask, "school_key"].nunique()),
                "result_rows": int(len(result_df)),
                "result_ok_rows": int(ok.shape[0]),
            }
        ]
    )
    report.to_csv(REPORT_PATH, index=False, encoding="utf-8-sig")
    print(report.to_string(index=False))
    return panel


def main() -> int:
    parser = argparse.ArgumentParser(description="Geocode active unmatched latest addresses using VWorld only.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.02)
    parser.add_argument("--apply-only", action="store_true")
    args = parser.parse_args()

    TEMP.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    if not args.apply_only:
        result_df = run_geocode(args)
        print(result_df["status"].value_counts(dropna=False).to_string())
    apply_results()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
