from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import time
from zipfile import ZipFile
import xml.etree.ElementTree as ET

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

EXCEL_PATH = TEMP / "폐(원)교 김근형(1~40).xlsx"
RESULT_PATH = TEMP / "manual_closed_excel_geocode_results.csv"
PANEL_PATH = PROCESSED / "schooldata_modeling_panel_2008_2025_geocoded.csv"
APPLIED_PATH = PROCESSED / "schooldata_manual_closed_excel_applied.csv"
CLOSED_UNMATCHED = PROCESSED / "schooldata_closed_unmatched_after_geocode.csv"
REPORT_PATH = REPORTS / "schooldata_manual_closed_excel_apply_report.csv"

COL_YEAR = "데이터_연도"
COL_SIDO = "시도"
COL_SGG = "행정구"
COL_LEVEL = "학교급"
COL_NAME = "학교명"
COL_STATUS = "상태"
COL_ADDRESS = "주소"

NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def col_to_idx(ref: str) -> int:
    letters = "".join(re.findall("[A-Z]+", ref))
    idx = 0
    for char in letters:
        idx = idx * 26 + ord(char) - 64
    return idx - 1


def read_xlsx_without_openpyxl(path: Path) -> pd.DataFrame:
    with ZipFile(path) as archive:
        shared_strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", NS):
                texts = [node.text or "" for node in item.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")]
                shared_strings.append("".join(texts))

        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows = []
        for row in sheet.findall(".//a:sheetData/a:row", NS):
            values = {}
            for cell in row.findall("a:c", NS):
                ref = cell.attrib.get("r", "")
                idx = col_to_idx(ref)
                typ = cell.attrib.get("t")
                node = cell.find("a:v", NS)
                value = ""
                if node is not None and node.text is not None:
                    value = shared_strings[int(node.text)] if typ == "s" else node.text
                values[idx] = value
            if values:
                rows.append([values.get(i, "") for i in range(max(values) + 1)])
    header, body = rows[0], rows[1:]
    return pd.DataFrame(body, columns=header)


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


def run_geocode(args: argparse.Namespace) -> pd.DataFrame:
    keys = load_api_keys()
    api_key = keys.get("VWORLD_API_KEY", "")
    if not api_key:
        raise RuntimeError("VWORLD_API_KEY is missing.")

    targets = read_xlsx_without_openpyxl(EXCEL_PATH)
    targets = targets[targets["school_key"].notna() & targets[COL_ADDRESS].notna()].copy()
    if RESULT_PATH.exists():
        existing = pd.read_csv(RESULT_PATH, low_memory=False)
    else:
        existing = pd.DataFrame()
    done = set(existing["school_key"].astype(str)) if not existing.empty else set()
    new_targets = targets[~targets["school_key"].astype(str).isin(done)].copy()
    if args.limit is not None:
        new_targets = new_targets.head(args.limit)

    rows = existing.to_dict("records") if not existing.empty else []
    print(f"excel targets: {len(targets):,}")
    print(f"already cached: {len(done):,}")
    print(f"new requests this run: {len(new_targets):,}")
    for row in new_targets.itertuples(index=False):
        address = getattr(row, COL_ADDRESS)
        result = geocode_vworld(address, api_key)
        rows.append(
            {
                "school_key": getattr(row, "school_key"),
                "school_name": getattr(row, COL_NAME),
                "status_label": getattr(row, COL_STATUS),
                "address": address,
                "status": result.get("status"),
                "source": result.get("source"),
                "lat": result.get("lat"),
                "lon": result.get("lon"),
                "matched_address": result.get("matched_address"),
            }
        )
        time.sleep(args.sleep)

    result_df = pd.DataFrame(rows).drop_duplicates("school_key", keep="last")
    result_df.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")
    return result_df


def apply_results() -> pd.DataFrame:
    panel = pd.read_csv(PANEL_PATH, low_memory=False)
    panel["lttud"] = pd.to_numeric(panel["lttud"], errors="coerce")
    panel["lgtud"] = pd.to_numeric(panel["lgtud"], errors="coerce")
    before_missing = int((panel["lttud"].isna() | panel["lgtud"].isna()).sum())
    closed_before = panel[panel[COL_STATUS].astype(str).str.contains("폐", na=False)]
    closed_before_missing = int((closed_before["lttud"].isna() | closed_before["lgtud"].isna()).sum())

    result_df = pd.read_csv(RESULT_PATH, low_memory=False)
    ok = result_df[result_df["status"].eq("ok") & result_df["lat"].notna() & result_df["lon"].notna()].copy()
    ok = ok.drop_duplicates("school_key", keep="last")
    panel = panel.merge(
        ok[["school_key", "lat", "lon", "source", "matched_address"]],
        on="school_key",
        how="left",
        suffixes=("", "_excel"),
    )
    fill_mask = (panel["lttud"].isna() | panel["lgtud"].isna()) & panel["lat"].notna() & panel["lon"].notna()
    panel.loc[fill_mask, "lttud"] = panel.loc[fill_mask, "lat"]
    panel.loc[fill_mask, "lgtud"] = panel.loc[fill_mask, "lon"]
    panel.loc[fill_mask, "coordinate_source"] = "manual_closed_excel_" + panel.loc[fill_mask, "source"].astype(str)
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

    closed = panel[panel[COL_STATUS].astype(str).str.contains("폐", na=False)].copy()
    closed_missing = closed[closed["lttud"].isna() | closed["lgtud"].isna()].copy()
    closed_missing[
        [COL_YEAR, COL_SIDO, COL_SGG, COL_LEVEL, COL_NAME, COL_STATUS, COL_ADDRESS, "school_key", "coordinate_source"]
    ].to_csv(CLOSED_UNMATCHED, index=False, encoding="utf-8-sig")

    after_missing = int((panel["lttud"].isna() | panel["lgtud"].isna()).sum())
    closed_after_missing = int((closed["lttud"].isna() | closed["lgtud"].isna()).sum())
    report = pd.DataFrame(
        [
            {
                "result_rows": len(result_df),
                "result_ok_rows": len(ok),
                "missing_before": before_missing,
                "missing_after": after_missing,
                "filled_rows": int(fill_mask.sum()),
                "filled_unique_school_keys": int(panel.loc[fill_mask, "school_key"].nunique()),
                "closed_missing_before": closed_before_missing,
                "closed_missing_after": closed_after_missing,
            }
        ]
    )
    report.to_csv(REPORT_PATH, index=False, encoding="utf-8-sig")
    print(report.to_string(index=False))
    return panel


def main() -> int:
    parser = argparse.ArgumentParser(description="Geocode manual closed-school Excel and apply to panel.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.02)
    parser.add_argument("--apply-only", action="store_true")
    args = parser.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    if not args.apply_only:
        result_df = run_geocode(args)
        print(result_df["status"].value_counts(dropna=False).to_string())
    apply_results()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
