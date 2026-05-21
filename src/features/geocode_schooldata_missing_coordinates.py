from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import time

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import requests

from config_loader import load_api_keys


ROOT = SRC.parent
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "outputs" / "reports"

PANEL_PATH = PROCESSED / "schooldata_modeling_panel_2008_2025.csv"
GEOCODE_RESULTS = PROCESSED / "schooldata_missing_coordinate_geocoding_results.csv"
GEOCODED_PANEL = PROCESSED / "schooldata_modeling_panel_2008_2025_geocoded.csv"
REPORT_PATH = REPORTS / "schooldata_coordinate_fill_report.csv"

COL_LEVEL = "학교급"
COL_NAME = "학교명"
COL_STATUS = "상태"
COL_ADDRESS = "주소"


def safe_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


PROVINCE_ALIASES = {
    "강원 ": "강원도 ",
    "경기 ": "경기도 ",
    "경남 ": "경상남도 ",
    "경북 ": "경상북도 ",
    "광주 ": "광주광역시 ",
    "대구 ": "대구광역시 ",
    "대전 ": "대전광역시 ",
    "부산 ": "부산광역시 ",
    "서울 ": "서울특별시 ",
    "울산 ": "울산광역시 ",
    "인천 ": "인천광역시 ",
    "전남 ": "전라남도 ",
    "전북 ": "전라북도 ",
    "제주 ": "제주특별자치도 ",
    "충남 ": "충청남도 ",
    "충북 ": "충청북도 ",
}


def normalize_old_address(address: str) -> list[str]:
    text = re.sub(r"\s+", " ", safe_text(address)).strip()
    if not text:
        return []

    variants = [text]
    cleaned = (
        text.replace(" 번지", "")
        .replace("번지", "")
        .replace("번", "")
        .replace("번 지", "")
        .strip()
    )
    variants.append(cleaned)

    # Old lot-number addresses often come as "연주리451" or "백전리101-2".
    spaced = re.sub(r"([가-힣](?:리|동|가|읍|면))(\d)", r"\1 \2", cleaned)
    variants.append(spaced)

    # Some rows accidentally glue a village name and road name: "두창리복분로 29".
    road_spaced = re.sub(
        r"([가-힣]+(?:리|동))([가-힣]+(?:로|길|대로)\s*\d)",
        r"\1 \2",
        spaced,
    )
    variants.append(road_spaced)

    # Some rows contain administrative village numbers such as "죽정리1리 372".
    village_spaced = re.sub(r"([가-힣]+리)(\d+리)\s*(\d)", r"\1 \2 \3", road_spaced)
    variants.append(village_spaced)

    for src, dst in PROVINCE_ALIASES.items():
        for base in [text, cleaned, spaced, road_spaced, village_spaced]:
            if base.startswith(src):
                variants.append(dst + base[len(src) :])

    return list(dict.fromkeys(v for v in variants if v))


def address_variants(address: str, school_name: str) -> list[str]:
    address = safe_text(address)
    school_name = safe_text(school_name)
    variants = []
    for base_address in normalize_old_address(address):
        variants.append(base_address)
        before_paren = address.split("(")[0].strip()
        for before_paren in normalize_old_address(before_paren):
            if before_paren and before_paren != base_address:
                variants.append(before_paren)
        if school_name and school_name not in base_address:
            variants.append(f"{base_address} {school_name}")
    return list(dict.fromkeys(v for v in variants if v))


def valid_korea_coord(lat: float | None, lon: float | None) -> bool:
    if lat is None or lon is None:
        return False
    return 33.0 <= lat <= 38.7 and 124.0 <= lon <= 131.5


def geocode_vworld(address: str, api_key: str, timeout: int = 10) -> dict[str, object] | None:
    if not api_key:
        return None
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
            return {"status": "error", "source": "vworld", "error": str(exc)}
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
    return None


def geocode_naver(address: str, client_id: str, client_secret: str, timeout: int = 10) -> dict[str, object] | None:
    if not client_id or not client_secret:
        return None
    url = "https://maps.apigw.ntruss.com/map-geocode/v2/geocode"
    headers = {
        "X-NCP-APIGW-API-KEY-ID": client_id,
        "X-NCP-APIGW-API-KEY": client_secret,
    }
    try:
        response = requests.get(url, params={"query": address}, headers=headers, timeout=timeout)
        data = response.json()
    except Exception as exc:
        return {"status": "error", "source": "naver", "error": str(exc)}
    addresses = data.get("addresses") or []
    if addresses:
        first = addresses[0]
        lon = float(first["x"])
        lat = float(first["y"])
        if valid_korea_coord(lat, lon):
            return {
                "status": "ok",
                "source": "naver",
                "lat": lat,
                "lon": lon,
                "matched_address": first.get("roadAddress") or first.get("jibunAddress") or address,
            }
    return None


def _kakao_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"KakaoAK {api_key}"}


def geocode_kakao_address(address: str, api_key: str, timeout: int = 10) -> dict[str, object] | None:
    if not api_key:
        return None
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    try:
        response = requests.get(url, params={"query": address}, headers=_kakao_headers(api_key), timeout=timeout)
        data = response.json()
    except Exception as exc:
        return {"status": "error", "source": "kakao_address", "error": str(exc)}
    documents = data.get("documents") or []
    if documents:
        first = documents[0]
        lon = float(first["x"])
        lat = float(first["y"])
        if valid_korea_coord(lat, lon):
            matched = first.get("road_address") or first.get("address") or {}
            return {
                "status": "ok",
                "source": "kakao_address",
                "lat": lat,
                "lon": lon,
                "matched_address": matched.get("address_name") or first.get("address_name") or address,
            }
    return None


def geocode_kakao_keyword(query: str, api_key: str, timeout: int = 10) -> dict[str, object] | None:
    if not api_key:
        return None
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    try:
        response = requests.get(url, params={"query": query}, headers=_kakao_headers(api_key), timeout=timeout)
        data = response.json()
    except Exception as exc:
        return {"status": "error", "source": "kakao_keyword", "error": str(exc)}
    documents = data.get("documents") or []
    if documents:
        first = documents[0]
        lon = float(first["x"])
        lat = float(first["y"])
        if valid_korea_coord(lat, lon):
            return {
                "status": "ok",
                "source": "kakao_keyword",
                "lat": lat,
                "lon": lon,
                "matched_address": first.get("road_address_name") or first.get("address_name") or query,
            }
    return None


def build_targets(panel: pd.DataFrame) -> pd.DataFrame:
    missing = panel[panel["lttud"].isna() | panel["lgtud"].isna()].copy()
    missing = missing[missing[COL_ADDRESS].notna() & missing[COL_NAME].notna()].copy()
    cols = ["school_key", COL_NAME, COL_LEVEL, COL_STATUS, COL_ADDRESS, "sido_name"]
    return (
        missing[cols]
        .drop_duplicates(["school_key", COL_ADDRESS])
        .sort_values(["sido_name", COL_NAME, COL_ADDRESS])
        .reset_index(drop=True)
    )


def load_existing_results() -> pd.DataFrame:
    if GEOCODE_RESULTS.exists():
        return pd.read_csv(GEOCODE_RESULTS, low_memory=False)
    return pd.DataFrame()


def apply_results_to_panel(panel: pd.DataFrame, result_df: pd.DataFrame, target_count: int) -> pd.DataFrame:
    if not result_df.empty:
        result_df = result_df.drop_duplicates("geocode_key", keep="last")
    ok = result_df[result_df["geocode_status"].eq("ok")].copy() if not result_df.empty else pd.DataFrame()

    filled = panel.copy()
    filled["coordinate_source"] = "eduinfo_current_match"
    filled.loc[filled["lttud"].isna() | filled["lgtud"].isna(), "coordinate_source"] = "missing"

    if not ok.empty:
        ok_map = ok[["geocode_key", "geocoded_lttud", "geocoded_lgtud", "coordinate_source"]].copy()
        filled["geocode_key"] = filled["school_key"].astype(str) + "|" + filled[COL_ADDRESS].astype(str)
        filled = filled.merge(ok_map, on="geocode_key", how="left", suffixes=("", "_geocode"))
        mask = filled["lttud"].isna() & filled["geocoded_lttud"].notna()
        filled.loc[mask, "lttud"] = filled.loc[mask, "geocoded_lttud"]
        filled.loc[mask, "lgtud"] = filled.loc[mask, "geocoded_lgtud"]
        filled.loc[mask, "coordinate_source"] = filled.loc[mask, "coordinate_source_geocode"]
        filled = filled.drop(
            columns=[
                c
                for c in ["geocode_key", "geocoded_lttud", "geocoded_lgtud", "coordinate_source_geocode"]
                if c in filled.columns
            ]
        )

    filled.to_csv(GEOCODED_PANEL, index=False, encoding="utf-8-sig")
    report = pd.DataFrame(
        [
            {
                "rows": len(panel),
                "coordinate_rows_before": int(panel["lttud"].notna().sum()),
                "coordinate_rate_before": float(panel["lttud"].notna().mean()),
                "unique_geocode_targets": int(target_count),
                "geocode_cache_rows_unique": int(len(result_df)),
                "geocode_success_rows_unique": int(result_df["geocode_status"].eq("ok").sum()) if len(result_df) else 0,
                "geocode_success_rate_unique": float(result_df["geocode_status"].eq("ok").mean()) if len(result_df) else 0.0,
                "coordinate_rows_after": int(filled["lttud"].notna().sum()),
                "coordinate_rate_after": float(filled["lttud"].notna().mean()),
                "coordinate_rows_filled": int(filled["lttud"].notna().sum() - panel["lttud"].notna().sum()),
            }
        ]
    )
    report.to_csv(REPORT_PATH, index=False, encoding="utf-8-sig")
    print(report.to_string(index=False))
    print("saved:", GEOCODED_PANEL)
    return filled


def main() -> int:
    parser = argparse.ArgumentParser(description="Fill missing school coordinates using VWorld and Naver geocoding.")
    parser.add_argument("--limit", type=int, default=None, help="Limit new geocoding requests for testing.")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many retry targets before applying limit.")
    parser.add_argument("--sleep", type=float, default=0.05, help="Sleep seconds between requests.")
    parser.add_argument("--retry-failed", action="store_true", help="Retry previously failed geocoding rows.")
    parser.add_argument("--apply-cache-only", action="store_true", help="Apply existing geocoding cache without new API calls.")
    parser.add_argument("--skip-kakao", action="store_true", help="Skip Kakao Local API calls.")
    parser.add_argument("--skip-naver", action="store_true", help="Skip Naver geocoding API calls.")
    args = parser.parse_args()

    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(PANEL_PATH, low_memory=False)
    panel["lttud"] = pd.to_numeric(panel["lttud"], errors="coerce")
    panel["lgtud"] = pd.to_numeric(panel["lgtud"], errors="coerce")
    targets = build_targets(panel)
    existing = load_existing_results()

    if args.apply_cache_only:
        apply_results_to_panel(panel, existing, len(targets))
        return 0

    keys = load_api_keys()
    done_keys = set()
    rows = []
    if not existing.empty:
        rows.extend(existing.to_dict("records"))
        if args.retry_failed:
            done_keys = set(existing.loc[existing["geocode_status"].eq("ok"), "geocode_key"].astype(str))
        else:
            done_keys = set(existing["geocode_key"].astype(str))

    new_targets = targets.copy()
    new_targets["geocode_key"] = new_targets["school_key"].astype(str) + "|" + new_targets[COL_ADDRESS].astype(str)
    new_targets = new_targets[~new_targets["geocode_key"].astype(str).isin(done_keys)].copy()
    if args.offset:
        new_targets = new_targets.iloc[args.offset :].copy()
    if args.limit is not None:
        new_targets = new_targets.head(args.limit)

    print(f"missing unique school/address targets: {len(targets):,}")
    print(f"already cached: {len(done_keys):,}")
    print(f"new requests this run: {len(new_targets):,}")

    for idx, row in enumerate(new_targets.itertuples(index=False), start=1):
        result = None
        tried = []
        for addr in address_variants(getattr(row, COL_ADDRESS), getattr(row, COL_NAME)):
            tried.append(addr)
            result = geocode_vworld(addr, keys.get("VWORLD_API_KEY", ""))
            if result and result.get("status") == "ok":
                break
            if not args.skip_kakao:
                result = geocode_kakao_address(addr, keys.get("KAKAO_REST_API_KEY", ""))
                if result and result.get("status") == "ok":
                    break
            if not args.skip_naver:
                result = geocode_naver(addr, keys.get("NAVER_CLIENT_ID", ""), keys.get("NAVER_CLIENT_SECRET", ""))
                if result and result.get("status") == "ok":
                    break
        if not result or result.get("status") != "ok":
            school_query = f"{getattr(row, 'sido_name')} {getattr(row, COL_NAME)}"
            tried.append(school_query)
            if not args.skip_kakao:
                result = geocode_kakao_keyword(school_query, keys.get("KAKAO_REST_API_KEY", ""))
        if not result or result.get("status") != "ok":
            result = {"status": "failed", "source": "", "lat": None, "lon": None, "matched_address": ""}
        rows.append(
            {
                "geocode_key": getattr(row, "geocode_key"),
                "school_key": getattr(row, "school_key"),
                "school_name": getattr(row, COL_NAME),
                "school_level": getattr(row, COL_LEVEL),
                "status_label": getattr(row, COL_STATUS),
                "sido_name": getattr(row, "sido_name"),
                "address": getattr(row, COL_ADDRESS),
                "geocode_status": result.get("status"),
                "coordinate_source": result.get("source"),
                "geocoded_lttud": result.get("lat"),
                "geocoded_lgtud": result.get("lon"),
                "matched_address": result.get("matched_address"),
                "tried_address_count": len(tried),
            }
        )
        if idx % 100 == 0:
            pd.DataFrame(rows).drop_duplicates("geocode_key", keep="last").to_csv(
                GEOCODE_RESULTS, index=False, encoding="utf-8-sig"
            )
            print(f"processed new {idx:,}/{len(new_targets):,}", flush=True)
        time.sleep(args.sleep)

    result_df = pd.DataFrame(rows).drop_duplicates("geocode_key", keep="last")
    result_df.to_csv(GEOCODE_RESULTS, index=False, encoding="utf-8-sig")
    apply_results_to_panel(panel, result_df, len(targets))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
