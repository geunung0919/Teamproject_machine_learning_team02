from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError, URLError
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

API_URL = "https://apis.data.go.kr/B553077/api/open/sdsc2/storeListInDong"

SIDO_CODES = {
    "11": "서울",
    "26": "부산",
    "27": "대구",
    "28": "인천",
    "29": "광주",
    "30": "대전",
    "31": "울산",
    "36": "세종",
    "41": "경기",
    "51": "강원",
    "43": "충북",
    "44": "충남",
    "52": "전북",
    "46": "전남",
    "47": "경북",
    "48": "경남",
    "50": "제주",
}

KEEP_COLUMNS = [
    "bizesId",
    "bizesNm",
    "indsLclsCd",
    "indsLclsNm",
    "indsMclsCd",
    "indsMclsNm",
    "indsSclsCd",
    "indsSclsNm",
    "ctprvnCd",
    "ctprvnNm",
    "signguCd",
    "signguNm",
    "adongCd",
    "adongNm",
    "ldongCd",
    "ldongNm",
    "rdnmAdr",
    "lon",
    "lat",
]


def get_json(url: str, timeout: int = 120) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, 6):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
            return json.loads(body)
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            wait = min(30, attempt * 3)
            print(f"request failed attempt={attempt}, wait={wait}s, error={exc}")
            time.sleep(wait)
    raise RuntimeError(f"request failed after retries: {last_error}")


def collect_sido(service_key: str, sido_code: str, sido_name: str, num_rows: int = 1000) -> pd.DataFrame:
    output = RAW / f"small_shop_{sido_code}_{sido_name}.csv"
    if output.exists() and output.stat().st_size > 0:
        cached = pd.read_csv(output, low_memory=False)
        cached_total = get_total_count(service_key, sido_code)
        if len(cached) >= cached_total:
            print(f"cached {sido_name}: {len(cached)}")
            return cached
        print(f"resume {sido_name}: cached={len(cached)} total={cached_total}")
        rows = cached.to_dict("records")
        page = len(cached) // num_rows + 1
        total = cached_total
    else:
        rows = []
        page = 1
        total = None

    while True:
        query = "serviceKey=" + urllib.parse.quote(service_key, safe="")
        query += "&" + urllib.parse.urlencode(
            {
                "divId": "ctprvnCd",
                "key": sido_code,
                "type": "json",
                "numOfRows": str(num_rows),
                "pageNo": str(page),
            }
        )
        data = get_json(API_URL + "?" + query)
        body = data.get("body", {}) if isinstance(data, dict) else {}
        items = body.get("items", []) or []
        total = int(body.get("totalCount") or total or len(items) or 0)
        rows.extend(items)
        print(f"small-shop {sido_name} page={page} items={len(items)} total={total}")
        if page % 25 == 0 or page * num_rows >= total:
            save_frame = pd.DataFrame(rows)
            for col in KEEP_COLUMNS:
                if col not in save_frame.columns:
                    save_frame[col] = pd.NA
            save_frame[KEEP_COLUMNS].drop_duplicates("bizesId").to_csv(
                output, index=False, encoding="utf-8-sig"
            )
        if not items or page * num_rows >= total:
            break
        page += 1
        time.sleep(0.03)

    frame = pd.DataFrame(rows)
    for col in KEEP_COLUMNS:
        if col not in frame.columns:
            frame[col] = pd.NA
    frame = frame[KEEP_COLUMNS].copy()
    frame.to_csv(output, index=False, encoding="utf-8-sig")
    return frame


def get_total_count(service_key: str, sido_code: str) -> int:
    query = "serviceKey=" + urllib.parse.quote(service_key, safe="")
    query += "&" + urllib.parse.urlencode(
        {"divId": "ctprvnCd", "key": sido_code, "type": "json", "numOfRows": "1", "pageNo": "1"}
    )
    data = get_json(API_URL + "?" + query)
    body = data.get("body", {}) if isinstance(data, dict) else {}
    return int(body.get("totalCount") or 0)


def build_sgg_summary(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["signguCd"] = frame["signguCd"].astype(str)
    frame["is_education"] = frame["indsLclsNm"].astype(str).str.contains("교육", na=False) | frame[
        "indsMclsNm"
    ].astype(str).str.contains("교육|학원|교습", na=False)
    frame["is_kids"] = (
        frame["indsMclsNm"].astype(str).str.contains("유아|아동|문구|서점|스포츠", na=False)
        | frame["indsSclsNm"].astype(str).str.contains("유아|아동|키즈|문구|서점|태권도|체육", na=False)
    )
    frame["is_medical"] = frame["indsLclsNm"].astype(str).str.contains("의료", na=False) | frame[
        "indsMclsNm"
    ].astype(str).str.contains("병원|의원|약국|의료", na=False)
    summary = (
        frame.groupby(["signguCd", "signguNm", "ctprvnCd", "ctprvnNm"], as_index=False)
        .agg(
            commercial_count=("bizesId", "count"),
            education_business_count=("is_education", "sum"),
            kids_business_count=("is_kids", "sum"),
            medical_business_count=("is_medical", "sum"),
        )
        .sort_values("signguCd")
    )
    return summary


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    service_key = load_api_keys().get("DATAGOKR_API_KEY")
    if not service_key:
        raise RuntimeError("DATAGOKR_API_KEY is missing")

    frames = []
    for sido_code, sido_name in SIDO_CODES.items():
        frames.append(collect_sido(service_key, sido_code, sido_name))

    national = pd.concat(frames, ignore_index=True)
    national.to_csv(RAW / "national_small_shop.csv", index=False, encoding="utf-8-sig")
    summary = build_sgg_summary(national)
    summary.to_csv(PROCESSED / "national_small_shop_sgg_summary.csv", index=False, encoding="utf-8-sig")
    print(f"national small-shop rows: {len(national)}")
    print(f"sgg summary rows: {len(summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

