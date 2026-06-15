from __future__ import annotations

from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from typing import Any

import pandas as pd
import requests


ROOT = SRC.parent
RAW = ROOT / "data" / "raw"

BASE = "https://www.eduinfo.go.kr"

SIDO_CODES = {
    "ALL": "전국",
    "B10": "서울",
    "C10": "부산",
    "D10": "대구",
    "E10": "인천",
    "F10": "광주",
    "G10": "대전",
    "H10": "울산",
    "I10": "세종",
    "J10": "경기",
    "K10": "강원",
    "M10": "충북",
    "N10": "충남",
    "P10": "전북",
    "Q10": "전남",
    "R10": "경북",
    "S10": "경남",
    "T10": "제주",
}


def request_json(session: requests.Session, path: str, data: dict[str, str], referer: str) -> dict[str, Any]:
    response = session.post(
        BASE + path,
        data=data,
        headers={"X-Requested-With": "XMLHttpRequest", "Referer": BASE + referer},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def collect_closed_schools(session: requests.Session) -> pd.DataFrame:
    session.get(BASE + "/portal/theme/abolSchMapPage.do", timeout=30)
    data = request_json(
        session,
        "/portal/map/abolSchInfoDetail.do",
        {"searchRg": "ALL", "searchOffc": "", "searchWd": "", "mapSeq": ""},
        "/portal/theme/abolSchMapPage.do",
    )
    rows = data.get("result", [])
    keep = [
        "mapSeq",
        "abolSchNm",
        "abolSchYy",
        "mtrpPrvcCd",
        "mtrpPrvcNm",
        "grade",
        "realAddr",
        "roadAddr",
        "strAddr",
        "pointX",
        "pointY",
        "useState",
        "loanCd",
        "loanDesc",
        "deptNm",
        "chrgrtel",
    ]
    return pd.DataFrame(rows).reindex(columns=keep)


def collect_current_school_list(session: requests.Session, sido_code: str = "N10") -> pd.DataFrame:
    session.get(BASE + "/portal/theme/schNmlStatusPage.do", timeout=30)
    data = request_json(
        session,
        "/portal/theme/searchSchNmlStatusList.do",
        {
            "mobileCheck": "N",
            "eduOffcDivCd": sido_code,
            "sggCd": "",
            "fondScCd": "",
            "schlKndCd": "",
            "schlNm": "",
        },
        "/portal/theme/schNmlStatusPage.do",
    )
    rows = data.get("result", [])
    keep = [
        "sdEduOffcDiv",
        "schlCd",
        "schlNm",
        "rgEduOffcDiv",
        "schlKndCd",
        "estbDivCd",
        "sggCd",
        "schulRdnma",
        "schulRdmda",
        "openDate",
        "estbDate",
        "lttud",
        "lgtud",
        "userTel",
        "schlUrl",
        "delYn",
        "regDttm",
    ]
    return pd.DataFrame(rows).reindex(columns=keep)


def collect_school_detail(
    session: requests.Session,
    schools: pd.DataFrame,
    cache_path: Path | None = None,
    save_every: int = 100,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    done_codes: set[str] = set()
    if cache_path and cache_path.exists():
        cached = pd.read_csv(cache_path, low_memory=False)
        rows = cached.to_dict("records")
        done_codes = set(cached["schlCd"].astype(str))

    targets = schools.copy()
    targets["schlCd"] = targets["schlCd"].astype(str)
    targets = targets[~targets["schlCd"].isin(done_codes)]

    for idx, (_, school) in enumerate(targets.iterrows(), start=1):
        school_code = str(school["schlCd"])
        school_level = str(school["schlKndCd"])
        try:
            chart = request_json(
                session,
                "/portal/theme/searchSchNmlStatusChart1.do",
                {"schlCd": school_code, "schlKndCd": school_level},
                "/portal/theme/schNmlStatusPage.do",
            ).get("data", [])
            sheet = request_json(
                session,
                "/portal/theme/searchSchNmlStatusSheet1.do",
                {"schlCd": school_code, "schlKndCd": school_level},
                "/portal/theme/schNmlStatusPage.do",
            ).get("data", [])
        except Exception as exc:  # noqa: BLE001
            rows.append({"schlCd": school_code, "schlKndCd": school_level, "error": str(exc)})
            continue
        merged: dict[str, Any] = {"schlCd": school_code, "schlKndCd": school_level}
        if chart:
            merged.update({f"chart_{key}": value for key, value in chart[0].items()})
        if sheet:
            merged.update({f"sheet_{key}": value for key, value in sheet[0].items()})
        rows.append(merged)
        if cache_path and idx % save_every == 0:
            pd.DataFrame(rows).to_csv(cache_path, index=False, encoding="utf-8-sig")
            print(f"saved detail cache: {len(rows)} / {len(schools)}")
    if cache_path:
        pd.DataFrame(rows).to_csv(cache_path, index=False, encoding="utf-8-sig")
    return pd.DataFrame(rows)


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    closed = collect_closed_schools(session)
    closed.to_csv(RAW / "eduinfo_closed_schools_national.csv", index=False, encoding="utf-8-sig")
    closed[closed["mtrpPrvcCd"].eq("N10")].to_csv(
        RAW / "eduinfo_closed_schools_chungnam.csv", index=False, encoding="utf-8-sig"
    )

    all_current_frames = []
    for sido_code, sido_name in SIDO_CODES.items():
        if sido_code == "ALL":
            continue
        frame = collect_current_school_list(session, sido_code)
        frame["requested_sido_code"] = sido_code
        frame["requested_sido_name"] = sido_name
        all_current_frames.append(frame)
        print(f"current schools {sido_name}: {len(frame)}")

    national_schools = pd.concat(all_current_frames, ignore_index=True)
    national_schools = national_schools.drop_duplicates("schlCd")
    national_schools.to_csv(RAW / "eduinfo_current_schools_national.csv", index=False, encoding="utf-8-sig")

    chungnam_schools = national_schools[national_schools["requested_sido_code"].eq("N10")].copy()
    chungnam_schools.to_csv(RAW / "eduinfo_current_schools_chungnam.csv", index=False, encoding="utf-8-sig")

    national_details = collect_school_detail(
        session,
        national_schools,
        cache_path=RAW / "eduinfo_current_school_detail_national.csv",
        save_every=100,
    )
    chungnam_details = national_details[
        national_details["schlCd"].astype(str).isin(chungnam_schools["schlCd"].astype(str))
    ].copy()
    chungnam_details.to_csv(RAW / "eduinfo_current_school_detail_chungnam.csv", index=False, encoding="utf-8-sig")

    print(f"closed national: {len(closed)}")
    print(f"closed chungnam: {int(closed['mtrpPrvcCd'].eq('N10').sum())}")
    print(f"current schools national: {len(national_schools)}")
    print(f"current schools chungnam: {len(chungnam_schools)}")
    print(f"school detail national rows: {len(national_details)}")
    print(f"school detail chungnam rows: {len(chungnam_details)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

