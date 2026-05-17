from __future__ import annotations

from pathlib import Path

import requests

from config_loader import load_api_keys


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "outputs" / "reports"

BASE = "https://www.eduinfo.go.kr"


def request_json(session: requests.Session, path: str, data: dict[str, str], referer: str) -> dict:
    response = session.post(
        BASE + path,
        data=data,
        headers={"X-Requested-With": "XMLHttpRequest", "Referer": BASE + referer},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def compact(row: dict, keys: list[str]) -> dict:
    return {key: row.get(key) for key in keys}


def main() -> int:
    keys = load_api_keys()
    edu_key = keys.get("EDU_INFO_API_KEY")
    session = requests.Session()
    lines: list[str] = []

    lines.append("[지방교육재정알리미 API 테스트]")
    lines.append(f"EDU_INFO_API_KEY 로드: {'성공' if edu_key else '실패'}")

    if edu_key:
        response = requests.get(
            "http://openapi.eduinfo.go.kr/openApi.do",
            params={"requestType": "opclTotal", "key": edu_key, "Type": "json", "pIndex": "1", "pSize": "3"},
            timeout=30,
        )
        lines.append(f"공식 OpenAPI 인증 테스트: HTTP {response.status_code}, 정상문구 포함={('정상 처리' in response.text)}")

    session.get(BASE + "/portal/theme/abolSchMapPage.do", timeout=30)
    closed_data = request_json(
        session,
        "/portal/map/abolSchInfoDetail.do",
        {"searchRg": "N10", "searchOffc": "", "searchWd": "", "mapSeq": ""},
        "/portal/theme/abolSchMapPage.do",
    )
    closed_rows = closed_data.get("result", [])
    lines.append(f"충남 폐교정보 조회: {len(closed_rows)}건")
    if closed_rows:
        lines.append(
            "폐교정보 샘플: "
            + str(
                compact(
                    closed_rows[0],
                    ["abolSchNm", "abolSchYy", "mtrpPrvcNm", "grade", "realAddr", "pointX", "pointY", "useState"],
                )
            )
        )

    session.get(BASE + "/portal/theme/schNmlStatusPage.do", timeout=30)
    school_data = request_json(
        session,
        "/portal/theme/searchSchNmlStatusList.do",
        {"mobileCheck": "N", "eduOffcDivCd": "N10", "sggCd": "", "fondScCd": "", "schlKndCd": "", "schlNm": ""},
        "/portal/theme/schNmlStatusPage.do",
    )
    school_rows = school_data.get("result", [])
    lines.append(f"충남 학교일반현황 조회: {len(school_rows)}건")
    if school_rows:
        first = school_rows[0]
        lines.append(
            "학교일반현황 샘플: "
            + str(
                compact(
                    first,
                    ["schlCd", "schlNm", "schlKndCd", "estbDivCd", "sggCd", "schulRdnma", "openDate", "lttud", "lgtud"],
                )
            )
        )
        detail = request_json(
            session,
            "/portal/theme/searchSchNmlStatusSheet1.do",
            {"schlCd": str(first.get("schlCd", "")), "schlKndCd": str(first.get("schlKndCd", ""))},
            "/portal/theme/schNmlStatusPage.do",
        )
        detail_rows = detail.get("data", [])
        if detail_rows:
            lines.append(
                "학생/학급 세부 샘플: "
                + str(
                    compact(
                        detail_rows[0],
                        ["ymq", "totStdtCnt", "totClassCnt", "gradCnt", "stdtCnt1", "stdtCnt2", "stdtCnt3"],
                    )
                )
            )

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "eduinfo_api_test_summary.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
