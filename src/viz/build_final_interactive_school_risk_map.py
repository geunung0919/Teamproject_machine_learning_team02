from __future__ import annotations

import json
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from project_config import (
    RISK_COLORS,
    RISK_LABEL_KO,
    RISK_SCORE_WEIGHTS,
    YEARS,
    coordinate_issue_reason,
    valid_sido_coord_mask,
)


ROOT = SRC.parent
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "outputs" / "reports"
MAPS = ROOT / "outputs" / "maps"
MAP_DATA = MAPS / "final_school_risk_data"


def clean_float(value: object, digits: int = 1) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), digits)


def clean_int(value: object) -> int | None:
    if pd.isna(value):
        return None
    return int(round(float(value)))


def extract_sgg_name(address: object, sido_name: object, sgg_code: object) -> str:
    parts = str(address).split()
    sido = str(sido_name)
    if sido == "세종":
        return "세종시"
    if len(parts) < 2:
        return str(sgg_code)
    metropolitan = {"서울", "부산", "대구", "인천", "광주", "대전", "울산"}
    if sido in metropolitan:
        return parts[1]
    if len(parts) >= 3 and parts[1].endswith("시") and parts[2].endswith("구"):
        return f"{parts[1]} {parts[2]}"
    return parts[1]


def make_payload(df: pd.DataFrame) -> list[dict[str, object]]:
    records = []
    for _, row in df.iterrows():
        risk = str(row["risk_label"])
        pressure_risk_value = row.get("pressure_model_risk_label", risk)
        change_risk_value = row.get("change_model_risk_label", risk)
        pressure_risk = risk if pd.isna(pressure_risk_value) else str(pressure_risk_value)
        change_risk = risk if pd.isna(change_risk_value) else str(change_risk_value)
        sgg_name = extract_sgg_name(row.get("schulRdnma"), row.get("requested_sido_name"), row.get("sgg_code"))
        records.append(
            {
                "year": int(row["forecast_year"]),
                "sido": str(row["requested_sido_name"]),
                "sgg": str(row["sgg_code"]),
                "sggName": sgg_name,
                "name": str(row["schlNm"]),
                "level": str(row["school_level"]),
                "lat": clean_float(row["lttud"], 6),
                "lon": clean_float(row["lgtud"], 6),
                "risk": risk,
                "riskKo": RISK_LABEL_KO.get(risk, risk),
                "color": RISK_COLORS.get(risk, "#64748b"),
                "pressureModelRisk": pressure_risk,
                "pressureModelRiskKo": RISK_LABEL_KO.get(pressure_risk, pressure_risk),
                "pressureModelColor": RISK_COLORS.get(pressure_risk, "#64748b"),
                "changeModelRisk": change_risk,
                "changeModelRiskKo": RISK_LABEL_KO.get(change_risk, change_risk),
                "changeModelColor": RISK_COLORS.get(change_risk, "#64748b"),
                "students2025": clean_int(row["student_count_2025"]),
                "forecastStudents": clean_int(row["forecast_student_count"]),
                "pressureModelForecastStudents": clean_int(row.get("pressure_model_forecast_student_count")),
                "changeModelForecastStudents": clean_int(row.get("change_model_forecast_student_count")),
                "cohortForecastStudents": clean_int(row.get("cohort_forecast_student_count")),
                "forecastModelBasis": str(row.get("forecast_model_basis", "sgg_pressure_model")),
                "pressure": clean_float(row["population_pressure_ratio"], 2),
                "pressureModelPressure": clean_float(row.get("pressure_model_population_pressure_ratio"), 2),
                "changeModelPressure": clean_float(row.get("change_model_pressure_ratio"), 2),
                "riskScore": clean_float(row["risk_score"], 0),
                "pressureModelRiskScore": clean_float(row.get("pressure_model_risk_score"), 0),
                "changeModelRiskScore": clean_float(row.get("change_model_risk_score"), 0),
                "diffScore": clean_float(row["differentiation_score"], 1),
                "isolation": clean_float(row["school_isolation_score"], 1),
                "commercial": clean_float(row["commercial_vulnerability_score"], 1),
                "regionalDecline": clean_float(row["regional_decline_risk_score"], 1),
                "objectivePct": clean_float(row["objective_closure_percentile"], 1),
                "nearestKm": clean_float(row["nearest_same_level_school_km"], 2),
                "same5km": clean_int(row["same_level_school_count_5km"]),
                "flagLowStudent": clean_int(row["pred_low_student_flag"]),
                "flagLongDecline": clean_int(row["long_term_decline_flag"]),
                "flagSevereDecline": clean_int(row["severe_decline_flag"]),
                "flagReplacementNear": clean_int(row["replacement_near_flag"]),
                "flagObjectiveTop10": clean_int(row["objective_top10_flag"]),
                "flagIsolationHigh": clean_int(row["isolation_high_flag"]),
                "flagCommercialVulnerable": clean_int(row["commercial_vulnerable_flag"]),
                "flagRegionalDeclineHigh": clean_int(row["regional_decline_high_flag"]),
            }
        )
    return records


def make_help_payload() -> str:
    help_text = {
        "riskScore": "높을수록 통폐합 검토 필요성이 큽니다.",
        "diffScore": "높을수록 이 프로젝트의 차별 피처(고립도, 상권, 학령수요 감소압력, EDSS 유사도)가 위험 쪽으로 강합니다.",
        "isolation": "높을수록 주변 같은 학교급이 멀거나 적어 교육공백 위험이 큽니다.",
        "commercial": "높을수록 주변 상권·교육·아동 관련 생활 기반이 약합니다.",
        "regionalDecline": "높을수록 출생, 출산율, 인구이동 흐름상 학령수요 감소압력이 큽니다.",
        "objectivePct": "폐교 확률이 아니라 과거 EDSS 패턴 기준 폐교·소멸 학교와의 전국 상대 유사도입니다.",
        "nearestKm": "높을수록 대체 통학 거리가 멉니다.",
        "same5km": "낮을수록 5km 안의 같은 학교급 대체 후보가 적습니다.",
        "pressure": "1보다 낮을수록 해당 지역 학령인구 감소 압력이 큽니다.",
    }
    return json.dumps(help_text, ensure_ascii=False, separators=(",", ":"))


def make_cohort_payload() -> str:
    cohort_path = PROCESSED / "school_level_cohort_scenario_2026_2040.csv"
    if not cohort_path.exists():
        return "[]"

    usecols = [
        "cohort_scenario",
        "forecast_year",
        "requested_sido_name",
        "school_level",
        "student_count_2025",
        "cohort_forecast_student_count",
    ]
    cohort = pd.read_csv(cohort_path, usecols=usecols, low_memory=False)
    cohort = cohort[cohort["forecast_year"].isin(YEARS)].copy()
    cohort["student_count_2025"] = pd.to_numeric(cohort["student_count_2025"], errors="coerce").fillna(0)
    cohort["cohort_forecast_student_count"] = pd.to_numeric(
        cohort["cohort_forecast_student_count"], errors="coerce"
    ).fillna(0)

    agg = (
        cohort.groupby(["cohort_scenario", "forecast_year", "requested_sido_name", "school_level"], as_index=False)
        .agg(
            students2025=("student_count_2025", "sum"),
            forecastStudents=("cohort_forecast_student_count", "sum"),
            schools=("cohort_forecast_student_count", "count"),
        )
        .sort_values(["cohort_scenario", "requested_sido_name", "school_level", "forecast_year"])
    )
    agg["forecast_year"] = agg["forecast_year"].astype(int)
    agg["students2025"] = agg["students2025"].round(0).astype(int)
    agg["forecastStudents"] = agg["forecastStudents"].round(0).astype(int)
    agg["schools"] = agg["schools"].astype(int)
    return json.dumps(agg.to_dict("records"), ensure_ascii=False, separators=(",", ":"))


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    MAPS.mkdir(parents=True, exist_ok=True)

    scenario_path = PROCESSED / "final_national_school_scenario_2026_2040.csv"
    df = pd.read_csv(scenario_path, low_memory=False)
    df = df[df["forecast_year"].isin(YEARS)].copy()
    change_scenario_path = PROCESSED / "final_national_school_scenario_change_model_2026_2040.csv"
    if change_scenario_path.exists():
        change = pd.read_csv(change_scenario_path, low_memory=False)
        change_cols = [
            "schlCd",
            "forecast_year",
            "forecast_student_count",
            "population_pressure_ratio",
            "risk_score",
            "risk_label",
        ]
        change = change[[col for col in change_cols if col in change.columns]].rename(
            columns={
                "forecast_student_count": "change_model_forecast_student_count",
                "population_pressure_ratio": "change_model_pressure_ratio",
                "risk_score": "change_model_risk_score",
                "risk_label": "change_model_risk_label",
            }
        )
        df = df.merge(change, on=["schlCd", "forecast_year"], how="left")
    df["lttud"] = pd.to_numeric(df["lttud"], errors="coerce")
    df["lgtud"] = pd.to_numeric(df["lgtud"], errors="coerce")

    valid_coord_mask = valid_sido_coord_mask(df)
    quality = (
        df.assign(valid_sido_coord=valid_coord_mask)
        .groupby("forecast_year", as_index=False)
        .agg(
            total_rows=("schlCd", "count"),
            valid_coord_rows=("valid_sido_coord", "sum"),
        )
    )
    quality["invalid_or_missing_coord_rows"] = quality["total_rows"] - quality["valid_coord_rows"]
    quality.to_csv(REPORTS / "final_coordinate_quality_report.csv", index=False, encoding="utf-8-sig")

    invalid = df[~valid_coord_mask].copy()
    invalid["coordinate_issue_reason"] = invalid.apply(coordinate_issue_reason, axis=1)
    invalid[
        [
            "forecast_year",
            "requested_sido_name",
            "sgg_code",
            "schlNm",
            "school_level",
            "lttud",
            "lgtud",
            "coordinate_issue_reason",
        ]
    ].to_csv(REPORTS / "final_invalid_coordinates_2026_2040.csv", index=False, encoding="utf-8-sig")

    bad_2040 = invalid[invalid["forecast_year"].eq(2040)].copy()
    bad_2040[
        ["requested_sido_name", "sgg_code", "schlNm", "school_level", "lttud", "lgtud", "coordinate_issue_reason"]
    ].to_csv(REPORTS / "final_invalid_coordinates_2040.csv", index=False, encoding="utf-8-sig")

    data_check = df[df["risk_label"].eq("data_check_needed")].copy()
    data_check[
        ["forecast_year", "requested_sido_name", "sgg_code", "schlNm", "school_level", "student_count_2025"]
    ].to_csv(REPORTS / "final_student_count_data_check_2026_2040.csv", index=False, encoding="utf-8-sig")

    special = df[df["risk_label"].eq("special_school_review")].copy()
    special[
        ["forecast_year", "requested_sido_name", "sgg_code", "schlNm", "school_level", "student_count_2025"]
    ].to_csv(REPORTS / "final_special_school_review_2026_2040.csv", index=False, encoding="utf-8-sig")

    df = df[valid_coord_mask].copy()
    help_payload = make_help_payload()
    risk_weight_payload = json.dumps(RISK_SCORE_WEIGHTS, ensure_ascii=False, separators=(",", ":"))
    cohort_payload = make_cohort_payload()

    sidos = ["전체"] + sorted(df["requested_sido_name"].dropna().astype(str).unique().tolist())
    default_sido = "충남" if "충남" in sidos else "전체"
    MAP_DATA.mkdir(parents=True, exist_ok=True)
    data_files = {}
    data_manifest = []
    initial_payload = "[]"
    for idx, sido in enumerate([s for s in sidos if s != "전체"], start=1):
        sido_df = df[df["requested_sido_name"].eq(sido)].copy()
        sido_payload = json.dumps(make_payload(sido_df), ensure_ascii=False, separators=(",", ":"))
        if sido == default_sido:
            initial_payload = sido_payload
        file_name = f"sido_{idx:02d}.js"
        data_files[sido] = f"final_school_risk_data/{file_name}"
        file_path = MAP_DATA / file_name
        file_path.write_text(
            "window.SCHOOL_RISK_DATA=window.SCHOOL_RISK_DATA||{};"
            f"window.SCHOOL_RISK_DATA[{json.dumps(sido, ensure_ascii=False)}]={sido_payload};\n",
            encoding="utf-8",
        )
        data_manifest.append({"sido": sido, "rows": len(sido_df), "file": str(file_path.relative_to(MAPS))})
    pd.DataFrame(data_manifest).to_csv(REPORTS / "final_map_lazy_data_manifest.csv", index=False, encoding="utf-8-sig")
    data_file_payload = json.dumps(data_files, ensure_ascii=False, separators=(",", ":"))
    sido_options = "\n".join(
        f'<option value="{sido}" {"selected" if sido == default_sido else ""}>{sido}</option>' for sido in sidos
    )
    year_options = "\n".join(
        f'<option value="{year}" {"selected" if year == 2040 else ""}>{year}</option>' for year in YEARS
    )

    html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>전국 학교 통폐합 위험 시나리오</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css" />
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet.markercluster@1.5.3/dist/MarkerCluster.css" />
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet.markercluster@1.5.3/dist/MarkerCluster.Default.css" />
  <style>
    html, body, #map {{ height: 100%; margin: 0; font-family: Arial, 'Malgun Gothic', sans-serif; }}
    .panel {{
      position: fixed; top: 58px; right: 18px; z-index: 1000; width: 390px; max-height: calc(100vh - 74px);
      overflow: auto; background: white;
      border: 1px solid #d1d5db; border-radius: 6px; box-shadow: 0 2px 10px rgba(15,23,42,.14);
      padding: 12px;
    }}
    .nav {{
      position: fixed; top: 12px; left: 50px; z-index: 1100; display: flex; gap: 6px;
      background: white; border: 1px solid #d1d5db; border-radius: 6px; padding: 6px;
      box-shadow: 0 2px 10px rgba(15,23,42,.12);
    }}
    .nav button {{
      border: 0; border-radius: 4px; padding: 7px 12px; background: white; color: #334155;
      font-weight: 700; cursor: pointer;
    }}
    .nav button.active {{ background: #2563eb; color: white; }}
    .panel h1 {{ font-size: 15px; margin: 0 0 4px; }}
    .subtitle {{ font-size: 12px; color: #475569; margin-bottom: 9px; line-height: 1.4; }}
    .controls {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
    .control-section {{
      border-top: 1px solid #e2e8f0;
      padding-top: 10px;
      margin-top: 10px;
    }}
    .control-section:first-of-type {{ border-top: 0; padding-top: 0; margin-top: 0; }}
    .control-title {{
      font-size: 12px; font-weight: 850; color: #334155; margin-bottom: 7px;
      display: flex; align-items: center; gap: 5px;
    }}
    .control-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
    .feature-controls {{
      margin-top: 10px; border-top: 1px solid #e2e8f0; padding-top: 9px;
      display: grid; grid-template-columns: 1fr 1fr; gap: 6px 10px;
    }}
    .feature-controls .title {{ grid-column: 1 / -1; font-size: 12px; font-weight: 700; color: #334155; }}
    .feature-controls label {{ display: flex; align-items: center; gap: 6px; font-size: 12px; color: #334155; }}
    .feature-controls input {{ margin: 0; }}
    .feature-note {{ grid-column: 1 / -1; font-size: 11px; color: #64748b; line-height: 1.35; }}
    .model-summary {{
      margin-top: 10px; border: 1px solid #e2e8f0; border-radius: 6px; background: #f8fafc;
      padding: 9px 10px; font-size: 11px; color: #334155; line-height: 1.45;
    }}
    .model-summary b {{ color: #111827; }}
    .model-summary .model-title {{ font-size: 12px; font-weight: 800; margin-bottom: 4px; }}
    .model-summary .tag {{
      display:inline-block; border-radius:999px; padding:1px 6px; margin-right:4px;
      font-size:10px; font-weight:800; background:#e0f2fe; color:#075985;
    }}
    .model-summary .tag.tuned {{ background:#dcfce7; color:#166534; }}
    .model-summary .tag.proxy {{ background:#fef3c7; color:#92400e; }}
    .model-comparison-table {{
      display: grid; gap: 0; margin-top: 8px; border: 1px solid #e2e8f0;
      border-radius: 6px; overflow: hidden; font-size: 11px; background:white;
    }}
    .model-row {{ display: grid; grid-template-columns: 56px 1fr 1fr; border-bottom: 1px solid #e2e8f0; }}
    .model-row:last-child {{ border-bottom: none; }}
    .model-row.header {{ background: #f1f5f9; font-weight: 850; color: #475569; }}
    .model-row > span {{ padding: 6px 8px; border-right: 1px solid #e2e8f0; }}
    .model-row > span:last-child {{ border-right: none; }}
    .model-type {{ font-weight: 800; color: #334155; background: #f8fafc; }}
    .base-cell {{ color: #64748b; }}
    .base-cell.active {{ background: #eff6ff; color: #1d4ed8; font-weight: 850; }}
    .tuned-cell {{ color: #166534; font-weight: 800; }}
    .tuned-cell.active {{ background: #f0fdf4; }}
    .base-header, .tuned-header {{ text-align: center; }}
    .tuned-header {{ color: #166534; }}
    .model-proxy-note {{ margin-top: 7px; font-size: 10px; color:#64748b; line-height:1.35; }}
    .stat-meta {{ margin-top: 8px; font-size: 11px; color:#475569; line-height:1.35; }}
    .stat-cards {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; margin-top: 8px; }}
    .stat-card {{ border: 1px solid #e2e8f0; border-radius: 6px; padding: 8px; text-align: center; background: white; }}
    .stat-value {{ font-size: 18px; font-weight: 850; color: #111827; }}
    .stat-label {{ font-size: 10px; color: #64748b; margin-top: 2px; }}
    .stat-card.accent {{ border-color: #3b82f6; }}
    .stat-card.accent .stat-value {{ color: #1d4ed8; }}
    .stat-card.danger {{ border-color: #ef4444; }}
    .stat-card.danger .stat-value {{ color: #dc2626; }}
    .stat-card.safe {{ border-color: #22c55e; }}
    .stat-card.safe .stat-value {{ color: #15803d; }}
    .risk-summary {{ margin-top: 7px; font-size: 11px; color: #475569; line-height: 1.4; }}
    label {{ display: grid; gap: 3px; font-size: 12px; color: #475569; }}
    select {{ height: 32px; border: 1px solid #cbd5e1; border-radius: 4px; padding: 0 8px; background: white; }}
    .stats {{ margin-top: 9px; font-size: 12px; color: #334155; line-height: 1.48; }}
    .graph-panel {{
      display: none; position: fixed; top: 58px; left: 50px; right: 50px; bottom: 28px; z-index: 1050;
      background: white; border: 1px solid #d1d5db; border-radius: 6px; padding: 16px;
      box-shadow: 0 2px 14px rgba(15,23,42,.16);
    }}
    .graph-panel h2 {{ margin: 0 0 4px; font-size: 18px; }}
    .graph-panel .sub {{ color: #475569; font-size: 13px; margin-bottom: 12px; }}
    .graph-controls {{
      display: grid; grid-template-columns: 220px 1fr; gap: 12px; align-items: start; margin-bottom: 12px;
    }}
    .graph-feature-controls {{
      display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 6px 10px;
      border: 1px solid #e2e8f0; border-radius: 6px; padding: 8px;
    }}
    .graph-feature-controls .title {{ grid-column: 1 / -1; font-size: 12px; font-weight: 700; color: #334155; }}
    .graph-feature-controls label {{ display: flex; align-items: center; gap: 6px; font-size: 12px; color: #334155; }}
    .chart-wrap {{ height: 430px; }}
    #studentChart {{ display: block; width: 100%; height: 430px; }}
    .graph-stats {{ margin-top: 12px; font-size: 13px; color: #334155; line-height: 1.55; }}
    .graph-model-summary {{
      border: 1px solid #e2e8f0; border-radius: 6px; background: #f8fafc;
      padding: 10px 12px; margin-bottom: 12px; font-size: 12px; color:#334155; line-height:1.5;
    }}
    .graph-model-cards {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 12px; }}
    .graph-model-card {{ border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 12px; background: #fafbfc; }}
    .card-title {{ font-size: 12px; font-weight: 850; color: #334155; margin-bottom: 8px; }}
    .card-compare {{ display: flex; align-items: center; gap: 8px; }}
    .card-item {{ flex: 1; padding: 6px 8px; border-radius: 6px; text-align: center; }}
    .card-item.base {{ background: #f1f5f9; border: 1px solid #e2e8f0; }}
    .card-item.tuned {{ background: #f0fdf4; border: 1px solid #bbf7d0; }}
    .card-tag {{ font-size: 10px; font-weight: 850; color: #64748b; margin-bottom: 2px; }}
    .card-item.tuned .card-tag {{ color: #166534; }}
    .card-name {{ font-size: 12px; font-weight: 800; color: #111827; }}
    .card-metric {{ font-size: 11px; color: #475569; margin-top: 2px; }}
    .card-badge {{
      display: inline-block; margin-top: 4px; padding: 1px 6px; border-radius: 999px;
      background: #dcfce7; color: #166534; font-size: 10px; font-weight: 850;
    }}
    .card-vs {{ font-size: 16px; color: #94a3b8; font-weight: 850; }}
    .card-note {{ margin-top: 6px; font-size: 10px; color: #64748b; }}
    .legend {{
      position: fixed; bottom: 22px; left: 50px; z-index: 1000; background: white;
      border: 1px solid #d1d5db; border-radius: 6px; padding: 10px 12px; font-size: 12px;
      box-shadow: 0 2px 10px rgba(15,23,42,.10);
    }}
    .dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; }}
    .leaflet-popup-content {{ margin: 16px 18px; }}
    .popup {{ min-width: 360px; max-width: 430px; color:#1f2937; }}
    .popup h3 {{ margin:0; font-size:18px; line-height:1.25; }}
    .popup .meta {{ margin-top:5px; color:#64748b; font-size:12px; }}
    .popup .risk-header {{
      display:flex; align-items:center; justify-content:space-between; gap:10px;
      margin-top:12px; padding:10px 12px; border:1px solid #e2e8f0; border-radius:8px; background:#f8fafc;
    }}
    .popup .risk-title {{ font-size:13px; color:#475569; margin-bottom:2px; }}
    .popup .risk-label {{ font-size:16px; font-weight:800; color:#111827; }}
    .popup .score-pill {{
      min-width:58px; text-align:center; border-radius:999px; padding:6px 9px;
      color:white; font-weight:800; background:#334155;
    }}
    .popup .score-pill small {{ display:block; color:rgba(255,255,255,.82); font-size:10px; font-weight:600; }}
    .popup .summary-grid {{ display:grid; grid-template-columns:repeat(3, 1fr); gap:7px; margin-top:10px; }}
    .popup .summary-card {{ border:1px solid #e2e8f0; border-radius:7px; padding:8px; background:white; }}
    .popup .summary-card .label {{ font-size:11px; color:#64748b; margin-bottom:3px; }}
    .popup .summary-card .value {{ font-size:15px; font-weight:800; color:#111827; }}
    .popup .summary-card .sub {{ margin-top:2px; font-size:10px; color:#64748b; }}
    .popup .section-title {{ margin-top:12px; margin-bottom:6px; font-size:12px; font-weight:800; color:#334155; }}
    .popup .chips {{ display:flex; flex-wrap:wrap; gap:5px; }}
    .popup .chip {{ border-radius:999px; background:#eff6ff; color:#1d4ed8; padding:4px 8px; font-size:11px; font-weight:700; }}
    .popup .chip.empty {{ background:#f1f5f9; color:#64748b; }}
    .popup .metric-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:7px; }}
    .popup .metric {{
      border:1px solid #e2e8f0; border-radius:7px; padding:8px; min-width:0; background:#fff;
    }}
    .popup .metric .metric-top {{ display:flex; align-items:center; justify-content:space-between; gap:6px; margin-bottom:4px; }}
    .popup .metric .label {{ font-size:11px; color:#475569; font-weight:700; }}
    .popup .metric .value {{ font-size:17px; font-weight:850; color:#1f2937; }}
    .popup .metric .hint {{ font-size:10px; color:#64748b; line-height:1.25; margin-top:2px; }}
    .popup .metric .badge {{ border-radius:999px; padding:2px 7px; font-size:10px; font-weight:800; white-space:nowrap; }}
    .popup .metric.low .value {{ color:#15803d; }}
    .popup .metric.low .badge {{ background:#dcfce7; color:#166534; }}
    .popup .metric.mid .value {{ color:#b45309; }}
    .popup .metric.mid .badge {{ background:#fef3c7; color:#92400e; }}
    .popup .metric.high .value {{ color:#dc2626; }}
    .popup .metric.high .badge {{ background:#fee2e2; color:#b91c1c; }}
    .popup .metric.neutral .value {{ color:#1f2937; }}
    .popup .metric.neutral .badge {{ background:#f1f5f9; color:#475569; }}
    .popup .note {{ margin-top:9px; font-size:11px; color:#64748b; line-height:1.35; }}
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="nav">
    <button id="mapTab" class="active" type="button">지도</button>
    <button id="graphTab" type="button">그래프</button>
  </div>
  <div class="panel">
    <h1>전국 학교 통폐합 위험 시나리오</h1>
    <div class="subtitle">2026~2040 1년 단위 · 한국 좌표 범위 밖 이상값 제외 · 학교고립도/상권취약도/학령수요 감소압력 표시</div>
    <div class="control-section">
      <div class="control-title">데이터 필터</div>
      <div class="control-grid">
        <label>연도<select id="yearSelect">{year_options}</select></label>
        <label>지역<select id="sidoSelect">{sido_options}</select></label>
        <label>학교급<select id="levelSelect">
          <option value="전체">전체</option>
          <option value="초등학교">초등학교</option>
          <option value="중학교">중학교</option>
          <option value="고등학교">고등학교</option>
          <option value="특수학교">특수학교</option>
          <option value="각종학교">각종학교</option>
        </select></label>
        <label>위험등급<select id="riskSelect">
          <option value="전체">전체</option>
          <option value="consolidation_high_risk">통폐합 가능 고위험</option>
          <option value="education_gap_high_risk">교육공백 우려 고위험</option>
          <option value="high_risk_review">고위험 검토</option>
          <option value="special_school_review">특수/각종학교 별도검토</option>
          <option value="data_check_needed">학생수 데이터 확인필요</option>
          <option value="mid_risk">중위험</option>
          <option value="low_risk">저위험</option>
        </select></label>
      </div>
    </div>
    <div class="control-section">
      <div class="control-title">모델 설정</div>
      <div class="control-grid">
        <label>예측모델<select id="modelSelect">
          <option value="cohort" selected>튜닝 모델 (출생 코호트)</option>
          <option value="change">변화량 모델 (인구이동 반영)</option>
          <option value="pressure">베이스 모델 (시군구 압력비)</option>
        </select></label>
        <label>표시단위<select id="displayMode">
          <option value="auto" selected>자동 전환</option>
          <option value="region">시군구 감소</option>
          <option value="school">학교별 점</option>
        </select></label>
      </div>
    </div>
    <div class="feature-controls" id="featureControls">
      <div class="title">위험 피처 선택 <small style="font-weight:400;color:#64748b;">전체 선택 기본</small></div>
      <label><input type="checkbox" class="feature-toggle" data-feature="lowStudent" checked> 학생수 규모</label>
      <label><input type="checkbox" class="feature-toggle" data-feature="decline" checked> 학령인구 감소</label>
      <label><input type="checkbox" class="feature-toggle" data-feature="isolation" checked> 학교 고립도</label>
      <label><input type="checkbox" class="feature-toggle" data-feature="commercial" checked> 상권 취약도</label>
      <label><input type="checkbox" class="feature-toggle" data-feature="regional" checked> 학령수요 감소압력</label>
      <label><input type="checkbox" class="feature-toggle" data-feature="replacement" checked> 대체학교 접근성</label>
      <div class="feature-note">EDSS 유사도는 고정 가산항목으로 반영되며 체크박스 조정 대상에서만 제외했습니다. 자동 전환은 넓게 볼 때 시군구 감소 버블, 확대 시 학교별 점으로 바뀝니다.</div>
    </div>
    <div class="model-summary">
      <div class="model-title">모델 기준</div>
      <div class="model-comparison-table">
        <div class="model-row header">
          <span></span>
          <span class="base-header">베이스</span>
          <span class="tuned-header">튜닝 (사용중)</span>
        </div>
        <div class="model-row">
          <span class="model-type">회귀</span>
          <span class="base-cell">Ridge</span>
          <span class="tuned-cell active">RandomForest</span>
        </div>
        <div class="model-row">
          <span class="model-type">장기 시나리오</span>
          <span class="base-cell">압력비</span>
          <span class="tuned-cell active">코호트/변화량</span>
        </div>
        <div class="model-row">
          <span class="model-type">분류</span>
          <span class="base-cell">Logistic</span>
          <span class="tuned-cell active">HistGB</span>
        </div>
      </div>
      <div class="model-proxy-note"><span class="tag proxy">참고</span>EDSS 소멸 proxy 모델은 고정 보조지표입니다. 체크박스에서는 숨겨 임의 조작하지 않도록 했습니다.</div>
    </div>
    <div class="stats" id="stats"></div>
  </div>
  <div class="legend">
    <div id="riskLegendBlock">
      <b>위험등급</b><br>
      <span class="dot" style="background:#de2d26"></span>통폐합 가능 고위험<br>
      <span class="dot" style="background:#756bb1"></span>교육공백 우려 고위험<br>
      <span class="dot" style="background:#fb6a4a"></span>고위험 검토<br>
      <span class="dot" style="background:#3182bd"></span>특수/각종학교 별도검토<br>
      <span class="dot" style="background:#737373"></span>학생수 데이터 확인필요<br>
      <span class="dot" style="background:#fdae6b"></span>중위험<br>
      <span class="dot" style="background:#2ca25f"></span>저위험
    </div>
    <div id="regionLegendBlock">
      <b>시군구 학생수 감소</b><br>
      <span class="dot" style="background:#b91c1c"></span>30% 이상 감소<br>
      <span class="dot" style="background:#f97316"></span>20~30% 감소<br>
      <span class="dot" style="background:#facc15"></span>10~20% 감소<br>
      <span class="dot" style="background:#22c55e"></span>10% 미만 감소/증가
    </div>
  </div>
  <div class="graph-panel" id="graphPanel">
    <h2>필터 기준 학생수 회귀 예측</h2>
    <div class="sub">학생수 회귀 예측은 회귀모델 결과이고, 위험 피처 선택은 위험등급별 학교 수 그래프에서만 적용됩니다.</div>
    <div class="graph-model-cards">
      <div class="graph-model-card">
        <div class="card-title">회귀모델: 학령인구 예측</div>
        <div class="card-compare">
          <div class="card-item base">
            <div class="card-tag">베이스</div>
            <div class="card-name">Ridge</div>
            <div class="card-metric">MAE 642</div>
          </div>
          <div class="card-vs">vs</div>
          <div class="card-item tuned">
            <div class="card-tag">튜닝</div>
            <div class="card-name">RandomForest</div>
            <div class="card-metric">MAE 566</div>
            <div class="card-badge">MAE 11.9% 개선</div>
          </div>
        </div>
        <div class="card-note">RMSE는 Ridge가 더 낮아, 발표에서는 MAE/RMSE를 함께 비교합니다.</div>
        <div class="card-note">변화량 모델은 성능 우위 주장보다 인구이동에 따른 지역별 감소 속도 비교용입니다.</div>
      </div>
      <div class="graph-model-card">
        <div class="card-title">분류모델: 2019~2022 소멸 검증</div>
        <div class="card-compare">
          <div class="card-item base">
            <div class="card-tag">베이스</div>
            <div class="card-name">Logistic</div>
            <div class="card-metric">F1 0.688</div>
          </div>
          <div class="card-vs">vs</div>
          <div class="card-item tuned">
            <div class="card-tag">튜닝</div>
            <div class="card-name">HistGB</div>
            <div class="card-metric">F1 0.702</div>
            <div class="card-badge">PR-AUC 우수</div>
          </div>
        </div>
        <div class="card-note">2009~2018 학습, 2019~2022 EDSS 학교ID 다음 해 소멸 여부로 검증했습니다.</div>
      </div>
    </div>
    <div class="graph-controls">
      <label>그래프 종류<select id="chartMode">
        <option value="student">학생수 회귀 예측</option>
        <option value="threeModel">3개 예측 모델 비교</option>
        <option value="cohort">출생 코호트 장기 시나리오 비교</option>
        <option value="risk">선택 피처 기준 위험등급별 학교 수</option>
      </select></label>
      <div class="graph-feature-controls" id="graphFeatureControls">
        <div class="title">위험등급 그래프용 피처 선택</div>
        <label><input type="checkbox" class="feature-toggle graph-feature-toggle" data-feature="lowStudent" checked> 학생수 규모</label>
        <label><input type="checkbox" class="feature-toggle graph-feature-toggle" data-feature="decline" checked> 학령인구 감소</label>
        <label><input type="checkbox" class="feature-toggle graph-feature-toggle" data-feature="isolation" checked> 학교 고립도</label>
        <label><input type="checkbox" class="feature-toggle graph-feature-toggle" data-feature="commercial" checked> 상권 취약도</label>
        <label><input type="checkbox" class="feature-toggle graph-feature-toggle" data-feature="regional" checked> 학령수요 감소압력</label>
        <label><input type="checkbox" class="feature-toggle graph-feature-toggle" data-feature="replacement" checked> 대체학교 접근성</label>
        <div class="feature-note">학생수 회귀 예측값은 바꾸지 않고, 위험점수/등급 시뮬레이션에만 적용됩니다.</div>
      </div>
    </div>
    <div class="chart-wrap"><canvas id="studentChart" width="1100" height="430"></canvas></div>
    <div class="graph-stats" id="graphStats"></div>
  </div>
  <script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/leaflet.markercluster@1.5.3/dist/leaflet.markercluster.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <script>
    window.SCHOOL_RISK_DATA = window.SCHOOL_RISK_DATA || {{}};
    window.SCHOOL_RISK_DATA[{json.dumps(default_sido, ensure_ascii=False)}] = {initial_payload};
    const SIDO_DATA_FILES = {data_file_payload};
    const DEFAULT_SIDO = {json.dumps(default_sido, ensure_ascii=False)};
    const HELP = {help_payload};
    const RISK_WEIGHTS = {risk_weight_payload};
    const COHORT_AGG = {cohort_payload};
    const DATA_LOAD_PROMISES = {{}};
    const RISK_LABELS = {{
      low_risk: '저위험',
      mid_risk: '중위험',
      high_risk_review: '고위험 검토',
      consolidation_high_risk: '통폐합 가능 고위험',
      education_gap_high_risk: '교육공백 우려 고위험',
      special_school_review: '특수/각종학교 별도검토',
      data_check_needed: '학생수 데이터 확인필요',
    }};
    const RISK_COLORS = {{
      low_risk: '#2ca25f',
      mid_risk: '#fdae6b',
      high_risk_review: '#fb6a4a',
      consolidation_high_risk: '#de2d26',
      education_gap_high_risk: '#756bb1',
      special_school_review: '#3182bd',
      data_check_needed: '#737373',
    }};
    const RISK_ORDER = [
      'consolidation_high_risk',
      'education_gap_high_risk',
      'high_risk_review',
      'mid_risk',
      'low_risk',
      'special_school_review',
      'data_check_needed',
    ];
    const map = L.map('map', {{ maxBounds: [[31.5, 123], [40.5, 133.5]], maxBoundsViscosity: 0.7 }}).setView([36.65, 126.85], 8);
    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
      attribution: '&copy; OpenStreetMap &copy; CARTO', maxZoom: 19
    }}).addTo(map);
    let cluster = L.markerClusterGroup({{ disableClusteringAtZoom: 10, chunkedLoading: true }});
    let regionLayer = L.layerGroup();
    map.addLayer(cluster);
    map.addLayer(regionLayer);
    let chart = null;
    let ACTIVE_DATA = window.SCHOOL_RISK_DATA[DEFAULT_SIDO] || [];

    function loadSidoData(sido) {{
      if (window.SCHOOL_RISK_DATA[sido]) return Promise.resolve(window.SCHOOL_RISK_DATA[sido]);
      if (DATA_LOAD_PROMISES[sido]) return DATA_LOAD_PROMISES[sido];
      const src = SIDO_DATA_FILES[sido];
      if (!src) return Promise.resolve([]);
      DATA_LOAD_PROMISES[sido] = new Promise((resolve, reject) => {{
        const script = document.createElement('script');
        script.src = src;
        script.async = true;
        script.onload = () => resolve(window.SCHOOL_RISK_DATA[sido] || []);
        script.onerror = () => reject(new Error(`데이터 로딩 실패: ${{sido}}`));
        document.head.appendChild(script);
      }});
      return DATA_LOAD_PROMISES[sido];
    }}

    async function ensureDataForSelection() {{
      const sido = document.getElementById('sidoSelect').value;
      if (sido === '전체') {{
        const allSidos = Object.keys(SIDO_DATA_FILES);
        await Promise.all(allSidos.map(loadSidoData));
        ACTIVE_DATA = allSidos.flatMap(name => window.SCHOOL_RISK_DATA[name] || []);
      }} else {{
        ACTIVE_DATA = await loadSidoData(sido);
      }}
      return ACTIVE_DATA;
    }}

    function markerHtml(color) {{
      return `<span style="display:block;width:13px;height:13px;border-radius:50%;background:${{color}};border:2px solid white;box-shadow:0 1px 4px rgba(0,0,0,.35);"></span>`;
    }}

    function valueOrDash(value, suffix = '') {{
      return value === null || value === undefined || Number.isNaN(value) ? '-' : `${{value}}${{suffix}}`;
    }}

    function selectedForecastModel() {{
      return document.getElementById('modelSelect')?.value || 'cohort';
    }}

    function updateModelHighlight() {{
      const model = selectedForecastModel();
      const isBase = model === 'pressure';
      const baseHeader = document.querySelector('.base-header');
      const tunedHeader = document.querySelector('.tuned-header');
      if (baseHeader && tunedHeader) {{
        baseHeader.textContent = isBase ? '베이스 (사용중)' : '베이스';
        tunedHeader.textContent = isBase ? '튜닝' : '튜닝/시나리오 (사용중)';
      }}
      document.querySelectorAll('.base-cell').forEach(el => el.classList.toggle('active', isBase));
      document.querySelectorAll('.tuned-cell').forEach(el => el.classList.toggle('active', !isBase));
    }}

    function modelView(d) {{
      const model = selectedForecastModel();
      if (model === 'change') {{
        const risk = d.changeModelRisk || d.risk;
        return {{
          forecast: d.changeModelForecastStudents ?? d.forecastStudents,
          pressure: d.changeModelPressure ?? d.pressure,
          risk,
          riskKo: RISK_LABELS[risk] || d.changeModelRiskKo || d.riskKo,
          color: RISK_COLORS[risk] || d.changeModelColor || d.color,
          score: d.changeModelRiskScore ?? d.riskScore,
          basisLabel: '변화량 모델 (인구이동 반영)',
        }};
      }}
      if (model === 'pressure') {{
        const risk = d.pressureModelRisk || d.risk;
        return {{
          forecast: d.pressureModelForecastStudents ?? d.forecastStudents,
          pressure: d.pressureModelPressure ?? d.pressure,
          risk,
          riskKo: RISK_LABELS[risk] || d.pressureModelRiskKo || d.riskKo,
          color: RISK_COLORS[risk] || d.pressureModelColor || d.color,
          score: d.pressureModelRiskScore ?? d.riskScore,
          basisLabel: '기존 시군구 압력비 기준',
        }};
      }}
      return {{
        forecast: d.forecastStudents,
        pressure: d.pressure,
        risk: d.risk,
        riskKo: d.riskKo,
        color: d.color,
        score: d.riskScore,
        basisLabel: '출생 코호트 기준',
      }};
    }}

    function selectedFeatures() {{
      const state = {{
        lowStudent: true,
        decline: true,
        isolation: true,
        commercial: true,
        regional: true,
        replacement: true,
      }};
      document.querySelectorAll('.panel .feature-toggle').forEach(el => state[el.dataset.feature] = el.checked);
      return state;
    }}

    function syncFeatureToggles(source) {{
      document.querySelectorAll(`.feature-toggle[data-feature="${{source.dataset.feature}}"]`).forEach(el => {{
        if (el !== source) el.checked = source.checked;
      }});
    }}

    function riskContributions(d) {{
      const features = selectedFeatures();
      const items = [
        {{ label: '저학생수 기준', active: features.lowStudent && d.flagLowStudent === 1, points: RISK_WEIGHTS.low_student }},
        {{ label: '장기 학령인구 감소', active: features.decline && d.flagLongDecline === 1, points: RISK_WEIGHTS.long_decline }},
        {{ label: '심각한 학령인구 감소', active: features.decline && d.flagSevereDecline === 1, points: RISK_WEIGHTS.severe_decline }},
        {{ label: '학교 고립도 높음', active: features.isolation && d.flagIsolationHigh === 1, points: RISK_WEIGHTS.isolation }},
        {{ label: '상권 취약도 높음', active: features.commercial && d.flagCommercialVulnerable === 1, points: RISK_WEIGHTS.commercial }},
        {{ label: '학령수요 감소압력 높음', active: features.regional && d.flagRegionalDeclineHigh === 1, points: RISK_WEIGHTS.regional }},
        {{ label: 'EDSS 유사도 상위 10%', active: d.flagObjectiveTop10 === 1, points: RISK_WEIGHTS.edss }},
        {{ label: '대체학교 가까움', active: features.replacement && d.flagReplacementNear === 1, points: RISK_WEIGHTS.replacement }},
      ];
      return items.map(item => ({{
        ...item,
        appliedPoints: item.active ? item.points : 0,
      }}));
    }}

    function contributionChips(d) {{
      const active = riskContributions(d).filter(item => item.active);
      if (!active.length) return '<span class="chip empty">가산 피처 없음</span>';
      return active.map(item => `<span class="chip">${{item.label}} +${{item.appliedPoints}}</span>`).join('');
    }}

    function metricTone(value, high = 70, mid = 45, reverse = false) {{
      const num = Number(value);
      if (!Number.isFinite(num)) return {{ cls: 'neutral', label: '확인' }};
      if (reverse) {{
        if (num <= high) return {{ cls: 'high', label: '위험' }};
        if (num <= mid) return {{ cls: 'mid', label: '주의' }};
        return {{ cls: 'low', label: '양호' }};
      }}
      if (num >= high) return {{ cls: 'high', label: '높음' }};
      if (num >= mid) return {{ cls: 'mid', label: '보통' }};
      return {{ cls: 'low', label: '낮음' }};
    }}

    function htmlMetric(label, value, suffix, hint, tone) {{
      const display = valueOrDash(value, suffix || '');
      return `<div class="metric ${{tone.cls}}">
        <div class="metric-top"><div class="label">${{label}}</div><span class="badge">${{tone.label}}</span></div>
        <div class="value">${{display}}</div>
        <div class="hint">${{hint}}</div>
      </div>`;
    }}

    function dynamicRisk(d) {{
      if (selectedForecastModel() === 'pressure' || selectedForecastModel() === 'change') {{
        const view = modelView(d);
        return {{ risk: view.risk, riskKo: view.riskKo, color: view.color, score: view.score }};
      }}
      if (d.risk === 'data_check_needed' || d.risk === 'special_school_review') {{
        return {{ risk: d.risk, riskKo: RISK_LABELS[d.risk], color: RISK_COLORS[d.risk], score: d.riskScore }};
      }}
      const contributions = riskContributions(d);
      const score = contributions.reduce((sum, item) => sum + item.appliedPoints, 0);
      const lowStudent = contributions[0].active;
      const isolation = contributions[3].active;
      const replacement = contributions[7].active;
      let risk = 'mid_risk';
      if (score < 35) risk = 'low_risk';
      else if (lowStudent && isolation && !replacement) risk = 'education_gap_high_risk';
      else if (lowStudent && replacement) risk = 'consolidation_high_risk';
      else if (score >= 60) risk = 'high_risk_review';
      return {{ risk, riskKo: RISK_LABELS[risk], color: RISK_COLORS[risk], score }};
    }}

    function popup(d) {{
      const dyn = dynamicRisk(d);
      const view = modelView(d);
      const delta = (d.students2025 && view.forecast !== null && view.forecast !== undefined)
        ? (((view.forecast / d.students2025) - 1) * 100).toFixed(1)
        : '-';
      const baseTone = metricTone(dyn.score, 60, 35);
      const diffTone = metricTone(d.diffScore, 70, 45);
      const isolationTone = metricTone(d.isolation, 70, 45);
      const commercialTone = metricTone(d.commercial, 70, 45);
      const regionalTone = metricTone(d.regionalDecline, 70, 45);
      const edssTone = metricTone(d.objectivePct, 90, 70);
      const nearestTone = metricTone(d.nearestKm, 3, 1.5);
      const sameTone = metricTone(d.same5km, 2, 5, true);
      return `<div class="popup">
        <h3>${{d.name}}</h3>
        <div class="meta">${{d.sido}} ${{d.sggName}} · ${{d.level}} · ${{d.year}}년</div>
        <div class="meta">행정코드 ${{d.sgg}}</div>

        <div class="risk-header">
          <div>
            <div class="risk-title">선택 피처 적용 위험등급</div>
            <div class="risk-label">${{dyn.riskKo}}</div>
          </div>
          <div class="score-pill" style="background:${{dyn.color}}">
            ${{dyn.score}}<small>점</small>
          </div>
        </div>

        <div class="summary-grid">
          <div class="summary-card">
            <div class="label">2025 학생수</div>
            <div class="value">${{valueOrDash(d.students2025, '명')}}</div>
          </div>
          <div class="summary-card">
            <div class="label">${{d.year}} 예측</div>
            <div class="value">${{valueOrDash(view.forecast, '명')}}</div>
            <div class="sub">2025 대비 ${{delta === '-' ? '-' : delta + '%'}}</div>
          </div>
          <div class="summary-card">
            <div class="label">학령인구 압력</div>
            <div class="value">${{valueOrDash(view.pressure)}}</div>
            <div class="sub">1보다 낮을수록 감소</div>
          </div>
        </div>

        <div class="section-title">점수에 반영된 피처</div>
        <div class="chips">${{contributionChips(d)}}</div>

        <div class="section-title">세부 지표</div>
        <div class="metric-grid">
          ${{htmlMetric('선택 피처 기준', dyn.score, '점', `${{dyn.riskKo}} · EDSS 고정 포함`, baseTone)}}
          ${{htmlMetric('차별 피처 점수', d.diffScore, '', '높을수록 고립/상권/지역 맥락 위험', diffTone)}}
          ${{htmlMetric('학교 고립도', d.isolation, '', '높을수록 주변 학교 접근성 낮음', isolationTone)}}
          ${{htmlMetric('상권 취약도', d.commercial, '', '높을수록 생활 기반 약함', commercialTone)}}
          ${{htmlMetric('학령수요 감소압력', d.regionalDecline, '', '전국 시군구 상대점수, 높을수록 감소 흐름 강함', regionalTone)}}
          ${{htmlMetric('EDSS 유사도', d.objectivePct, '%', '확률 아님, 전국 상대순위', edssTone)}}
          ${{htmlMetric('가까운 같은 학교급', d.nearestKm, 'km', '멀수록 교육공백 위험', nearestTone)}}
          ${{htmlMetric('5km 내 같은 학교급', d.same5km, '개', '적을수록 대체학교 부족', sameTone)}}
        </div>
        <div class="note">체크박스는 회귀 예측값이 아니라 위험점수 산식만 시뮬레이션합니다.</div>
      </div>`;
    }}

    function declineColor(changePct) {{
      if (changePct <= -30) return '#b91c1c';
      if (changePct <= -20) return '#f97316';
      if (changePct <= -10) return '#facc15';
      return '#22c55e';
    }}

    function declineLabel(changePct) {{
      if (changePct <= -30) return '급감 위험';
      if (changePct <= -20) return '큰 폭 감소';
      if (changePct <= -10) return '감소';
      return '완만/증가';
    }}

    function aggregateRegions(rows) {{
      const grouped = new Map();
      rows.forEach(d => {{
        const key = `${{d.sido}}-${{d.sgg}}`;
        if (!grouped.has(key)) {{
          grouped.set(key, {{
            sido: d.sido,
            sgg: d.sgg,
            sggName: d.sggName || d.sgg,
            latSum: 0,
            lonSum: 0,
            coordCount: 0,
            schools: 0,
            base: 0,
            forecast: 0,
            highRiskSchools: 0,
          }});
        }}
        const item = grouped.get(key);
        const view = modelView(d);
        item.schools += 1;
        item.base += Number(d.students2025 || 0);
        item.forecast += Number(view.forecast || 0);
        if (Number.isFinite(Number(d.lat)) && Number.isFinite(Number(d.lon))) {{
          item.latSum += Number(d.lat);
          item.lonSum += Number(d.lon);
          item.coordCount += 1;
        }}
        const dyn = dynamicRisk(d);
        if (dyn.risk === 'consolidation_high_risk' || dyn.risk === 'education_gap_high_risk' || dyn.risk === 'high_risk_review') {{
          item.highRiskSchools += 1;
        }}
      }});
      return [...grouped.values()].filter(item => item.coordCount > 0).map(item => {{
        const changePct = item.base > 0 ? ((item.forecast / item.base - 1) * 100) : 0;
        return {{
          ...item,
          lat: item.latSum / item.coordCount,
          lon: item.lonSum / item.coordCount,
          changePct,
          declineStudents: item.forecast - item.base,
        }};
      }});
    }}

    function regionPopup(r, year) {{
      const color = declineColor(r.changePct);
      return `<div class="popup">
        <h3>${{r.sido}} ${{r.sggName}}</h3>
        <div class="meta">행정코드 ${{r.sgg}}</div>
        <div class="meta">시군구 학생수 감소 요약 · ${{year}}년</div>
        <div class="risk-header">
          <div>
            <div class="risk-title">학생수 변화 등급</div>
            <div class="risk-label">${{declineLabel(r.changePct)}}</div>
          </div>
          <div class="score-pill" style="background:${{color}}">${{r.changePct.toFixed(1)}}<small>%</small></div>
        </div>
        <div class="summary-grid">
          <div class="summary-card"><div class="label">2025 학생수</div><div class="value">${{Math.round(r.base).toLocaleString()}}명</div></div>
          <div class="summary-card"><div class="label">${{year}} 예측</div><div class="value">${{Math.round(r.forecast).toLocaleString()}}명</div></div>
          <div class="summary-card"><div class="label">증감 인원</div><div class="value">${{Math.round(r.declineStudents).toLocaleString()}}명</div></div>
        </div>
        <div class="section-title">지역 내 학교</div>
        <div class="metric-grid">
          <div class="metric neutral"><div class="metric-top"><div class="label">표시 학교 수</div><span class="badge">학교</span></div><div class="value">${{r.schools.toLocaleString()}}개</div><div class="hint">현재 필터 기준</div></div>
          <div class="metric high"><div class="metric-top"><div class="label">고위험/검토 학교</div><span class="badge">위험</span></div><div class="value">${{r.highRiskSchools.toLocaleString()}}개</div><div class="hint">선택 피처 기준</div></div>
        </div>
        <div class="note">이 버블은 학교 위치의 시군구별 평균 좌표에 표시한 지역 요약입니다. 상세 학교는 표시단위를 학교별 점으로 바꾸면 볼 수 있습니다.</div>
      </div>`;
    }}

    function effectiveDisplayMode() {{
      const selected = document.getElementById('displayMode').value;
      if (selected !== 'auto') return selected;
      return map.getZoom() >= 11 ? 'school' : 'region';
    }}

    function updateLegend(displayMode) {{
      const riskLegend = document.getElementById('riskLegendBlock');
      const regionLegend = document.getElementById('regionLegendBlock');
      if (!riskLegend || !regionLegend) return;
      riskLegend.style.display = displayMode === 'school' ? 'block' : 'none';
      regionLegend.style.display = displayMode === 'region' ? 'block' : 'none';
    }}

    function filterData(year = null) {{
      const sido = document.getElementById('sidoSelect').value;
      const risk = document.getElementById('riskSelect').value;
      const level = document.getElementById('levelSelect').value;
      return ACTIVE_DATA.filter(d =>
        (year === null || d.year === year) &&
        (sido === '전체' || d.sido === sido) &&
        (risk === '전체' || dynamicRisk(d).risk === risk) &&
        (level === '전체' || d.level === level)
      );
    }}

    function aggregateByYear() {{
      const rows = filterData(null);
      const byYear = new Map();
      rows.forEach(d => {{
        if (!byYear.has(d.year)) byYear.set(d.year, {{ forecast: 0, base: 0 }});
        const item = byYear.get(d.year);
        item.forecast += Number(modelView(d).forecast || 0);
        item.base += Number(d.students2025 || 0);
      }});
      const years = [...new Set(ACTIVE_DATA.map(d => d.year))].sort((a, b) => a - b);
      return {{
        years,
        forecast: years.map(y => byYear.get(y)?.forecast ?? 0),
        base: years.map(y => byYear.get(y)?.base ?? 0),
      }};
    }}

    function aggregatePressureModelByYear() {{
      const rows = filterData(null);
      const byYear = new Map();
      rows.forEach(d => {{
        if (!byYear.has(d.year)) byYear.set(d.year, {{ forecast: 0, base: 0 }});
        const item = byYear.get(d.year);
        item.forecast += Number(d.pressureModelForecastStudents ?? d.forecastStudents ?? 0);
        item.base += Number(d.students2025 || 0);
      }});
      const years = [...new Set(ACTIVE_DATA.map(d => d.year))].sort((a, b) => a - b);
      return {{
        years,
        forecast: years.map(y => byYear.get(y)?.forecast ?? 0),
        base: years.map(y => byYear.get(y)?.base ?? 0),
      }};
    }}

    function aggregateThreeModelsByYear() {{
      const rows = filterData(null);
      const byYear = new Map();
      rows.forEach(d => {{
        if (!byYear.has(d.year)) {{
          byYear.set(d.year, {{ pressure: 0, change: 0, cohort: 0, base: 0 }});
        }}
        const item = byYear.get(d.year);
        item.pressure += Number(d.pressureModelForecastStudents ?? d.forecastStudents ?? 0);
        item.change += Number(d.changeModelForecastStudents ?? d.forecastStudents ?? 0);
        item.cohort += Number(d.forecastStudents ?? 0);
        item.base += Number(d.students2025 || 0);
      }});
      const years = [...new Set(ACTIVE_DATA.map(d => d.year))].sort((a, b) => a - b);
      return {{
        years,
        pressure: years.map(y => byYear.get(y)?.pressure ?? 0),
        change: years.map(y => byYear.get(y)?.change ?? 0),
        cohort: years.map(y => byYear.get(y)?.cohort ?? 0),
        base: years.map(y => byYear.get(y)?.base ?? 0),
      }};
    }}

    function aggregateRiskByYear() {{
      const rows = filterData(null);
      const years = [...new Set(ACTIVE_DATA.map(d => d.year))].sort((a, b) => a - b);
      const byYear = new Map(years.map(y => [y, Object.fromEntries(RISK_ORDER.map(r => [r, 0]))]));
      rows.forEach(d => {{
        const dyn = dynamicRisk(d);
        byYear.get(d.year)[dyn.risk] += 1;
      }});
      return {{ years, byYear }};
    }}

    function aggregateCohortByYear() {{
      const sido = document.getElementById('sidoSelect').value;
      const level = document.getElementById('levelSelect').value;
      const years = [...new Set(COHORT_AGG.map(d => d.forecast_year))].sort((a, b) => a - b);
      const scenarios = ['baseline', 'optimistic', 'pessimistic'];
      const byScenario = Object.fromEntries(
        scenarios.map(s => [s, new Map(years.map(y => [y, {{ forecast: 0, base: 0, schools: 0 }}]))])
      );
      COHORT_AGG.forEach(d => {{
        if (sido !== '전체' && d.requested_sido_name !== sido) return;
        if (level !== '전체' && d.school_level !== level) return;
        const scenario = d.cohort_scenario;
        if (!byScenario[scenario]) return;
        const item = byScenario[scenario].get(d.forecast_year);
        item.forecast += Number(d.forecastStudents || 0);
        item.base += Number(d.students2025 || 0);
        item.schools += Number(d.schools || 0);
      }});
      return {{
        years,
        baseline: years.map(y => byScenario.baseline.get(y)?.forecast ?? 0),
        optimistic: years.map(y => byScenario.optimistic.get(y)?.forecast ?? 0),
        pessimistic: years.map(y => byScenario.pessimistic.get(y)?.forecast ?? 0),
        base: years.map(y => byScenario.baseline.get(y)?.base ?? 0),
        schools: years.map(y => byScenario.baseline.get(y)?.schools ?? 0),
      }};
    }}

    function updateChart() {{
      const mode = document.getElementById('chartMode').value;
      document.getElementById('graphFeatureControls').style.display = mode === 'risk' ? 'grid' : 'none';
      const ctx = document.getElementById('studentChart');
      if (chart) chart.destroy();
      if (mode === 'risk') {{
        const agg = aggregateRiskByYear();
        chart = new Chart(ctx, {{
          type: 'line',
          data: {{
            labels: agg.years,
            datasets: RISK_ORDER.map(risk => ({{
              label: RISK_LABELS[risk],
              data: agg.years.map(year => agg.byYear.get(year)[risk]),
              borderColor: RISK_COLORS[risk],
              backgroundColor: RISK_COLORS[risk],
              tension: 0.2,
              pointRadius: 2,
            }})),
          }},
          options: {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{
              legend: {{ labels: {{ boxWidth: 10, font: {{ size: 11 }} }} }},
              tooltip: {{ callbacks: {{ label: ctx => `${{ctx.dataset.label}}: ${{Math.round(ctx.raw).toLocaleString()}}개교` }} }},
            }},
            scales: {{
              x: {{ ticks: {{ font: {{ size: 10 }} }} }},
              y: {{ ticks: {{ callback: value => Number(value).toLocaleString(), font: {{ size: 10 }} }} }},
            }},
          }},
        }});
        const first = agg.years[0];
        const last = agg.years[agg.years.length - 1];
        const lastCounts = agg.byYear.get(last);
        const firstCounts = agg.byYear.get(first);
        document.getElementById('graphStats').innerHTML =
          `선택 피처 기준 ${{first}}년 고위험군: <b>${{((firstCounts.consolidation_high_risk || 0) + (firstCounts.education_gap_high_risk || 0)).toLocaleString()}}개교</b><br>` +
          `선택 피처 기준 ${{last}}년 고위험군: <b>${{((lastCounts.consolidation_high_risk || 0) + (lastCounts.education_gap_high_risk || 0)).toLocaleString()}}개교</b><br>` +
          `체크박스를 바꾸면 위험점수 산식 기여분이 빠져 위험등급별 학교 수가 다시 계산됩니다.`;
        return agg;
      }}
      if (mode === 'threeModel') {{
        const agg = aggregateThreeModelsByYear();
        chart = new Chart(ctx, {{
          type: 'line',
          data: {{
            labels: agg.years,
            datasets: [
              {{
                label: '절대값/압력비 모델',
                data: agg.pressure,
                borderColor: '#64748b',
                backgroundColor: 'rgba(100,116,139,.08)',
                fill: false,
                tension: 0.25,
                pointRadius: 2,
              }},
              {{
                label: '변화량 모델(인구이동 반영)',
                data: agg.change,
                borderColor: '#f97316',
                backgroundColor: 'rgba(249,115,22,.08)',
                fill: false,
                tension: 0.25,
                pointRadius: 2,
              }},
              {{
                label: '출생 코호트 기준',
                data: agg.cohort,
                borderColor: '#dc2626',
                backgroundColor: 'rgba(220,38,38,.08)',
                fill: false,
                tension: 0.25,
                pointRadius: 2,
              }},
              {{
                label: '2025 기준 학생수',
                data: agg.base,
                borderColor: '#94a3b8',
                borderDash: [2, 4],
                fill: false,
                pointRadius: 0,
              }},
            ],
          }},
          options: {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{
              legend: {{ labels: {{ boxWidth: 10, font: {{ size: 11 }} }} }},
              tooltip: {{ callbacks: {{ label: ctx => `${{ctx.dataset.label}}: ${{Math.round(ctx.raw).toLocaleString()}}명` }} }},
            }},
            scales: {{
              x: {{ ticks: {{ font: {{ size: 10 }} }} }},
              y: {{ ticks: {{ callback: value => Number(value).toLocaleString(), font: {{ size: 10 }} }} }},
            }},
          }},
        }});
        const first = agg.base[0] || 0;
        const lastIdx = agg.years.length - 1;
        const pct = value => first > 0 ? ((value / first - 1) * 100).toFixed(1) : '0.0';
        document.getElementById('graphStats').innerHTML =
          `위험등급 필터와 피처 체크박스는 제외하고 지역/학교급 필터만 적용한 학생수 예측 비교입니다.<br>` +
          `2040 기준 — 압력비: <b>${{Math.round(agg.pressure[lastIdx]).toLocaleString()}}명</b> (${{pct(agg.pressure[lastIdx])}}%) | ` +
          `변화량: <b>${{Math.round(agg.change[lastIdx]).toLocaleString()}}명</b> (${{pct(agg.change[lastIdx])}}%) | ` +
          `코호트: <b>${{Math.round(agg.cohort[lastIdx]).toLocaleString()}}명</b> (${{pct(agg.cohort[lastIdx])}}%)`;
        return agg;
      }}
      if (mode === 'cohort') {{
        const pressureAgg = aggregatePressureModelByYear();
        const cohortAgg = aggregateCohortByYear();
        chart = new Chart(ctx, {{
          type: 'line',
          data: {{
            labels: cohortAgg.years,
            datasets: [
              {{
                label: '기존 학교별 압력비 예측',
                data: pressureAgg.forecast,
                borderColor: '#2563eb',
                backgroundColor: 'rgba(37,99,235,.08)',
                fill: false,
                tension: 0.25,
                pointRadius: 2,
              }},
              {{
                label: '출생 코호트 기준',
                data: cohortAgg.baseline,
                borderColor: '#dc2626',
                backgroundColor: 'rgba(220,38,38,.08)',
                fill: false,
                tension: 0.25,
                pointRadius: 2,
              }},
              {{
                label: '코호트 낙관',
                data: cohortAgg.optimistic,
                borderColor: '#16a34a',
                borderDash: [5, 4],
                fill: false,
                tension: 0.25,
                pointRadius: 1,
              }},
              {{
                label: '코호트 비관',
                data: cohortAgg.pessimistic,
                borderColor: '#f97316',
                borderDash: [5, 4],
                fill: false,
                tension: 0.25,
                pointRadius: 1,
              }},
              {{
                label: '2025 기준 학생수',
                data: cohortAgg.base,
                borderColor: '#94a3b8',
                borderDash: [2, 4],
                fill: false,
                pointRadius: 0,
              }},
            ],
          }},
          options: {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{
              legend: {{ labels: {{ boxWidth: 10, font: {{ size: 11 }} }} }},
              tooltip: {{ callbacks: {{ label: ctx => `${{ctx.dataset.label}}: ${{Math.round(ctx.raw).toLocaleString()}}명` }} }},
            }},
            scales: {{
              x: {{ ticks: {{ font: {{ size: 10 }} }} }},
              y: {{ ticks: {{ callback: value => Number(value).toLocaleString(), font: {{ size: 10 }} }} }},
            }},
          }},
        }});
        const first = cohortAgg.base[0] || 0;
        const pressureLast = pressureAgg.forecast[pressureAgg.forecast.length - 1] || 0;
        const cohortLast = cohortAgg.baseline[cohortAgg.baseline.length - 1] || 0;
        const optLast = cohortAgg.optimistic[cohortAgg.optimistic.length - 1] || 0;
        const pessLast = cohortAgg.pessimistic[cohortAgg.pessimistic.length - 1] || 0;
        const pct = value => first > 0 ? ((value / first - 1) * 100).toFixed(1) : '0.0';
        document.getElementById('graphStats').innerHTML =
          `이 그래프는 지역/학교급 필터만 적용하고, 위험등급 필터와 체크박스는 적용하지 않습니다.<br>` +
          `2025 기준 학생수: <b>${{Math.round(first).toLocaleString()}}명</b><br>` +
          `2040 기존 압력비 예측: <b>${{Math.round(pressureLast).toLocaleString()}}명</b> (${{pct(pressureLast)}}%)<br>` +
          `2040 출생 코호트 기준: <b>${{Math.round(cohortLast).toLocaleString()}}명</b> (${{pct(cohortLast)}}%)<br>` +
          `2040 코호트 낙관/비관: <b>${{Math.round(optLast).toLocaleString()}}명</b> (${{pct(optLast)}}%) / <b>${{Math.round(pessLast).toLocaleString()}}명</b> (${{pct(pessLast)}}%)`;
        return cohortAgg;
      }}
      const agg = aggregateByYear();
      chart = new Chart(ctx, {{
        type: 'line',
        data: {{
          labels: agg.years,
          datasets: [
            {{
              label: '예측 학생수 합계',
              data: agg.forecast,
              borderColor: '#2563eb',
              backgroundColor: 'rgba(37,99,235,.12)',
              fill: true,
              tension: 0.25,
              pointRadius: 2,
            }},
            {{
              label: '2025 기준 학생수 합계',
              data: agg.base,
              borderColor: '#94a3b8',
              borderDash: [4, 4],
              fill: false,
              pointRadius: 0,
            }},
          ],
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          plugins: {{
            legend: {{ labels: {{ boxWidth: 10, font: {{ size: 11 }} }} }},
            tooltip: {{ callbacks: {{ label: ctx => `${{ctx.dataset.label}}: ${{Math.round(ctx.raw).toLocaleString()}}명` }} }},
          }},
          scales: {{
            x: {{ ticks: {{ font: {{ size: 10 }} }} }},
            y: {{ ticks: {{ callback: value => Number(value).toLocaleString(), font: {{ size: 10 }} }} }},
          }},
        }},
      }});
      const first = agg.base[0] || 0;
      const last = agg.forecast[agg.forecast.length - 1] || 0;
      const delta = first > 0 ? ((last / first - 1) * 100) : 0;
      document.getElementById('graphStats').innerHTML =
        `2025 기준 학생수 합계: <b>${{Math.round(first).toLocaleString()}}명</b><br>` +
        `2040 예측 학생수 합계: <b>${{Math.round(last).toLocaleString()}}명</b><br>` +
        `2025 대비 2040 변화율: <b>${{delta.toFixed(1)}}%</b>`;
      return agg;
    }}

    function openTab(tab) {{
      const isGraph = tab === 'graph';
      document.getElementById('graphPanel').style.display = isGraph ? 'block' : 'none';
      document.getElementById('mapTab').classList.toggle('active', !isGraph);
      document.getElementById('graphTab').classList.toggle('active', isGraph);
      if (isGraph) updateChart();
      if (!isGraph) setTimeout(() => map.invalidateSize(), 50);
    }}

    async function render() {{
      document.getElementById('stats').innerHTML = '데이터 불러오는 중...';
      await ensureDataForSelection();
      const year = Number(document.getElementById('yearSelect').value);
      const displayMode = effectiveDisplayMode();
      updateLegend(displayMode);
      const filtered = filterData(year);
      cluster.clearLayers();
      regionLayer.clearLayers();
      const counts = {{}};
      let forecastTotal = 0;
      let baseTotal = 0;
      filtered.forEach(d => {{
        const dyn = dynamicRisk(d);
        const view = modelView(d);
        counts[dyn.riskKo] = (counts[dyn.riskKo] || 0) + 1;
        forecastTotal += Number(view.forecast || 0);
        baseTotal += Number(d.students2025 || 0);
      }});

      if (displayMode === 'region') {{
        const regions = aggregateRegions(filtered).sort((a, b) => a.changePct - b.changePct);
        regions.forEach(r => {{
          const color = declineColor(r.changePct);
          const radius = Math.max(8, Math.min(34, 8 + Math.sqrt(Math.abs(r.declineStudents)) * 0.12));
          L.circleMarker([r.lat, r.lon], {{
            radius,
            color: '#ffffff',
            weight: 2,
            fillColor: color,
            fillOpacity: 0.78,
          }}).bindPopup(regionPopup(r, year), {{ maxWidth: 430 }})
            .bindTooltip(`${{r.sido}} ${{r.sggName}} · ${{r.changePct.toFixed(1)}}%`)
            .addTo(regionLayer);
        }});
      }} else {{
        filtered.forEach(d => {{
          const dyn = dynamicRisk(d);
          const icon = L.divIcon({{ html: markerHtml(dyn.color), className: '', iconSize: [17,17], iconAnchor: [8,8] }});
          L.marker([d.lat, d.lon], {{ icon }}).bindPopup(popup(d), {{ maxWidth: 430 }}).bindTooltip(`${{d.name}} | ${{dyn.riskKo}}`).addTo(cluster);
        }});
      }}
      const agg = aggregateByYear();
      const changeRate = baseTotal > 0 ? ((forecastTotal / baseTotal - 1) * 100) : 0;
      const countText = Object.entries(counts).map(([key, value]) => `${{key}} ${{value.toLocaleString()}}`).join(' · ');
      const rawMode = document.getElementById('displayMode').value;
      const modeText = rawMode === 'auto'
        ? `자동 전환: ${{displayMode === 'region' ? '시군구 감소 버블' : '학교별 점'}}`
        : (displayMode === 'region' ? '시군구 감소 버블' : '학교별 점');
      const modelText = selectedForecastModel() === 'pressure'
        ? '베이스 모델: 시군구 압력비 기준'
        : (selectedForecastModel() === 'change' ? '변화량 모델: 인구이동 반영' : '튜닝 모델: 출생 코호트 기준');
      document.getElementById('stats').innerHTML =
        `<div class="stat-meta">${{modelText}} · ${{modeText}}</div>` +
        `<div class="stat-cards">` +
          `<div class="stat-card"><div class="stat-value">${{filtered.length.toLocaleString()}}</div><div class="stat-label">표시 학교</div></div>` +
          `<div class="stat-card accent"><div class="stat-value">${{Math.round(forecastTotal).toLocaleString()}}</div><div class="stat-label">예측 학생수</div></div>` +
          `<div class="stat-card ${{changeRate <= -20 ? 'danger' : 'safe'}}"><div class="stat-value">${{changeRate.toFixed(1)}}%</div><div class="stat-label">2025 대비</div></div>` +
        `</div>` +
        `<div class="risk-summary">2025 기준 학생수: ${{Math.round(baseTotal).toLocaleString()}}명<br>${{countText || '조건에 맞는 학교 없음'}}</div>`;
      if (document.getElementById('graphPanel').style.display === 'block') updateChart();
    }}

    ['yearSelect','sidoSelect','modelSelect','riskSelect','levelSelect','displayMode'].forEach(id => document.getElementById(id).addEventListener('change', () => {{
      updateModelHighlight();
      render();
    }}));
    document.querySelectorAll('.feature-toggle').forEach(el => el.addEventListener('change', event => {{
      syncFeatureToggles(event.target);
      render();
    }}));
    document.getElementById('chartMode').addEventListener('change', updateChart);
    document.getElementById('mapTab').addEventListener('click', () => openTab('map'));
    document.getElementById('graphTab').addEventListener('click', () => openTab('graph'));
    map.on('zoomend', () => {{
      if (document.getElementById('displayMode').value === 'auto') render();
    }});
    updateModelHighlight();
    render();
  </script>
</body>
</html>
"""

    output = MAPS / "final_national_interactive_school_risk_scenario.html"
    output.write_text(html, encoding="utf-8")
    print("saved:", output)
    print("initial embedded region:", default_sido)
    print("initial embedded rows:", len(df[df["requested_sido_name"].eq(default_sido)]))
    print("lazy-load total rows:", len(df))
    print("coordinate report:", REPORTS / "final_coordinate_quality_report.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

