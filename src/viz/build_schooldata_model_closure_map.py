from __future__ import annotations

import json
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd


ROOT = SRC.parent
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "outputs" / "reports"
MAPS = ROOT / "outputs" / "maps"

CURRENT_PATH = PROCESSED / "schooldata_current_closure_risk_2025.csv"
PREDICTIONS_PATH = REPORTS / "schooldata_closure_classifier_predictions.csv"
METRICS_PATH = REPORTS / "schooldata_closure_classifier_metrics.csv"
TOPK_PATH = REPORTS / "schooldata_closure_classifier_topk_metrics.csv"
POP_FORECAST_PATH = PROCESSED / "final_national_sgg_population_forecast_2026_2040.csv"
POP_CHANGE_FORECAST_PATH = PROCESSED / "final_national_sgg_population_forecast_change_model_2026_2040.csv"
DASHBOARD_PATH = MAPS / "school_project_dashboard.html"

SIDO_CENTERS = {
    "서울": (37.5665, 126.9780),
    "부산": (35.1796, 129.0756),
    "대구": (35.8714, 128.6014),
    "인천": (37.4563, 126.7052),
    "광주": (35.1595, 126.8526),
    "대전": (36.3504, 127.3845),
    "울산": (35.5384, 129.3114),
    "세종": (36.4800, 127.2890),
    "경기": (37.4138, 127.5183),
    "강원": (37.8228, 128.1555),
    "충북": (36.8000, 127.7000),
    "충남": (36.5184, 126.8000),
    "전북": (35.7175, 127.1530),
    "전남": (34.8679, 126.9910),
    "경북": (36.4919, 128.8889),
    "경남": (35.4606, 128.2132),
    "제주": (33.4996, 126.5312),
}

SIDO_CODE_TO_NAME = {
    "11": "서울",
    "21": "부산",
    "22": "대구",
    "23": "인천",
    "24": "광주",
    "25": "대전",
    "26": "울산",
    "29": "세종",
    "31": "경기",
    "32": "강원",
    "33": "충북",
    "34": "충남",
    "35": "전북",
    "36": "전남",
    "37": "경북",
    "38": "경남",
    "39": "제주",
    "41": "경기",
    "42": "강원",
    "43": "충북",
    "44": "충남",
    "45": "전북",
    "46": "전남",
    "47": "경북",
    "48": "경남",
    "50": "제주",
    "51": "강원",
    "52": "전북",
}


def clean_float(value: object, digits: int = 4) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), digits)


def clean_int(value: object) -> int | None:
    if pd.isna(value):
        return None
    return int(round(float(value)))


def prob_color(prob: float) -> str:
    if prob >= 0.15:
        return "#dc2626"
    if prob >= 0.05:
        return "#f97316"
    if prob >= 0.01:
        return "#facc15"
    return "#22c55e"


def decline_color(rate: float) -> str:
    if rate <= -30:
        return "#b91c1c"
    if rate <= -20:
        return "#f97316"
    if rate <= -10:
        return "#facc15"
    return "#22c55e"


def model_id(model_key: str) -> str:
    return {
        "histgb": "tuned_histgb_schooldata_closure",
        "logistic": "base_logistic_schooldata_closure",
    }[model_key]


def load_current() -> pd.DataFrame:
    df = pd.read_csv(CURRENT_PATH, low_memory=False)
    df = df[df["lttud"].notna() & df["lgtud"].notna()].copy()
    
    # sgg_code 복구를 위해 인구 예측 데이터의 시군구 코드 매핑 테이블 구축
    try:
        pop_df = pd.read_csv(POP_FORECAST_PATH, low_memory=False)
        pop_df["sido_name"] = pop_df["sido_code"].astype(str).str.zfill(2).map(SIDO_CODE_TO_NAME).fillna("")
        sgg_map = {}
        for _, row in pop_df.iterrows():
            sido = str(row["sido_name"]).strip()
            sgg = str(row["sgg_name"]).strip()
            code = str(row["sgg_code"]).zfill(5)
            if sido and sgg:
                sgg_map[(sido, sgg)] = code
        
        sgg_codes = []
        for _, row in df.iterrows():
            sido = str(row.get("시도", "")).strip()
            sgg = str(row.get("행정구", "")).strip()
            code = sgg_map.get((sido, sgg), None)
            if code is None:
                # 특별 지자체나 표기 불일치 예외 처리용 부분 매칭 시도
                matched = False
                for (s_name, sg_name), s_code in sgg_map.items():
                    if s_name == sido and (sg_name in sgg or sgg in sg_name):
                        sgg_codes.append(s_code)
                        matched = True
                        break
                if not matched:
                    # 최종 매칭 실패 시 빈 문자열
                    sgg_codes.append("")
            else:
                sgg_codes.append(code)
        df["sgg_code"] = sgg_codes
    except Exception as e:
        print(f"Warning: Failed to recover sgg_code using POP_FORECAST_PATH: {e}")
        df["sgg_code"] = df["school_key"].astype(str).str.extract(r"(\d{5})", expand=False)
        
    df["join_sido"] = df["시도"].astype(str).str.strip()
    df["join_sgg"] = df["행정구"].astype(str).str.strip()
    df["student_count"] = pd.to_numeric(df["student_count"], errors="coerce").fillna(0)
    return df


def build_current_payload(df: pd.DataFrame) -> list[dict[str, object]]:
    records = []
    for _, row in df.iterrows():
        histgb = float(row.get("tuned_histgb_schooldata_closure_probability", 0) or 0)
        logistic = float(row.get("base_logistic_schooldata_closure_probability", 0) or 0)
        records.append(
            {
                "name": str(row["학교명"]),
                "sido": str(row["시도"]),
                "sgg": str(row["행정구"]),
                "sggCode": str(row.get("sgg_code", "")),
                "level": str(row["학교급"]),
                "branch": str(row.get("본분교", "")),
                "student": clean_int(row.get("student_count")),
                "lat": clean_float(row.get("lttud"), 6),
                "lon": clean_float(row.get("lgtud"), 6),
                "histgb": round(histgb, 6),
                "logistic": round(logistic, 6),
                "feasibleHistgb": round(float(row.get("tuned_histgb_schooldata_closure_feasibility_probability", histgb) or histgb), 6),
                "feasibleLogistic": round(float(row.get("base_logistic_schooldata_closure_feasibility_probability", logistic) or logistic), 6),
                "eligible": int(row.get("candidate_eligible", 1) or 0),
                "candidateReason": str(row.get("candidate_reason", "")),
                "policyCategory": str(row.get("final_policy_category", "")),
                "rank": clean_int(row.get("risk_rank")),
                "riskGroup": str(row.get("model_risk_group", "")),
                "nearestKm": clean_float(row.get("nearest_same_level_school_km"), 2),
                "same5km": clean_int(row.get("same_level_school_count_5km")),
                "isolation": clean_float(row.get("school_isolation_score"), 1),
                "replacement": clean_float(row.get("replacement_available_score"), 1),
                "protection": int(row.get("isolation_protection_flag", 0) or 0),
                "shops1km": clean_int(row.get("radius_1_0km_all_shops")),
            }
        )
    return records


def build_test_payload() -> list[dict[str, object]]:
    pred = pd.read_csv(PREDICTIONS_PATH, low_memory=False)
    pred = pred[pred["lttud"].notna() & pred["lgtud"].notna()].copy()
    pred["modelKey"] = pred["model"].map(
        {
            "tuned_histgb_schooldata_closure": "histgb",
            "base_logistic_schooldata_closure": "logistic",
        }
    )
    pred = pred[pred["modelKey"].notna()].copy()
    rows = []
    for _, row in pred.iterrows():
        rows.append(
            {
                "year": int(row["year"]),
                "model": row["modelKey"],
                "name": str(row["학교명"]),
                "sido": str(row["시도"]),
                "sgg": str(row["행정구"]),
                "level": str(row["학교급"]),
                "student": clean_int(row.get("student_count")),
                "lat": clean_float(row.get("lttud"), 6),
                "lon": clean_float(row.get("lgtud"), 6),
                "actual": int(row.get("closure_within_3yr_label", 0) or 0),
                "prob": round(float(row.get("probability", 0) or 0), 6),
                "prediction": int(row.get("prediction", 0) or 0),
                "rank": clean_int(row.get("rank_in_test")),
                "candidateReason": str(row.get("candidate_reason", "")),
            }
        )
    return rows


def build_scenario_payload(current: pd.DataFrame) -> list[dict[str, object]]:
    raise NotImplementedError("Scenario rows are computed in browser from current schools and population ratios.")


def _scenario_record(row: pd.Series, ratio: float, model: str, year: int) -> dict[str, object]:
    base_student = float(row.get("student_count", 0) or 0)
    forecast_student = max(0, round(base_student * ratio))
    decline_pct = (ratio - 1) * 100
    base_prob = float(row.get("tuned_histgb_schooldata_closure_probability", 0) or 0)
    logistic_prob = float(row.get("base_logistic_schooldata_closure_probability", 0) or 0)
    level = str(row["학교급"])
    low_threshold = 80 if "초" in level else 330
    low_bonus = max(0, (low_threshold - forecast_student) / max(low_threshold, 1)) * 0.18
    decline_bonus = max(0, -decline_pct) / 100 * 0.18
    future_prob = max(0, min(0.95, base_prob + low_bonus + decline_bonus))
    future_logistic = max(0, min(0.95, logistic_prob + low_bonus + decline_bonus))
    return {
        "year": year,
        "model": model,
        "name": str(row["학교명"]),
        "sido": str(row["시도"]),
        "sgg": str(row["행정구"]),
        "sggCode": str(row.get("sgg_code", "")),
        "level": level,
        "lat": clean_float(row.get("lttud"), 6),
        "lon": clean_float(row.get("lgtud"), 6),
        "student2025": clean_int(base_student),
        "forecastStudent": clean_int(forecast_student),
        "ratio": round(ratio, 4),
        "declinePct": round(decline_pct, 2),
        "futureHistgb": round(future_prob, 6),
        "futureLogistic": round(future_logistic, 6),
        "isolation": clean_float(row.get("school_isolation_score"), 1),
        "same5km": clean_int(row.get("same_level_school_count_5km")),
        "policyCategory": str(row.get("final_policy_category", "")),
    }


def build_population_payload() -> list[dict[str, object]]:
    pop = pd.read_csv(POP_FORECAST_PATH, low_memory=False)
    change_pop = pd.read_csv(POP_CHANGE_FORECAST_PATH, low_memory=False)
    
    # 상위 통합시와 하위 구의 이중 계상(중복) 제거하여 실제 학령인구 정합성 복구
    exclude_sgg = [41110, 41130, 41170, 41270, 41280, 41460, 43110, 44130, 45110, 47110, 48120]
    pop = pop[~pop["sgg_code"].isin(exclude_sgg)]
    change_pop = change_pop[~change_pop["sgg_code"].isin(exclude_sgg)]
    
    # 영유아 및 유치원 연령대(만 0세~5세) 제외 스케일링 보정
    # 2026년 기준 0~19세 전체 청소년(770만 명) 중 만 6~18세 초중고 재학생 비중(약 62.66%) 반영
    SCALE_FACTOR = 0.6266
    pop["forecast_school_age_pop_0_19"] = pop["forecast_school_age_pop_0_19"] * SCALE_FACTOR
    change_pop["forecast_school_age_pop_0_19"] = change_pop["forecast_school_age_pop_0_19"] * SCALE_FACTOR
    
    pop["model"] = "ridge"
    change_pop["model"] = "rf"
    df = pd.concat([pop, change_pop], ignore_index=True)
    df["sgg_code"] = df["sgg_code"].astype(str).str.zfill(5)
    df["sido"] = df["sido_code"].astype(str).str.zfill(2).map(SIDO_CODE_TO_NAME).fillna("")
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "model": row["model"],
                "year": int(row["forecast_year"]),
                "sido": str(row["sido"]),
                "sgg": str(row["sgg_name"]),
                "sggCode": str(row["sgg_code"]),
                "population": round(float(row["forecast_school_age_pop_0_19"]), 1),
                "ratio": round(float(row["population_pressure_ratio"]), 4),
            }
        )
    return rows


def metric_payload() -> dict[str, object]:
    metrics = pd.read_csv(METRICS_PATH)
    topk = pd.read_csv(TOPK_PATH)
    metric_rows = []
    for _, row in metrics.iterrows():
        metric_rows.append(
            {
                "model": row["model"],
                "precision": round(float(row["precision"]), 3),
                "recall": round(float(row["recall"]), 3),
                "f1": round(float(row["f1"]), 3),
                "rocAuc": round(float(row["roc_auc"]), 3),
                "prAuc": round(float(row["pr_auc"]), 3),
            }
        )
    topk_rows = []
    for _, row in topk.iterrows():
        if str(row.get("segment", "all")) == "all":
            topk_rows.append(
                {
                    "model": row["model"],
                    "k": int(row["k"]),
                    "hits": int(row["hits_at_k"]),
                    "precision": round(float(row["precision_at_k"]), 3),
                    "recall": round(float(row["recall_at_k"]), 3),
                    "lift": round(float(row["lift_at_k"]), 1),
                }
            )
    return {"metrics": metric_rows, "topk": topk_rows}


def make_html(payload: dict[str, object]) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    html = r"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>학교 통폐합 위험 분석 대시보드</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    html, body { height:100%; margin:0; font-family:Arial,'Malgun Gothic',sans-serif; color:#172033; }
    #map { position:fixed; inset:0; background:#eef3f5; }
    .nav { position:fixed; top:14px; left:52px; z-index:1200; display:flex; gap:6px; background:#fff; border:1px solid #d9e2ec; border-radius:7px; padding:6px; box-shadow:0 5px 18px rgba(15,23,42,.14); }
    .nav button { border:0; border-radius:5px; padding:8px 12px; background:#fff; color:#334155; font-weight:850; cursor:pointer; }
    .nav button.active { background:#2563eb; color:white; }
    .panel { position:fixed; top:58px; right:18px; width:430px; max-height:calc(100vh - 78px); overflow:auto; z-index:1100; background:white; border:1px solid #d9e2ec; border-radius:8px; box-shadow:0 10px 30px rgba(15,23,42,.16); padding:16px; }
    h1 { font-size:18px; margin:0 0 5px; }
    .sub { font-size:12px; color:#64748b; line-height:1.45; margin-bottom:12px; }
    .section { border-top:1px solid #e2e8f0; margin-top:12px; padding-top:12px; }
    .section:first-of-type { border-top:0; margin-top:0; padding-top:0; }
    .title { font-size:12px; font-weight:900; color:#334155; margin-bottom:8px; }
    .controls { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
    label { display:block; font-size:12px; color:#64748b; margin:0 0 4px; }
    select { width:100%; padding:8px 9px; border:1px solid #cbd5e1; border-radius:5px; background:white; font-size:13px; }
    .stats { display:grid; grid-template-columns:repeat(3,1fr); gap:8px; }
    .stat { border:1px solid #e2e8f0; border-radius:7px; padding:9px; background:#fbfdff; }
    .stat .k { font-size:11px; color:#64748b; }
    .stat .v { margin-top:3px; font-size:17px; font-weight:900; color:#0f172a; }
    .stat.blue { background:#eff6ff; border-color:#bfdbfe; }
    .stat.red { background:#fff7f7; border-color:#fecaca; }
    .legend { position:fixed; left:18px; bottom:18px; z-index:1100; background:white; border:1px solid #d9e2ec; border-radius:8px; box-shadow:0 8px 22px rgba(15,23,42,.14); padding:12px; font-size:12px; }
    .legend b { display:block; margin-bottom:6px; }
    .dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; border:1px solid white; }
    .popup { min-width:330px; font-family:Arial,'Malgun Gothic',sans-serif; }
    .popup h2 { margin:0; font-size:20px; }
    .popup .meta { color:#64748b; font-size:12px; margin:4px 0 10px; }
    .badge { display:inline-block; color:white; border-radius:20px; padding:6px 10px; font-weight:900; }
    .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:10px; }
    .mini { border:1px solid #e2e8f0; border-radius:7px; padding:8px; }
    .mini .k { color:#64748b; font-size:11px; }
    .mini .v { font-size:15px; font-weight:850; margin-top:2px; }
    table { width:100%; border-collapse:collapse; font-size:11px; }
    th,td { padding:5px 3px; border-bottom:1px solid #e2e8f0; text-align:right; }
    th:first-child,td:first-child { text-align:left; }
    #chartWrap { display:none; position:fixed; left:60px; right:470px; bottom:24px; height:260px; z-index:1000; background:white; border:1px solid #d9e2ec; border-radius:8px; box-shadow:0 10px 30px rgba(15,23,42,.16); padding:14px; }
    #chartWrap.show { display:block; }
    #chartWrap.full { display:block; top:80px; bottom:24px; left:18px; right:470px; height:auto; }
    canvas { width:100% !important; height:100% !important; }
    .note { color:#64748b; font-size:11px; line-height:1.45; margin-top:8px; }
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="nav">
    <button data-tab="test">모델 테스트</button>
    <button data-tab="scenario" class="active">2025~2040 시나리오</button>
    <button data-tab="population">인구 회귀</button>
  </div>
  <aside class="panel">
    <h1 id="panelTitle">2025~2040 시나리오</h1>
    <div class="sub" id="panelSub"></div>
    <div class="section">
      <div class="title">필터</div>
      <div class="controls" id="controls"></div>
    </div>
    <div class="section">
      <div class="title">요약</div>
      <div class="stats" id="stats"></div>
      <div class="note" id="note"></div>
    </div>
    <div class="section" id="tableSection">
      <div class="title" id="tableTitle">검증표</div>
      <div id="tableBox"></div>
    </div>
  </aside>
  <div class="legend" id="legend"></div>
  <div id="chartWrap"><canvas id="chart"></canvas></div>
  <script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <script>
    const PAYLOAD = __PAYLOAD__;
    const KOREA_CENTER = [36.35, 127.85];
    const KOREA_BOUNDS = [[32.8, 124.0], [39.7, 132.2]];
    const map = L.map('map', { preferCanvas:false, maxBounds: KOREA_BOUNDS, maxBoundsViscosity: 0.85 }).setView(KOREA_CENTER, 7);
    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', { maxZoom:19, attribution:'&copy; OpenStreetMap &copy; CARTO' }).addTo(map);
    let layer = L.layerGroup().addTo(map);
    let tab = 'scenario';
    let chart = null;

    const modelIds = { histgb:'tuned_histgb_schooldata_closure', logistic:'base_logistic_schooldata_closure' };
    const modelNames = { histgb:'튜닝 HistGB', logistic:'베이스 Logistic', rf:'튜닝 RandomForest', ridge:'베이스 Ridge' };
    const probColor = p => p >= .15 ? '#dc2626' : p >= .05 ? '#f97316' : p >= .01 ? '#facc15' : '#22c55e';
    const declineColor = r => r <= -30 ? '#b91c1c' : r <= -20 ? '#f97316' : r <= -10 ? '#facc15' : '#22c55e';
    const fmt = v => v === null || v === undefined || Number.isNaN(v) ? '-' : Number(v).toLocaleString();
    const pct = v => `${(Number(v || 0)*100).toFixed(2)}%`;
    const rawPct = v => `${Number(v || 0).toFixed(1)}%`;
    const uniq = arr => [...new Set(arr)].sort((a,b)=>String(a).localeCompare(String(b),'ko'));
    const centerOf = rows => {
      const lat = rows.reduce((s,d)=>s+Number(d.lat||0),0)/Math.max(rows.length,1);
      const lon = rows.reduce((s,d)=>s+Number(d.lon||0),0)/Math.max(rows.length,1);
      return [lat, lon];
    };
    const ratioLookup = new Map();
    for (const r of PAYLOAD.population) {
      ratioLookup.set(`${r.model}|${r.year}|${r.sido}|${r.sgg}`, r.ratio);
    }
    function selectHtml(id, label, options, value) {
      return `<div><label>${label}</label><select id="${id}">${options.map(o=>`<option value="${o.value}" ${o.value===value?'selected':''}>${o.label}</option>`).join('')}</select></div>`;
    }
    function bindControls() {
      document.querySelectorAll('#controls select').forEach(el => el.addEventListener('change', render));
    }
    function setStats(items) {
      document.getElementById('stats').innerHTML = items.map((it,i)=>`<div class="stat ${it.tone||''}"><div class="k">${it.k}</div><div class="v">${it.v}</div></div>`).join('');
    }
    function fit(rows, maxZoom=10) {
      if (!rows.length) return;
      const bounds = rows.map(d=>[d.lat,d.lon]).filter(p=>p[0]&&p[1]&&!Number.isNaN(p[0])&&!Number.isNaN(p[1]));
      if (bounds.length) {
        const rightPadding = window.innerWidth > 1200 ? 450 : (window.innerWidth > 768 ? 200 : 50);
        map.fitBounds(bounds, { padding:[34, rightPadding], maxZoom });
      }
    }
    function marker(lat, lon, color, radius, html, tip) {
      if (lat === null || lat === undefined || Number.isNaN(Number(lat)) || lon === null || lon === undefined || Number.isNaN(Number(lon))) {
        return;
      }
      const m = L.circleMarker([lat,lon], { radius, color:'#fff', weight:1.4, fillColor:color, fillOpacity:.86 });
      m.bindPopup(html, { maxWidth:460 });
      if (tip) m.bindTooltip(tip, { direction:'top', opacity:.9 });
      m.addTo(layer);
    }
    function updateLegend(kind) {
      if (kind === 'prob') {
        legend.innerHTML = `<b>폐교 위험 확률</b><span class="dot" style="background:#dc2626"></span>15% 이상<br><span class="dot" style="background:#f97316"></span>5-15%<br><span class="dot" style="background:#facc15"></span>1-5%<br><span class="dot" style="background:#22c55e"></span>1% 미만`;
      } else {
        legend.innerHTML = `<b>학생수 감소율</b><span class="dot" style="background:#b91c1c"></span>30% 이상 감소<br><span class="dot" style="background:#f97316"></span>20-30% 감소<br><span class="dot" style="background:#facc15"></span>10-20% 감소<br><span class="dot" style="background:#22c55e"></span>10% 미만 감소/증가`;
      }
    }
    function popupSchool(d, p) {
      return `<div class="popup"><h2>${d.name}</h2><div class="meta">${d.sido} ${d.sgg} · ${d.level}</div>
      <span class="badge" style="background:${probColor(p)}">${pct(p)}</span>
      <div class="grid2"><div class="mini"><div class="k">학생수</div><div class="v">${fmt(d.student || d.student2025)}명</div></div><div class="mini"><div class="k">정책분류</div><div class="v">${d.policyCategory || '-'}</div></div><div class="mini"><div class="k">고립도</div><div class="v">${d.isolation ?? '-'}</div></div><div class="mini"><div class="k">5km 같은급</div><div class="v">${d.same5km ?? '-'}개</div></div></div></div>`;
    }
    function popupScenario(d, kind) {
      if (kind === 'school') {
        return `<div class="popup"><h2>${d.name}</h2><div class="meta">${d.sido} ${d.sgg} · ${d.level} · ${d.year}년</div><span class="badge" style="background:${declineColor(d.declinePct)}">${rawPct(d.declinePct)}</span><div class="grid2"><div class="mini"><div class="k">2025 학생수</div><div class="v">${fmt(d.student2025)}명</div></div><div class="mini"><div class="k">예측 학생수</div><div class="v">${fmt(d.forecastStudent)}명</div></div><div class="mini"><div class="k">폐교 위험</div><div class="v">${pct(d.futureHistgb)}</div></div><div class="mini"><div class="k">고립도</div><div class="v">${d.isolation ?? '-'}</div></div></div></div>`;
      }
      return `<div class="popup"><h2>${d.name}</h2><div class="meta">${d.year}년 · ${kind === 'sido' ? '시도 단위' : '시군구 단위'}</div><span class="badge" style="background:${declineColor(d.declinePct)}">${rawPct(d.declinePct)}</span><div class="grid2"><div class="mini"><div class="k">학교 수</div><div class="v">${fmt(d.count)}개</div></div><div class="mini"><div class="k">2025 학생수</div><div class="v">${fmt(d.base)}명</div></div><div class="mini"><div class="k">예측 학생수</div><div class="v">${fmt(d.forecast)}명</div></div><div class="mini"><div class="k">고위험 학교</div><div class="v">${fmt(d.high)}개</div></div></div></div>`;
    }

    function renderTest() {
      chartWrap.classList.remove('show');
      updateLegend('prob');
      panelTitle.textContent = '실제 모델 테스트 시각화';
      panelSub.textContent = '2019~2022년 검증 데이터에서 후보학교의 3년 이내 실제 폐교 라벨과 모델 예측 확률을 비교합니다.';
      const model = document.getElementById('model')?.value || 'histgb';
      const year = document.getElementById('year')?.value || 'all';
      const sido = document.getElementById('sido')?.value || 'all';
      const top = document.getElementById('top')?.value || '1000';
      const years = uniq(PAYLOAD.test.map(d=>d.year)).map(y=>({value:String(y),label:String(y)}));
      const sidos = uniq(PAYLOAD.test.map(d=>d.sido)).map(s=>({value:s,label:s}));
      controls.innerHTML = selectHtml('model','모델',[{value:'histgb',label:'튜닝 HistGB'},{value:'logistic',label:'베이스 Logistic'}],model)
        + selectHtml('year','검증연도',[{value:'all',label:'전체'},...years],year)
        + selectHtml('sido','지역',[{value:'all',label:'전체'},...sidos],sido)
        + selectHtml('top','상위 N개',[{value:'1000',label:'1000'},{value:'2000',label:'2000'},{value:'all',label:'전체'},{value:'100',label:'100'},{value:'200',label:'200'},{value:'500',label:'500'}],top);
      bindControls();
      layer.clearLayers();
      let rows = PAYLOAD.test.filter(d=>d.model===model && (year==='all'||String(d.year)===year) && (sido==='all'||d.sido===sido));
      rows.sort((a,b)=>b.prob-a.prob);
      if (top !== 'all') rows = rows.slice(0, Number(top));
      rows.forEach(d=>marker(d.lat,d.lon,probColor(d.prob),d.actual?8:5,popupSchool(d,d.prob),`${d.name} · ${pct(d.prob)} · 실제 ${d.actual?'폐교':'유지'}`));
      const actual = rows.reduce((s,d)=>s+d.actual,0);
      const avg = rows.reduce((s,d)=>s+d.prob,0)/Math.max(rows.length,1);
      const m = PAYLOAD.metrics.metrics.find(r=>r.model===modelIds[model]);
      setStats([{k:'표시 학교-연도',v:fmt(rows.length)},{k:'실제 폐교 라벨',v:fmt(actual),tone:'red'},{k:'평균 예측확률',v:pct(avg),tone:'blue'},{k:'F1',v:m?.f1?.toFixed(3) ?? '-'},{k:'ROC-AUC',v:m?.rocAuc?.toFixed(3) ?? '-'},{k:'PR-AUC',v:m?.prAuc?.toFixed(3) ?? '-'}]);
      const topRows = PAYLOAD.metrics.topk.filter(r=>r.model===modelIds[model] && [50,100,200,500,1000].includes(r.k));
      tableTitle.textContent = 'Top-K 검증';
      tableBox.innerHTML = `<table><tr><th>K</th><th>적중</th><th>P@K</th><th>R@K</th><th>Lift</th></tr>${topRows.map(r=>`<tr><td>${r.k}</td><td>${r.hits}</td><td>${r.precision}</td><td>${r.recall}</td><td>${r.lift}</td></tr>`).join('')}</table>`;
      note.textContent = '테스트 탭은 학교-연도 단위라 전체 3만 건을 한 번에 그리면 느릴 수 있어 기본은 상위 1000개로 둡니다. 전체 선택도 가능합니다.';
      fit(rows, 9);
    }

    function aggregateScenario(rows, unit) {
      const mapObj = new Map();
      for (const d of rows) {
        if (!d.lat || !d.lon || Number.isNaN(Number(d.lat)) || Number.isNaN(Number(d.lon))) continue;
        const key = unit === 'sido' ? d.sido : `${d.sido}|${d.sgg}`;
        if (!mapObj.has(key)) mapObj.set(key,{name:unit==='sido'?d.sido:`${d.sido} ${d.sgg}`,year:d.year,count:0,base:0,forecast:0,high:0,lat:0,lon:0,declinePct:0});
        const x = mapObj.get(key);
        x.count++; x.base += d.student2025||0; x.forecast += d.forecastStudent||0; x.high += d.futureHistgb >= .15 ? 1 : 0; x.lat += Number(d.lat); x.lon += Number(d.lon);
      }
      return [...mapObj.values()].map(x=>({...x,lat:x.lat/x.count,lon:x.lon/x.count,declinePct:x.base?((x.forecast/x.base)-1)*100:0}));
    }
    function scenarioRows(year, model, sido, schoolLevel) {
      return PAYLOAD.current
        .filter(d => (sido==='all'||d.sido===sido) && (schoolLevel==='all'||d.level===schoolLevel))
        .map(d => {
          const ratio = Number(year) === 2025 ? 1 : (ratioLookup.get(`${model}|${year}|${d.sido}|${d.sgg}`) ?? 1);
          const base = Number(d.student || 0);
          const forecast = Math.max(0, Math.round(base * ratio));
          const lowThreshold = String(d.level).includes('초') ? 80 : 330;
          const lowBonus = Math.max(0, (lowThreshold - forecast) / Math.max(lowThreshold, 1)) * 0.18;
          const declinePct = (ratio - 1) * 100;
          const declineBonus = Math.max(0, -declinePct) / 100 * 0.18;
          const futureHistgb = Math.max(0, Math.min(0.95, Number(d.histgb || 0) + lowBonus + declineBonus));
          const futureLogistic = Math.max(0, Math.min(0.95, Number(d.logistic || 0) + lowBonus + declineBonus));
          return {...d, year:Number(year), model, student2025:base, forecastStudent:forecast, ratio, declinePct, futureHistgb, futureLogistic};
        });
    }
    function autoScenarioUnit(sido) {
      const z = map.getZoom();
      if (sido === 'all') {
        if (z < 8) return 'sido';
        if (z < 9) return 'sgg';
        return 'school';
      }
      if (z < 8) return 'sgg';
      return 'school';
    }
    function renderScenario(shouldFit = true) {
      chartWrap.classList.remove('show');
      updateLegend('decline');
      panelTitle.textContent = '2025~2040 예측 시나리오';
      panelSub.textContent = '2025년 학교를 기준으로 시군구 학령인구 회귀 예측을 학교 학생수에 반영한 미래 시나리오입니다.';
      const year = document.getElementById('year')?.value || '2040';
      const model = document.getElementById('popModel')?.value || 'rf';
      const sido = document.getElementById('sido')?.value || 'all';
      const unit = document.getElementById('unit')?.value || 'auto';
      const schoolLevel = document.getElementById('level')?.value || 'all';
      const years = [2025, ...uniq(PAYLOAD.population.map(d=>d.year))].map(y=>({value:String(y),label:String(y)}));
      const sidos = uniq(PAYLOAD.current.map(d=>d.sido)).map(s=>({value:s,label:s}));
      const levels = uniq(PAYLOAD.current.map(d=>d.level)).map(s=>({value:s,label:s}));
      controls.innerHTML = selectHtml('year','연도',years,year)
        + selectHtml('popModel','회귀모델',[{value:'rf',label:'튜닝 RandomForest'},{value:'ridge',label:'베이스 Ridge'}],model)
        + selectHtml('sido','지역',[{value:'all',label:'전체'},...sidos],sido)
        + selectHtml('unit','표시단위',[{value:'auto',label:'자동 전환'},{value:'sido',label:'시도 단위'},{value:'sgg',label:'시군구 단위'},{value:'school',label:'학교 단위'}],unit)
        + selectHtml('level','학교급',[{value:'all',label:'전체'},...levels],schoolLevel);
      bindControls();
      layer.clearLayers();
      let rows = scenarioRows(year, model, sido, schoolLevel);
      let drawUnit = unit === 'auto' ? autoScenarioUnit(sido) : unit;
      let drawRows = drawUnit === 'school' ? rows : aggregateScenario(rows, drawUnit);
      if (drawUnit === 'school') {
        drawRows.forEach(d=>marker(d.lat,d.lon,declineColor(d.declinePct),d.futureHistgb>=.15?8:5,popupScenario(d,'school'),`${d.name} · ${rawPct(d.declinePct)}`));
      } else {
        drawRows.forEach(d=>marker(d.lat,d.lon,declineColor(d.declinePct),Math.min(22,7+Math.sqrt(d.count)*1.4),popupScenario(d,drawUnit),`${d.name} · ${rawPct(d.declinePct)}`));
      }
      const base = rows.reduce((s,d)=>s+(d.student2025||0),0);
      const forecast = rows.reduce((s,d)=>s+(d.forecastStudent||0),0);
      const decline = base ? ((forecast/base)-1)*100 : 0;
      const high = rows.filter(d=>d.futureHistgb>=.15).length;
      setStats([{k:'표시 단위',v:drawUnit==='sido'?'시도':drawUnit==='sgg'?'시군구':'학교'},{k:'학교 수',v:fmt(rows.length)},{k:'고위험 학교',v:fmt(high),tone:'red'},{k:'2025 학생수',v:fmt(base)},{k:`${year} 예측`,v:fmt(forecast),tone:'blue'},{k:'2025 대비',v:rawPct(decline),tone:'red'}]);
      tableTitle.textContent = '해석';
      tableBox.innerHTML = '<div class="note">대한민국 전체가 기본 화면입니다. 자동 전환은 낮은 줌에서 시도 단위, 중간 줌에서 시군구 단위, 확대 시 학교 단위로 바뀝니다. 2025는 기준연도라 변화율이 0%입니다.</div>';
      note.textContent = '회귀모델은 학령인구를 예측하고, 학교별 위험확률은 현재 폐교 모델 확률에 미래 학생수 감소 압력을 더해 시나리오로 표시합니다.';
      if (shouldFit) {
        if (sido === 'all') {
          if (drawUnit === 'school') {
            map.setView(KOREA_CENTER, 8);
          } else {
            map.setView(KOREA_CENTER, 7);
          }
        }
        else fit(drawRows, drawUnit==='school'?10:8);
      }
    }

    function renderPopulation() {
      updateLegend('decline');
      panelTitle.textContent = '인구 회귀 모델 시각화';
      panelSub.textContent = '시군구별 0~19세 학령인구 예측을 Ridge와 RandomForest 회귀모델로 비교합니다.';
      const model = document.getElementById('popModel')?.value || 'rf';
      const sido = document.getElementById('sido')?.value || 'all';
      const year = document.getElementById('year')?.value || '2040';
      const years = uniq(PAYLOAD.population.map(d=>d.year)).map(y=>({value:String(y),label:String(y)}));
      const sidos = uniq(PAYLOAD.population.map(d=>d.sido)).map(s=>({value:s,label:s}));
      controls.innerHTML = selectHtml('year','연도',years,year)
        + selectHtml('popModel','회귀모델',[{value:'rf',label:'튜닝 RandomForest'},{value:'ridge',label:'베이스 Ridge'}],model)
        + selectHtml('sido','지역',[{value:'all',label:'전체'},...sidos],sido);
      bindControls();
      layer.clearLayers();
      const rows = PAYLOAD.population.filter(d=>d.model===model && String(d.year)===year && (sido==='all'||d.sido===sido));
      rows.forEach(d=>{
        const center = PAYLOAD.sggCenters[d.sggCode] || [36.4,127.8];
        const decline = (d.ratio-1)*100;
        marker(center[0],center[1],declineColor(decline),Math.min(20,6+Math.sqrt(Math.max(d.population,1))/55),`<div class="popup"><h2>${d.sido} ${d.sgg}</h2><div class="meta">${d.year}년 · ${modelNames[model]}</div><span class="badge" style="background:${declineColor(decline)}">${rawPct(decline)}</span><div class="grid2"><div class="mini"><div class="k">예측 학령인구</div><div class="v">${fmt(Math.round(d.population))}명</div></div><div class="mini"><div class="k">2025 대비</div><div class="v">${rawPct(decline)}</div></div></div></div>`,`${d.sido} ${d.sgg} · ${rawPct(decline)}`);
      });
      const total = rows.reduce((s,d)=>s+d.population,0);
      const ratio = rows.reduce((s,d)=>s+d.ratio,0)/Math.max(rows.length,1);
      setStats([{k:'시군구 수',v:fmt(rows.length)},{k:'예측 학령인구',v:fmt(Math.round(total)),tone:'blue'},{k:'평균 변화율',v:rawPct((ratio-1)*100),tone:'red'}]);
      tableTitle.textContent = '회귀모델 비교 그래프';
      tableBox.innerHTML = '<div class="note">아래 그래프는 현재 지역 필터 기준으로 연도별 학령인구 합계를 보여줍니다.</div>';
      note.textContent = 'Ridge는 베이스 회귀모델, RandomForest는 비선형/상호작용을 반영한 튜닝 회귀모델입니다.';
      drawPopulationChart(sido);
      fit(rows.map(d=>({lat:(PAYLOAD.sggCenters[d.sggCode]||[36.4,127.8])[0],lon:(PAYLOAD.sggCenters[d.sggCode]||[36.4,127.8])[1]})),8);
    }
    function drawPopulationChart(sido) {
      chartWrap.classList.add('show');
      const years = uniq(PAYLOAD.population.map(d=>d.year));
      const series = {};
      for (const model of ['ridge','rf']) {
        series[model] = years.map(y => PAYLOAD.population.filter(d=>d.model===model && d.year===y && (sido==='all'||d.sido===sido)).reduce((s,d)=>s+d.population,0));
      }
      if (chart) chart.destroy();
      chart = new Chart(document.getElementById('chart'), { type:'line', data:{ labels:years, datasets:[{label:'베이스 Ridge', data:series.ridge, borderColor:'#64748b', tension:.25},{label:'튜닝 RandomForest', data:series.rf, borderColor:'#2563eb', tension:.25}] }, options:{ responsive:true, maintainAspectRatio:false, plugins:{legend:{position:'bottom'}}, scales:{y:{ticks:{callback:v=>Number(v).toLocaleString()}}} } });
    }
    function render(shouldFit = true) {
      document.querySelectorAll('.nav button').forEach(b=>b.classList.toggle('active',b.dataset.tab===tab));
      const mapEl = document.getElementById('map');
      if (tab === 'population') {
        mapEl.style.visibility = 'hidden';
        chartWrap.classList.add('full');
      } else {
        mapEl.style.visibility = 'visible';
        chartWrap.classList.remove('full');
      }
      if (tab === 'test') renderTest();
      if (tab === 'scenario') renderScenario(shouldFit);
      if (tab === 'population') renderPopulation();
    }
    document.querySelectorAll('.nav button').forEach(b=>b.addEventListener('click',()=>{ tab=b.dataset.tab; render(); }));
    map.on('zoomend', () => {
      const unit = document.getElementById('unit')?.value;
      if (tab === 'scenario' && unit === 'auto') render(false);
    });
    render();
  </script>
</body>
</html>"""
    return html.replace("__PAYLOAD__", payload_json)


def main() -> int:
    MAPS.mkdir(parents=True, exist_ok=True)
    current = load_current()
    sgg_centers = (
        current.groupby("sgg_code", as_index=False)
        .agg(lat=("lttud", "mean"), lon=("lgtud", "mean"))
        .dropna()
    )
    payload = {
        "current": build_current_payload(current),
        "test": build_test_payload(),
        "population": build_population_payload(),
        "metrics": metric_payload(),
        "sggCenters": {
            str(row["sgg_code"]).zfill(5): [round(float(row["lat"]), 6), round(float(row["lon"]), 6)]
            for _, row in sgg_centers.iterrows()
        },
    }
    html = make_html(payload)
    DASHBOARD_PATH.write_text(html, encoding="utf-8")
    print("saved:", DASHBOARD_PATH)
    print("current:", len(payload["current"]))
    print("test:", len(payload["test"]))
    print("population:", len(payload["population"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
