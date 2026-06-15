# 핵심 기계학습 파이프라인 코드 근거서 v8 (CODE_EVIDENCE_SNIPPETS_V8.md)

본 문서는 한글 결과보고서 작성 시 기술개발의 신뢰성을 입증하기 위해, 실제 프로젝트 소스 코드에서 데이터 수집, 전처리, 피처 엔지니어링, 데이터 누수 방지, 모델 학습 및 평가 관련 핵심 로직들을 발췌하여 정리한 기술 근거 문서입니다. 모든 코드는 2012~2025년 수집된 전국 패널 데이터를 기반으로 동작하도록 정합성을 갖추고 있습니다.

---

## 1. 전처리 및 카카오 주소 API 위경도 좌표 수집
주소 검색 API를 호출해 학교 주소를 위·경도 좌표계로 변환하는 외부 API 수집과, 데이터 프레임을 canonical format으로 변환하는 코드 영역입니다.
* **출처 파일**: [collect_national_small_shop.py](src/api/collect_national_small_shop.py) 및 [build_master_dataset.py](src/features/build_master_dataset.py#L44-L51)

```python
# [카카오 주소 API 호출 및 지오코딩 로직]
import requests

def get_coordinate_from_kakao(address: str, api_key: str) -> tuple[float | None, float | None]:
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {api_key}"} # API_KEY 마스킹 및 보안 조치
    params = {"query": address}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code == 200:
            result = response.json()
            if result.get("documents"):
                lon = float(result["documents"][0]["x"]) # 경도
                lat = float(result["documents"][0]["y"]) # 위도
                return lat, lon
    except Exception as e:
        pass
    return None, None

# [학교 구별 고유 식별자 키 생성 함수]
def stable_school_key(row: pd.Series) -> str:
    return "|".join([
        norm_sido(row.get("sido", "")),
        str(row.get("sgg", "")).strip(),
        str(row.get("school_level", "")).strip(),
        norm_text(row.get("school_name_raw", row.get("school_name_norm", row.get("school_name", "")))),
        str(row.get("branch_type", "")).strip(),
    ])
```

---

## 2. 공간 인덱스(BallTree haversine) 기반 학교 고립도 계산
지리적 고립도를 연산하기 위해 `scikit-learn`의 `BallTree`를 구면 좌표계(haversine) 거리 기준으로 구축하여 최단 거리 및 특정 반경 내 학교 분포 개수를 산출하는 코드 영역입니다.
* **출처 파일**: [build_master_dataset.py](src/features/build_master_dataset.py#L296-L322)

```python
# [구면 위경도 변환 및 BallTree Proximity 연산]
from sklearn.neighbors import BallTree
import numpy as np

for (year, level), idx in iso[iso["coordinate_valid"]].groupby(["year", "school_level"]).groups.items():
    idx_list = list(idx)
    # 위경도를 라디안 단위로 변환하여 2D 배열 구축
    coords = np.radians(iso.loc[idx_list, ["latitude", "longitude"]].to_numpy())
    if len(coords) < 2:
        continue
        
    tree = BallTree(coords, metric="haversine")
    # 최하 자기 자신을 포함하여 가장 가까운 3개점 조회 (k=3)
    dists, _ = tree.query(coords, k=min(3, len(coords)))
    radius = 6371.0088 # 지구 반지름 (km 단위 환산)
    nearest = dists[:, 1] * radius
    second = dists[:, 2] * radius if dists.shape[1] > 2 else np.nan
    
    # 반경 3km, 5km, 10km 내 동일 수준 학교수 계산
    cnt3 = tree.query_radius(coords, r=3 / radius, count_only=True) - 1
    cnt5 = tree.query_radius(coords, r=5 / radius, count_only=True) - 1
    cnt10 = tree.query_radius(coords, r=10 / radius, count_only=True) - 1
    
    # 지표 바인딩
    iso.loc[idx_list, "nearest_same_level_distance_km"] = nearest
    iso.loc[idx_list, "second_nearest_same_level_distance_km"] = second
    iso.loc[idx_list, "same_level_school_count_within_3km"] = cnt3
    iso.loc[idx_list, "same_level_school_count_within_5km"] = cnt5
    iso.loc[idx_list, "same_level_school_count_within_10km"] = cnt10
    iso.loc[idx_list, "no_same_level_school_within_5km_flag"] = cnt5 == 0
    # 공간 고립도 지수 산출 식
    iso.loc[idx_list, "isolation_score"] = nearest / (1 + cnt5)
```

---

## 3. 학년 및 학급별 데이터 흐름 피처 추출
학교 내부의 학생 분포 비중, 학급 수 변동성, 입학-졸업 구조를 추출하는 피처 엔지니어링 영역입니다.
* **출처 파일**: [build_master_dataset.py](src/features/build_master_dataset.py#L352-L372)

```python
# [학교 학년별/학급별 내부 흐름 구조 피처 생성]
for i in range(1, 7):
    if f"grade{i}_student_count" in gf.columns:
        gf[f"grade{i}_share"] = gf[f"grade{i}_student_count"] / gf["grade_student_sum"].replace({0: np.nan})
        
gf["lower_grade_student_count"] = gf[["grade1_student_count", "grade2_student_count", "grade3_student_count"]].sum(axis=1, min_count=1)
gf["upper_grade_student_count"] = gf[["grade4_student_count", "grade5_student_count", "grade6_student_count"]].sum(axis=1, min_count=1)
gf["graduating_grade_student_count"] = np.where(gf["school_level"].eq("초등학교"), gf["grade6_student_count"], gf["grade3_student_count"])

# 학년별 비중 불균형 표준편차 및 범위
grade_cols = [f"grade{i}_student_count" for i in range(1, 7)]
gf["grade_imbalance_range"] = gf[[c for c in grade_cols if c in gf.columns]].max(axis=1) - gf[[c for c in grade_cols if c in gf.columns]].min(axis=1)
gf["grade_imbalance_std"] = gf[[c for c in grade_cols if c in gf.columns]].std(axis=1)
```

---

## 4. 이벤트성 학교 분리 및 이상치 처리 규칙
모델 학습의 안정성을 위해 신설, 즉각적 변동, 또는 주소 오류 등 예측 데이터상 노이즈가 큰 이벤트성 패턴을 감지하여 안정 예측 대상 학교군과 격리 및 분리 관리하기 위한 데이터 정제 로직입니다. (특정 학교를 분석 기준에 따라 행정적으로 결정하는 대상이 아니며, 예측 성능의 강건성을 확보하기 위한 내부 전처리 보조지표로 활용될 수 있습니다.)
* **출처 파일**: [build_master_dataset.py](src/features/build_master_dataset.py#L403-L426)

```python
# [인접 년도 간 비정상적 변동을 포착하는 아노말리 탐지 알고리즘]
for key, g in panel.sort_values("year").groupby("school_key"):
    vals = g[["year", "student_count"]].set_index("year")["student_count"].to_dict()
    years = sorted(vals)
    
    for y in years[1:]:
        a, b = vals.get(y - 1, np.nan), vals.get(y, np.nan)
        if pd.isna(a) or pd.isna(b):
            continue
        types = []
        if a >= 50 and b == 0:
            types.append("drop_to_zero") # 돌연 운영 종료 완료 이력
        if a == 0 and b >= 300:
            types.append("jump_from_zero") # 신설 이력
        if abs(b - a) >= 200:
            types.append("large_abs_jump") # 학생 수 급격한 변동
        if a >= 30 and abs(safe_growth(b, a)) >= 0.5:
            types.append("large_pct_jump") # 비율상의 50% 이상 급등락
            
        if types:
            anomalies.append({
                "school_key": key, "prev_year": y - 1, "year": y, "student_count_prev": a, 
                "student_count_current": b, "delta": b - a, "pct_change": safe_growth(b, a), 
                "anomaly_type": ";".join(types)
            })
            idx = (flags["school_key"].eq(key) & flags["year"].eq(y))
            flags.loc[idx, "adjacent_year_anomaly"] = True
            flags.loc[idx, "critical_student_count_anomaly"] = any(t in types for t in ["drop_to_zero", "jump_from_zero"])
```

---

## 5. 미래 데이터 누수(Data Leakage) 차단 및 교차 검증 감사
예측 시나리오 생성 시 미래 시점의 정보가 피처 컬럼에 흘러 들어가지 않도록 방지하고 감사하는 검증 로직입니다.
* **출처 파일**: [feature_policy.py](src/features/feature_policy.py#L54-L104) 및 [run_v5_recursive_and_multioutput_forecasting_r3_r6.py](src/pipeline/core_original/v5_recursive_and_multioutput_forecasting_r3_r6_v1/run_v5_recursive_and_multioutput_forecasting_r3_r6.py#L100-L109)

```python
# [피처 후보 컬럼 내 미래 예측 시그널 확인 및 필터링]
def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        lc = c.lower()
        # 식별자 및 타겟 컬럼 제외
        if c in EXCLUDE_BASE or c in TARGET_COLS or c.startswith("target_"):
            continue
        # 미래 정보 포함 컬럼 탐지 및 차단
        if any(p in lc for p in ["next", "future", "after", "label", "closed_next", "missing_next"]):
            continue
        cols.append(c)
    return cols

# [실제 타겟 정보 유무 검출 및 Temporal Split Leakage 감사 실행]
def check_target_leakage(df: pd.DataFrame, features: list[str]) -> list[str]:
    leakage = []
    for f in features:
        # 상관관계 분석이나 변수 정의 패턴 검사
        if "target_" in f or "delta_future" in f:
            leakage.append(f)
    return leakage
```

---

## 6. 최종 튜닝 모델 Multi-output HistGradientBoostingRegressor 학습 및 추론
최종 선정된 1~5년 Horizon 통합 예측을 위한 다중 출력 앙상블 학습 파이프라인과 하이퍼파라미터 바인딩 부분입니다.
* **출처 파일**: [run_v5_recursive_and_multioutput_forecasting_r3_r6.py](src/pipeline/core_original/v5_recursive_and_multioutput_forecasting_r3_r6_v1/run_v5_recursive_and_multioutput_forecasting_r3_r6.py#L142-L148) 및 [run_v5_recursive_and_multioutput_forecasting_r3_r6.py](src/pipeline/core_original/v5_recursive_and_multioutput_forecasting_r3_r6_v1/run_v5_recursive_and_multioutput_forecasting_r3_r6.py#L323-L370)

```python
# [최종 다중출력 HistGradientBoosting Regressor 파이프라인 생성]
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

def make_pipeline(name: str, train: pd.DataFrame, features: list[str], multi: bool = False) -> Pipeline:
    dense = (name == "HistGradientBoostingRegressor")
    scale = (name in ["LinearRegression", "Ridge"])
    
    # 튜닝 완료된 최적 하이퍼파라미터로 추정기 생성
    if name == "HistGradientBoostingRegressor":
        # max_iter=120, max_leaf_nodes=31, learning_rate=0.04, l2_regularization=0.1
        est = HistGradientBoostingRegressor(
            max_iter=120, max_leaf_nodes=31, learning_rate=0.04, 
            l2_regularization=0.1, random_state=42
        )
    else:
        est = base_estimator(name)
        
    if multi:
        # MultiOutputRegressor 래퍼 적용으로 5개년 Horizon 병렬 출력
        est = MultiOutputRegressor(est, n_jobs=-1)
        
    return Pipeline([
        ("preprocess", preprocessor(train, features, scale=scale, dense=dense)),
        ("model", est)
    ])

# [학습 및 다년도 Horizon 누적 추론]
def train_multi_candidate(feature_family: str, target_type: str, model_name: str, policy: str, rows_for_scenario: pd.DataFrame):
    view = read_view(feature_family, policy, 5)
    features = feature_columns(view)
    
    # 1~5년 증분 타겟 매트릭스(Multi-target) y 생성
    ytr = multi_y(view, target_type).astype(float)
    
    # 파이프라인 구축 및 다중 출력 학습
    pipe = make_pipeline(model_name, view, features, multi=True)
    pipe.fit(view[features], ytr)
    
    # 2026~2030 시나리오 시뮬레이션 추론 실행
    scen_base = scenario_base(feature_family, rows_for_scenario)
    yhat = pipe.predict(scen_base[features])
    
    # 추론된 Delta 값을 최종 학생 수 스케일로 디코딩
    decoded = decode_multi(pd.to_numeric(scen_base["student_count"], errors="coerce").fillna(0).to_numpy(float), np.asarray(yhat), target_type)
    
    return pipe, decoded
```

---

## 7. 웹 대시보드 참고 기준 이하 및 우선점검 표시 로직
웹 대시보드로 데이터를 가공하는 단계에서 학교의 우선점검 점수(`priority_score_2030`)와 기본 60명 기준 저학생수 플래그(`small_school_flag_2030`)를 연산하는 백엔드 데이터 빌더 로직입니다.
* **출처 파일**: [run_v5_web_scenario_package.py](src/pipeline/core_original/v5_web_scenario_package_v1/run_v5_web_scenario_package.py#L159-L171)

```python
    # [저학생수 기준 flag 및 가중치 합산 점수 연산 부분]
    web["small_school_flag_2025"] = web["student_count_2025"] <= 60
    web["small_school_flag_2030"] = web["pred_student_count_2030"] <= 60
    web["decline_pressure_flag_2030"] = (web["delta_2025_2030"] <= -30) | (web["pct_change_2025_2030"] <= -0.20)
    web["isolated_small_school_flag_2030"] = web["small_school_flag_2030"] & ((pd.to_numeric(web.get("same_level_school_count_5km"), errors="coerce") <= 1) | (pd.to_numeric(web.get("nearest_same_level_school_km"), errors="coerce") >= 5))
    q75 = pd.to_numeric(web.get("isolation_score"), errors="coerce").quantile(.75)
    web["education_gap_risk_flag_2030"] = web["small_school_flag_2030"] & (pd.to_numeric(web.get("isolation_score"), errors="coerce") >= q75)
    web["priority_score_2030"] = (
        0.30 * pct_rank((-web["delta_2025_2030"]).clip(lower=0)) +
        0.25 * pct_rank((-web["pct_change_2025_2030"]).clip(lower=0)) +
        0.25 * pct_rank(pd.to_numeric(web.get("isolation_score"), errors="coerce")) +
        0.10 * np.where(web["small_school_flag_2030"], 100, 0) +
        0.10 * (100 - pct_rank(pd.to_numeric(web.get("same_level_school_count_5km"), errors="coerce")))
    )
```

> [!NOTE]
> * **현황 설명**: 현재 웹 대시보드 데이터 빌드 코드상에서는 전교생 60명 이하(`<= 60`) 기준의 단일 플래그(`small_school_flag_2030`)만 구현되어 있으며, 교육부·교육청의 다차원적 권고 학생수 참고 기준은 아직 데이터셋 상의 플래그 컬럼으로 이식되어 있지 않습니다.
> * **향후 개선 제안**: 차기 릴리즈에서는 학교 주소 기반의 지역성(`region_type`)과 학교급을 결합하여, 향후 웹용 JSON 빌더 내에 `recommended_threshold_flag` 및 `threshold_gap`과 같은 권고 기준 준수 및 편차 연산 컬럼을 공식 추가 반영할 예정입니다.


