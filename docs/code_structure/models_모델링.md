# 모델링 코드

`src/models`는 프로젝트의 핵심 모델링 코드가 들어있는 폴더다.  
회귀모델, 분류모델, 출생 코호트 시나리오, 최종 위험등급 산출을 담당한다.

## 파일별 역할

### `model_edss_closure_risk_national.py`

EDSS 장기 학교 패널을 이용해 `5년 뒤 학교ID 소멸` proxy 분류모델을 학습한다.  
최종 분류모델 성능평가용이 아니라, 현재 학교가 과거 소멸 학교 패턴과 얼마나 유사한지 계산하는 보조 피처 생성용으로 사용한다.

주요 함수:

- `read_edss_csv(prefix, usecols)`: EDSS CSV 파일을 prefix 기준으로 읽는다.
- `load_panel()`: EDSS 학교 속성, 개황, 학생수 데이터를 결합해 연도별 패널을 만든다.
- `add_closure_labels(panel)`: 5년 뒤 학교ID가 사라지는지 proxy label을 만든다.
- `train_classifier(dataset)`: EDSS proxy 모델을 학습해 현재학교 EDSS 유사도 피처 생성에 필요한 모델을 만든다.
- `main()`: EDSS 패널, 학습 데이터, 모델, 성능 리포트를 저장한다.

### `final_national_training_pipeline.py`

전국 단위 핵심 학습 파이프라인이다.  
현재 학교 피처 결합, 학령인구 회귀모델 학습, 변화량 모델, 최종 위험 시나리오 생성을 담당한다.

주요 함수:

- `minmax_score(series, reverse)`: 값을 0~100 상대점수로 변환한다.
- `safe_mape(actual, pred, min_denominator)`: 작은 지역에서 MAPE가 과도해지는 문제를 줄인 MAPE를 계산한다.
- `latest_by_year(frame, year_col, key_col, latest_year)`: 기준연도 최신값을 추출한다.
- `add_sido_fallback(base, cols, key_col)`: 시군구 값이 없을 때 시도 단위 fallback을 붙인다.
- `load_current_school_features()`: 학교, 인구, 출산, 출생, 이동, 상권, EDSS 피처를 결합한다.
- `add_isolation_features(schools)`: 같은 학교급 거리와 5km 내 대체학교 수로 학교 고립도를 만든다.
- `add_vulnerability_scores(schools)`: 상권 취약도와 학령수요 감소압력 점수를 만든다.
- `train_population_regression()`: Ridge, RandomForest, 변화량 모델을 학습하고 회귀 성능을 저장한다.
- `forecast_sgg_population(panel, model)`: 시군구 학령인구를 2026~2040까지 예측한다.
- `forecast_sgg_population_by_change(panel, change_model)`: 변화량 타깃 모델로 장기 학령인구를 예측한다.
- `build_final_school_scenario(...)`: 학교별 예측 학생수와 위험점수/위험등급을 만든다.
- `train_objective_classifier_with_context()`: EDSS proxy 유사도 피처를 최종 리포트로 정리한다.
- `build_map(scenario)`: 단순 Folium 지도 초안을 만든다.
- `main()`: 전국 모델링 파이프라인 전체를 실행한다.

### `build_school_level_cohort_scenario.py`

출생 코호트 기반 장기 학생수 시나리오를 만든다.  
2026~2040 예측에서 출생아수 구조를 반영하기 위한 핵심 보정 단계다.

주요 함수:

- `level_birth_years(level, forecast_year)`: 학교급과 예측연도에 대응하는 출생연도 범위를 계산한다.
- `build_birth_lookup(scenario_factor)`: 출생아수 데이터를 시나리오 배율로 보정해 lookup 테이블을 만든다.
- `build_sido_migration_rate(schools)`: 시도별 순이동률을 계산한다.
- `adjusted_migration_rate(rate, mode)`: 시나리오별 이동률을 조정한다.
- `cohort_sum(birth, sgg_code, years)`: 시군구와 출생연도 묶음의 출생아수를 합산한다.
- `build_scenario_one(scenario_name, scenario_config)`: 하나의 코호트 시나리오를 생성한다.
- `main()`: 기준/낙관/비관 코호트 시나리오를 저장한다.

### `apply_cohort_scenario_to_risk.py`

출생 코호트 기준 예측 학생수를 최종 위험등급 산식에 다시 적용한다.

주요 함수:

- `main()`: 기존 최종 시나리오에 코호트 예측값을 병합하고 위험점수/위험등급을 재계산한다.

### `train_temporal_closure_classifier.py`

최종 발표용 분류 성능평가 모델이다.  
2009~2018년 데이터를 학습하고, 2019~2022년에 EDSS 학교ID가 다음 해 사라지는지 검증한다.

주요 함수:

- `build_temporal_dataset()`: EDSS 패널에 출생/출산율, 인구이동, 상권 proxy를 결합한다.
- `train_and_score(...)`: Logistic baseline과 HistGB tuned 모델을 학습하고 성능을 계산한다.
- `main()`: 시간분할 분류 성능표, 예측 결과, 모델 pkl을 저장한다.

주의:

- 타깃은 실제 폐교 확정명부가 아니라 EDSS 학교ID 다음 해 소멸 proxy다.
- 과거 EDSS에는 좌표 이력이 없어 학교 고립도/대체학교 거리/학교 반경 상권 피처는 직접 넣지 못한다.

## 모델별 역할 구분

| 구분 | 역할 |
|---|---|
| 회귀모델 | 시군구 학령인구와 학교별 학생수 예측 |
| 변화량 모델 | 인구이동을 반영한 감소 속도 비교 |
| 출생 코호트 시나리오 | 출생아수 기반 장기 학생수 보정 |
| 분류모델 | 2019~2022 EDSS 다음 해 소멸 proxy 검증 |
| EDSS proxy 모델 | 최종 분류 피처 중 EDSS 유사도 생성 |
| 위험등급 산식 | 예측 학생수, 고립도, 상권, 학령수요 감소, EDSS, 대체학교 접근성 결합 |

## 발표 연결 포인트

- 회귀모델은 “학생수 예측”을 담당한다.
- 위험등급은 “예측값 + 정책 피처”를 결합한 우선순위 산식이다.
- EDSS는 확률이 아니라 과거 소멸 학교 패턴과의 상대 유사도다.

