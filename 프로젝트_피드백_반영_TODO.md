# 프로젝트 피드백 반영 TODO

기준 파일: `C:\Users\wltjd\Downloads\프로젝트_분석_피드백.md`

## 반영 완료

- [x] 회귀모델 baseline 비교 추가
  - 파일: `src/final_national_training_pipeline.py`
  - 산출물: `outputs/reports/final_national_population_regression_metrics.csv`
  - 추가 항목: `baseline_previous_year_population`
  - 의미: “올해 학령인구를 내년에도 그대로 유지한다”는 naive baseline과 ML 모델을 비교할 수 있게 함

- [x] 분류모델 표현 정리
  - 현재 지도/문서 표현은 “폐교 확률”이 아니라 `EDSS 폐교패턴 유사도`, `통폐합 검토 우선순위`로 정리됨
  - EDSS 백분위는 확률이 아니라 전국 상대순위라고 팝업에 표시

- [x] 지도 초기 로딩 최적화
  - 전국 데이터를 HTML 하나에 넣지 않고 지역별 JS 파일로 지연 로딩
  - 기본 지역은 충남
  - 산출물: `outputs/maps/final_school_risk_data/*.js`

- [x] 지도 UX 개선
  - 팝업을 위험등급 카드, 학생수 요약, 점수 기여 chip, 세부 지표 카드로 재구성
  - 세부 지표에 위험 방향 배지/색상 추가
  - 시군구 감소 버블과 학교별 점 자동 전환 추가

- [x] 공통 설정 모듈화
  - 파일: `src/project_config.py`
  - 포함: 위험등급 라벨/색상, 좌표 bounds, 학교급 정규화, 위험점수 산식

- [x] 좌표/데이터 품질 리포트 강화
  - `outputs/reports/final_invalid_coordinates_2026_2040.csv`
  - `outputs/reports/final_student_count_data_check_2026_2040.csv`
  - `outputs/reports/final_special_school_review_2026_2040.csv`

- [x] 위험등급 임계값 민감도 분석 추가
  - 파일: `src/analyze_risk_threshold_sensitivity.py`
  - 산출물: `outputs/reports/risk_threshold_sensitivity.csv`
  - 분석: 2029/2035/2040년 기준 고립도 60/70/80, 상권·지역감소 60/70/80 조합별 고위험 학교 수 변화

- [x] 학교 반경 기반 상권 피처 생성
  - 파일: `src/build_school_radius_commercial_features.py`
  - 산출물: `data/processed/school_radius_commercial_features.csv`
  - 요약: `outputs/reports/school_radius_commercial_feature_summary.csv`
  - 반경: 500m, 1km, 2km
  - 업종: 전체, 교육, 아동/키즈, 의료

- [x] 학교급별 코호트 이동 시나리오 추가
  - 파일: `src/build_school_level_cohort_scenario.py`
  - 산출물: `data/processed/school_level_cohort_scenario_2026_2040.csv`
  - 요약: `outputs/reports/school_level_cohort_scenario_total_summary.csv`
  - 방식: 초등/중등/고등별 출생연도 코호트 합계 비교
  - 시나리오: 비관/기준/낙관

## 부분 반영

- [~] 지역별 출산율 반영
  - 회귀모델 피처에 `total_fertility_rate`, `tfr_yoy_rate`, `birth_count`, `birth_count_yoy_rate` 포함
  - 위험등급의 `regional_decline_risk_score`에도 출산율/출생아 수 반영
  - 단, 현재 최종 예측은 학년별 코호트 이동 모델이 아니라 시군구 0~19세 학령인구 압력비 기반 시나리오임

- [~] 상권 데이터 활용
  - 학교 반경 500m/1km/2km 기반 상권 피처는 생성 완료
  - 단, 최종 위험 산식은 아직 기존 시군구 상권 취약도 기반
  - 다음 단계는 반경 기반 상권 취약도와 기존 지표를 비교한 뒤 최종 산식에 반영하는 것

- [~] 차별 피처의 분류 성능 기여 검증
  - 정책 위험점수에는 학교 고립도/상권/지역감소가 반영됨
  - 하지만 EDSS supervised 학습에는 과거 학교 좌표 매칭 한계로 학교별 고립도 피처가 직접 들어가지 못함
  - 따라서 “성능 향상”보다 “정책 해석력 강화”로 표현해야 함

## 아직 미반영: 우선순위 높음

- [x] 회귀 타겟을 변화량/변화율로 바꾼 실험 추가
  - 현재 타겟: 다음 해 0~19세 학령인구 수
  - 개선 타겟 후보:
    - `school_age_pop_0_19(t+1) - school_age_pop_0_19(t)`
    - `(t+1 / t) - 1`
  - 목적: 현재 학령인구 피처가 중요도를 지배하는 문제 완화

## 아직 미반영: 후속 개선

- [ ] 코호트 시나리오를 최종 지도 UI에 통합
  - 현재 최종 지도는 기존 시군구 학령인구 압력비 기반 시나리오 표시
  - 코호트 시나리오는 별도 CSV/리포트로 생성됨
  - 다음 단계: 지도 그래프 탭에 기존 모델 vs 코호트 시나리오 비교 추가

- [ ] EDSS-학교알리미 매칭으로 분류 학습에 고립도 피처 추가
  - 학교명 + 시도 + 학교급 + 설립일 등으로 매칭
  - 좌표가 붙으면 과거 폐교 proxy 분류에 고립도 직접 투입 가능

- [x] 시간 기반 교차 검증 추가
  - 회귀: expanding/sliding window
  - 분류: base_year 기준 rolling split

- [ ] 하이퍼파라미터 탐색 기록 추가
  - RandomizedSearchCV 또는 기본값 대비 성능 비교

- [x] `requirements.txt`, `.gitignore` 추가
- [ ] 최소 테스트 추가
  - `.env`, `data/raw/`, `outputs/models/*.pkl` 제외
  - `compute_policy_risk_score`, `assign_policy_risk_label`, `valid_sido_coord_mask` 단위 테스트

## 발표 시 반드시 지킬 표현

- [x] “폐교 예측”보다 “통폐합 검토 우선순위”로 표현
- [x] R² 0.999를 “정확도 99.9%”라고 말하지 않기
- [x] EDSS 백분위는 확률이 아니라 상대 유사도라고 설명
- [x] 상권/고립도는 성능 향상보다 정책 해석력/차별점으로 설명
- [x] 2040 결과는 확정 예측이 아니라 시나리오로 설명
## 2차 피드백 반영 체크

- [x] 학교 반경 1km 상권 피처를 최종 위험 산식에 반영
  - `src/final_national_training_pipeline.py`
  - `outputs/reports/commercial_vulnerability_score_distribution.csv`
- [x] 회귀모델 변화량 타깃 실험 추가
  - `outputs/reports/final_national_population_regression_metrics.csv`
  - `outputs/reports/change_target_regression_feature_importance.csv`
- [x] 시군구 규모별 회귀 성능 분리 보고 추가
  - `outputs/reports/regression_performance_by_sgg_size.csv`
- [x] 코호트 시나리오에 지역별 순이동 보정 추가
  - `src/build_school_level_cohort_scenario.py`
  - `outputs/reports/school_level_cohort_scenario_total_summary.csv`
- [x] 기존 시군구 압력비 모델과 출생 코호트 모델 비교 시각화 추가
  - `src/generate_comparison_chart.py`
  - `outputs/figures/model_vs_cohort_comparison.png`
  - `outputs/reports/model_vs_cohort_comparison.csv`
- [x] 최종 HTML 지도 재생성
  - `outputs/maps/final_national_interactive_school_risk_scenario.html`
- [x] 출생 코호트 기준 학생수를 최종 위험등급 산식에 적용
  - `src/apply_cohort_scenario_to_risk.py`
  - `outputs/reports/cohort_applied_risk_summary_2026_2040.csv`
- [x] 현재 학교 적용용 supervised 피처 결측 일부 보완
  - `src/final_supervised_training.py`
  - `outputs/reports/final_supervised_current_feature_missing_rate.csv`
- [x] 회귀/분류 베이스모델 vs 튜닝모델 비교 산출물 생성
  - 회귀: 전년 유지/Ridge vs RandomForest
  - 분류: LogisticRegression vs HistGradientBoostingClassifier
  - `outputs/reports/regression_base_vs_tuned_comparison.csv`
  - `outputs/reports/classification_base_vs_tuned_comparison.csv`
  - `outputs/figures/regression_base_vs_tuned_comparison.png`
  - `outputs/figures/classification_base_vs_tuned_comparison.png`

## 계속 남은 과제

- [ ] EDSS 학교 이력과 현재 학교 매칭 고도화
- [x] 시간 기반 교차검증 추가
- [x] requirements/gitignore 정리
- [ ] tests 정리
## 3차 피드백 반영 TODO

- [x] 분류모델 현재학교 추론 피처 100% 결측 완화
  - 수정 파일: `src/final_supervised_training.py`
  - `entrants`, `graduates`: 학교급별 학생수 기반 추정
  - `student_growth_1yr`, `land_area`, `playground_area`, `regular_classrooms`: EDSS 2023 시도/학교급 중앙값 fallback
  - 결과 리포트: `outputs/reports/final_supervised_current_feature_missing_rate.csv`

- [x] 변화량 타깃 회귀모델 피처 중요도 차트 생성
  - 수정 파일: `src/extract_model_feature_importance.py`
  - 그림: `outputs/figures/change_target_regression_feature_importance.png`
  - 핵심 피처: 15~19세 인구, 순이동 인구, 전출 인구

- [x] 시간 기반 교차검증 추가
  - 수정 파일: `src/final_national_training_pipeline.py`
  - 방식: expanding window
  - 결과: `outputs/reports/regression_time_cv_results.csv`

- [x] 재현성 파일 추가
  - `requirements.txt`
  - `.gitignore`

- [x] 최종 지도 HTML 재생성
  - 코호트 시나리오 재적용: `src/apply_cohort_scenario_to_risk.py`
  - 최종 HTML: `outputs/maps/final_national_interactive_school_risk_scenario.html`

- [ ] 현재학교 상세 API에서 실제 입학생/졸업생/부지면적 원자료 추가 확보
  - 현재는 추정/fallback으로 결측을 줄인 상태
  - 실제 원자료 확보 시 supervised 현재학교 적용 신뢰도 개선 가능
