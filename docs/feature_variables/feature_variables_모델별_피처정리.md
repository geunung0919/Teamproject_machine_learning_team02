# 모델별 피처 변수 정리

이 문서는 최종 발표 기준 모델만 정리한다.  
정책 후보 재현 실험은 실제 검증 목적과 맞지 않아 `archive/policy_reproduction_experiment_20260518`로 이동했다.

## 회귀모델

### 목표

2026~2040년 미래 학령인구와 학교별 예상 학생수를 예측한다.

### 모델

| 구분 | 모델 | 역할 |
|---|---|---|
| 베이스 | Ridge | 선형 기준모델 |
| 튜닝 | RandomForest | 비선형 패턴 반영 |

### 주요 피처

| 그룹 | 주요 컬럼 |
|---|---|
| 학령인구 | `school_age_pop_0_19`, `pop_0_4`, `pop_5_9`, `pop_10_14`, `pop_15_19` |
| 출생/출산율 | `birth_count`, `birth_count_yoy_rate`, `total_fertility_rate`, `tfr_yoy_rate` |
| 인구이동 | `in_migration_total`, `out_migration_total`, `net_migration_total`, `in_migration_yoy_rate`, `out_migration_yoy_rate` |
| 상권 | `commercial_count`, `education_business_count`, `kids_business_count`, `medical_business_count` |
| 지역효과 | `sido_code`, `sgg_code`, `year` |

### 성능

| 모델 | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Ridge | 642.4 | 943.9 | 0.9993 |
| RandomForest | 566.0 | 1004.6 | 0.9992 |

## 분류모델

### 목표

2009~2018년 과거 데이터를 학습하고, 2019~2022년에 EDSS 학교ID가 다음 해 사라지는지 검증한다.

이 검증은 실제 폐교 확정명부와 완전히 같지는 않지만, 현재 보유한 장기 학교 패널에서 만들 수 있는 가장 일관된 시간분할 검증이다.

### 모델

| 구분 | 모델 | 역할 |
|---|---|---|
| 베이스 | Logistic Regression | 선형 분류 기준모델 |
| 튜닝 | HistGradientBoosting | 비선형 조합 학습 |

### 주요 피처

| 그룹 | 주요 컬럼 |
|---|---|
| EDSS 학교 규모 | `student_count`, `students_per_class`, `students_per_teacher`, `student_growth_1yr` |
| 학교 속성 | `sido_name`, `학제명`, `설립구분명` |
| 출생/출산율 | `birth_count`, `avg_total_fertility_rate`, `avg_birth_count_yoy_rate` |
| 인구이동 | `net_migration_total`, `in_migration_total`, `out_migration_total`, `net_migration_per_birth` |
| 상권 proxy | `commercial_count`, `education_business_count`, `kids_business_count`, `commercial_per_birth`, `education_per_birth` |

과거 EDSS에 없어 검증모델에 직접 넣지 못한 피처:

- 학교 고립도
- 가장 가까운 같은 학교급 거리
- 5km 내 같은 학교급 수
- 학교 반경 상권 피처

이 피처들은 최종 지도 위험점수 산식에는 사용하지만, 2009~2022 과거 검증에는 좌표 이력이 없어 직접 검증하지 못한다.

### 성능

| 모델 | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic | 0.695 | 0.717 | 0.706 | 0.970 | 0.608 |
| HistGB | 0.688 | 0.717 | 0.702 | 0.983 | 0.725 |

현재 분류모델은 학습-추론 불일치를 줄이기 위해 현재 학교 예측/서빙에서 안정적으로 확보 가능한 10개 피처만 사용한다.  
제외한 피처 목록은 `outputs/reports/temporal_closure_feature_set_report.csv`에 저장된다.

F1 기준으로는 Logistic이 소폭 높지만, 튜닝한 HistGB는 ROC-AUC와 PR-AUC가 더 높아 폐교 가능 후보를 상위 랭킹으로 정렬하는 성능이 더 좋다.

## 최종 지도 위험점수

지도에 표시되는 위험등급은 분류모델 확률만으로 만든 것이 아니라, 다음 피처를 결합한 최종 위험점수 산식이다.

| 그룹 | 주요 컬럼 |
|---|---|
| 회귀 예측 | `forecast_student_count`, `population_pressure_ratio` |
| 현재 학교 규모 | `student_count_2025` |
| 학교 고립도 | `school_isolation_score`, `nearest_same_level_school_km`, `same_level_school_count_5km` |
| 상권 | `commercial_vulnerability_score` |
| 학령수요 감소 | `regional_decline_risk_score` |
| EDSS 유사도 | `objective_closure_percentile`, `objective_top10_flag` |

EDSS는 최종 분류모델 자체가 아니라, 과거 소멸 학교와의 유사도를 나타내는 보조 피처다.

## 주요 출력

```text
outputs/reports/final_national_population_regression_metrics.csv
outputs/reports/regression_base_vs_tuned_comparison.csv
outputs/reports/temporal_closure_classifier_metrics.csv
outputs/reports/classification_base_vs_tuned_comparison.csv
outputs/reports/regression_feature_importance.csv
outputs/reports/classification_temporal_closure_feature_importance.csv
outputs/maps/final_national_interactive_school_risk_scenario.html
```
