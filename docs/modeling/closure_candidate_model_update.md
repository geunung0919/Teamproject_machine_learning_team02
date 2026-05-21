# 통폐합 후보군 모델 재정의 및 성능 기록

## 변경 배경

초기 분류 모델은 전체 학교를 대상으로 `1년 뒤 폐교`를 맞히는 구조였다.  
하지만 실제 폐교는 전체 학교 중 극히 일부라 양성 비율이 너무 낮고, 정책적으로도 “모든 학교 중 아무 학교나 폐교 예측”은 적절하지 않았다.

따라서 모델 목적을 다음처럼 바꿨다.

> 전체 학교 폐교 확정 예측이 아니라, 통폐합 검토 대상군 내부에서 3년 이내 폐교 가능성이 높은 학교를 분류한다.

## 새 모델 정의

### 후보군 필터

학습과 예측 모두 아래 후보군 안에서 수행한다.

- 기존 학교만 포함
- 학생수 0명/결측 제외
- 아래 둘 중 하나를 만족
  - 학교급별 저학생수 기준 충족
  - 분교/분교장

학교급별 저학생수 기준:

- 초등학교: 80명 이하
- 중학교: 330명 이하
- 고등학교: 330명 이하

### 교육공백 보호대상 분리

학교 고립도는 폐교 위험을 높이는 피처가 아니라, 폐교하면 교육공백이 커지는 보호 신호로 분리했다.

추가 피처:

- `replacement_available_score`: 가까운 같은 학교급과 5km 내 같은 학교급 수를 이용한 대체학교 가능성
- `isolation_protection_flag`: 대체학교가 부족하거나 고립도가 높은 학교 보호 플래그
- `closure_feasibility_score`: 통폐합 가능성 보정 점수

중요:

- 폐교 모델 성능 검증은 실제 폐교 라벨로 수행한다.
- 교육공백 보호대상은 실제 폐교 라벨로 성능 검증하지 않는다.
- 최종 지도에서는 `통폐합 검토 후보`와 `교육공백 보호대상`을 별도 정책 분류로 표시한다.

### 타깃 라벨

`closure_within_3yr_label`

현재 연도 `t` 기준으로 `t+1`, `t+2`, `t+3` 중 한 번이라도 상태가 `폐(원)교`가 되면 1로 둔다.

### 학습/검증 기간

- 학습: 2008~2018
- 검증: 2019~2022
- 2023~2025는 3년 이내 폐교 여부가 완전히 관측되지 않으므로 검증에서 제외

## 입력 피처

- 학생수, 학급수, 교원수
- 학생수 증감량/증감률
- 학생당 교지면적
- 가장 가까운 같은 학교급 거리
- 5km 내 같은 학교급 수
- 학교 고립도
- 반경 0.5km/1km 상권 수
- 시도, 학교급, 설립, 본분교

## 성능 요약

### Tuned HistGB

- Precision: 0.292
- Recall: 0.663
- F1: 0.406
- ROC-AUC: 0.971
- PR-AUC: 0.344

### Base Logistic Regression

- Precision: 0.183
- Recall: 0.395
- F1: 0.250
- ROC-AUC: 0.942
- PR-AUC: 0.154

## 학교급별 성능, HistGB

- 고등학교: F1 0.508
- 중학교: F1 0.456
- 초등학교: F1 0.375

## Top-K 성능, HistGB

- Top 10: 실제 폐교 7개 포함, Precision@K 0.700
- Top 20: 실제 폐교 10개 포함, Precision@K 0.500
- Top 50: 실제 폐교 23개 포함, Precision@K 0.460
- Top 100: 실제 폐교 45개 포함, Precision@K 0.450
- Top 500: 실제 폐교 150개 포함, Recall@K 0.575
- Top 1000: 실제 폐교 228개 포함, Recall@K 0.874

## 해석

이전 모델보다 성능과 해석력이 크게 개선되었다.  
이유는 다음과 같다.

- 라벨을 1년 뒤 폐교에서 3년 이내 폐교로 바꿔 정책 검토 기간과 맞춤
- 전체 학교가 아니라 실제 통폐합 검토 후보군 내부에서 학습
- 학생수 큰 신설/대형 학교가 위험 상위에 뜨는 문제 완화
- Top-K 기준으로 “검토 우선순위” 설명 가능

## 산출물

- 모델 학습 코드: `src/models/train_schooldata_closure_classifier.py`
- 성능표: `outputs/reports/schooldata_closure_classifier_metrics.csv`
- 학교급별 성능표: `outputs/reports/schooldata_closure_classifier_metrics_by_level.csv`
- Top-K 지표: `outputs/reports/schooldata_closure_classifier_topk_metrics.csv`
- 검증 예측값: `outputs/reports/schooldata_closure_classifier_predictions.csv`
- 2025 현재 후보 위험도: `data/processed/schooldata_current_closure_risk_2025.csv`
- HTML 지도: `outputs/maps/schooldata_model_closure_risk_2025.html`
