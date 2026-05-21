# 학령인구 감소 기반 학교 통폐합 위험 분석

전국 학교-연도 데이터와 복구 좌표를 이용해 학령인구 감소, 학교 고립도, 상권 취약도, 대체학교 접근성을 함께 분석하는 기계학습 프로젝트입니다.

## 바로 볼 파일

- 최종 3탭 대시보드: `outputs/maps/schooldata_model_closure_risk_2025.html`
- 동일한 대시보드 별칭: `outputs/maps/school_project_dashboard.html`

대시보드는 다음 3개 탭으로 구성됩니다.

1. `모델 테스트`: 2019~2022 검증 데이터에서 실제 폐교 라벨과 모델 예측 결과를 지도에 표시
2. `2025~2040 시나리오`: 2025년 학교를 기준으로 회귀모델의 학령인구 감소율을 반영한 미래 시나리오 표시
3. `인구 회귀`: Ridge와 RandomForest 회귀모델의 시군구 학령인구 예측 비교

과거 `final_national_interactive_school_risk_scenario.html`은 구버전 모델 결과이므로 최종 발표용으로는 사용하지 않는 것이 좋습니다.

## 실행 순서

현재 최종 모델을 다시 학습하고 2025 지도를 생성하려면:

```powershell
pip install -r requirements.txt
python src/models/train_schooldata_closure_classifier.py
python src/viz/build_schooldata_model_closure_map.py
```

위 명령을 실행하면 최종 3탭 대시보드가 다시 생성됩니다.

## 최종 모델 성능

최종 분류 모델은 전체 학교가 아니라 통폐합 검토 후보군 안에서 `3년 이내 폐교`를 예측합니다.

- Tuned HistGB: F1 `0.406`, ROC-AUC `0.972`, PR-AUC `0.344`
- Base Logistic: F1 `0.250`, ROC-AUC `0.942`, PR-AUC `0.154`
- HistGB Top 100: 실제 폐교 `45개` 포함
- HistGB Top 1000: 실제 폐교 `228개` 포함

## 주요 데이터

- 장기 학교 패널: `data/processed/schooldata_modeling_panel_2008_2025_geocoded.csv`
- 2025 폐교 후보 위험 결과: `data/processed/schooldata_current_closure_risk_2025.csv`
- 회귀 시군구 예측 결과: `data/processed/final_national_sgg_population_forecast_2026_2040.csv`
- 변화량 회귀 시군구 예측 결과: `data/processed/final_national_sgg_population_forecast_change_model_2026_2040.csv`

## 공유 주의

`.env`에는 API 키가 들어갈 수 있으므로 팀원에게 공유하거나 GitHub에 올리지 마세요.
