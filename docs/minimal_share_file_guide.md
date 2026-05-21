# 최소 공유 파일 가이드

## 목적

팀원이 프로젝트를 받았을 때 최종 모델을 읽고, 재학습하고, HTML 시각화를 다시 만들 수 있는 최소 파일만 남긴 구조다.

## 남긴 핵심 데이터

| 경로 | 역할 |
|---|---|
| `data/processed/schooldata_modeling_panel_2008_2025_geocoded.csv` | 최종 학교-연도 master dataset. 학생수, 상태, 좌표, 고립도, 상권 피처 포함 |
| `data/processed/schooldata_current_closure_risk_2025.csv` | 2025년 현존 통폐합 후보군에 대한 모델 예측 결과 |
| `data/processed/schooldata_closed_unmatched_after_geocode.csv` | 끝까지 좌표 복구가 안 된 폐교 목록 |

## 남긴 핵심 산출물

| 경로 | 역할 |
|---|---|
| `outputs/maps/schooldata_model_closure_risk_2025.html` | 최종 모델 기반 지도 |
| `outputs/models/base_logistic_schooldata_closure.pkl` | Base Logistic 모델 |
| `outputs/models/tuned_histgb_schooldata_closure.pkl` | Tuned HistGB 모델 |
| `outputs/reports/schooldata_closure_classifier_metrics.csv` | 모델 성능표 |
| `outputs/reports/schooldata_closure_classifier_metrics_by_level.csv` | 학교급별 성능표 |
| `outputs/reports/schooldata_closure_classifier_topk_metrics.csv` | Top-K 평가 지표 |
| `outputs/reports/schooldata_closure_classifier_predictions.csv` | 검증 기간 예측 결과 |

## 남긴 핵심 코드

| 경로 | 역할 |
|---|---|
| `src/models/train_schooldata_closure_classifier.py` | 최종 분류 모델 학습 및 현재 위험도 생성 |
| `src/viz/build_schooldata_model_closure_map.py` | 최종 HTML 지도 생성 |
| `src/features/*geocode*.py` | 좌표 복구 과정 재현용 코드 |
| `src/config_loader.py` | `.env` API 키 로더 |

## 재실행 방법

```powershell
pip install -r requirements.txt
python src/models/train_schooldata_closure_classifier.py
python src/viz/build_schooldata_model_closure_map.py
```

## API 키 필요 여부

최종 모델 재학습과 지도 생성에는 API 키가 필요 없다.

API 키가 필요한 경우는 다음뿐이다.

- 원본 주소를 다시 지오코딩할 때
- 공공데이터/KOSIS/상권 데이터를 처음부터 다시 수집할 때

공유 시 `.env`는 개인 키가 들어 있으므로 팀원에게 직접 공유하지 않는 것이 좋다.

## 아카이브 위치

정리하며 제외한 파일은 아래로 이동했다.

```text
archive/share_cleanup_20260521
```

포함 내용:

- `data/temp`: 수동 주소 복구 임시 파일
- `data/raw`: API 원본/중간 파일
- 구버전 지도/모델/리포트
- 과거 실험용 루트 문서와 노트북

현재 결과를 이해하거나 재학습하는 데는 아카이브 파일이 필요 없다.
