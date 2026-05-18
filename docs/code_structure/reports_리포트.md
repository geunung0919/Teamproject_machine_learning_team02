# 리포트 생성 코드

`src/reports`는 모델 결과를 발표용 표, 그래프, Markdown 리포트로 바꾸는 폴더다.

## 파일별 역할

### `analyze_risk_threshold_sensitivity.py`

위험등급 기준값을 바꿨을 때 결과가 얼마나 달라지는지 확인한다.

주요 함수:

- `relabel_with_thresholds(frame, isolation_threshold, context_threshold)`: 기준값을 바꿔 위험등급을 다시 계산한다.
- `main()`: 여러 기준값 조합의 결과를 저장한다.

### `extract_model_feature_importance.py`

회귀/분류/변화량 모델의 피처 중요도를 추출하고 그래프로 저장한다.

주요 함수:

- `collapse_onehot_name(name)`: one-hot 피처명을 보기 쉽게 줄인다.
- `feature_importance(pipe, model_step)`: tree 모델의 importance를 추출한다.
- `permutation_feature_importance(...)`: permutation 방식으로 중요도를 계산한다.
- `save_bar(frame, title, output)`: 중요도 bar chart를 저장한다.
- `clean_change_feature_name(name)`: 변화량 모델 피처명을 발표용 이름으로 정리한다.
- `save_change_target_importance_chart()`: 변화량 모델 중요도 차트를 저장한다.
- `main()`: 중요도 CSV/PNG를 생성한다.

### `fertility_pathway_analysis.py`

출산율이 어떤 경로로 학생수 변화에 연결되는지 분석한다.

주요 함수:

- `normalize_sido_code(series)`: 시도 코드를 정규화한다.
- `make_panel()`: 출산율, 출생아수, 인구, 이동 데이터를 결합한 패널을 만든다.
- `model_importance(panel, target, features, model_name)`: target별 모델 중요도와 성능을 계산한다.
- `ridge_coefficients(panel)`: Ridge 계수 기반 영향 방향을 계산한다.
- `save_pathway_figure(panel)`: 출산율 경로 분석 그림을 저장한다.
- `main()`: 경로 분석 CSV, PNG, Markdown을 생성한다.

### `generate_comparison_chart.py`

압력비 모델, 변화량 모델, 출생 코호트 모델을 비교하는 발표용 그래프를 만든다.

주요 함수:

- `set_korean_font()`: 한글 폰트를 설정한다.
- `load_yearly_comparison()`: 연도별 모델 예측 합계를 불러온다.
- `plot_three_model_lines(comparison)`: 3개 모델 장기 예측선 그래프를 만든다.
- `build_sido_comparison()`: 시도별 모델 차이를 요약한다.
- `plot_sido_decrease_heatmap(sido_comparison)`: 시도별 감소율 heatmap을 만든다.
- `main()`: 모델 비교 CSV/PNG를 생성한다.

### `generate_model_comparison_visuals.py`

베이스 모델과 튜닝 모델의 성능 비교 그래프를 만든다.

주요 함수:

- `save_regression_comparison()`: Ridge/RandomForest 회귀모델 비교 그림을 만든다.
- `save_classifier_comparison()`: Logistic/HistGB 분류모델 비교 그림을 만든다.
- `main()`: 회귀/분류 비교 그래프를 저장한다.

## 아카이브로 이동한 구버전 리포트

다음 파일들은 EDSS 단독 성능평가나 과거 제한적 backtest 중심이라 최종 발표 기준에서 제외하고 `archive/legacy_edss_only_and_backtest_20260518`로 이동했다.

- `generate_final_reports_and_visuals.py`
- `evaluate_final_policy_backtest.py`

## 주요 출력

```text
outputs/reports/
outputs/figures/
```

## 발표 연결 포인트

- 모델 성능표
- 베이스 모델 vs 튜닝 모델 비교
- 회귀/분류 피처 중요도
- 출산율 경로 분석
- 위험등급 결과 요약

