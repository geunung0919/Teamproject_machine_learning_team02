# 피처 엔지니어링 코드

`src/features`는 원천 데이터와 학교 데이터를 결합해서 모델에 넣을 추가 피처를 만드는 폴더다.

## 파일별 역할

### `build_school_radius_commercial_features.py`

학교 반경 1km 안의 상권 데이터를 집계한다.  
프로젝트 차별점인 “상권 취약도” 계산에 직접 연결된다.

입력 데이터:

- 현재 학교 좌표
- 전국 상가업소 데이터

출력 데이터:

```text
data/processed/school_radius_commercial_features.csv
```

주요 함수:

- `classify_shop(frame)`: 업종명/분류명을 기준으로 교육, 아동, 생활 관련 업소를 구분한다.
- `count_radius_features()`: 학교 좌표와 상가업소 좌표를 이용해 반경 내 업소 수를 계산한다.
- `main()`: 학교별 반경 상권 피처를 저장한다.

### `build_modeling_master_dataset.py`

최종 피처들을 발표/분석용 마스터 데이터셋으로 다시 묶는다.  
시각화에서 쓰는 학교명은 EDSS가 아니라 현재학교 API의 `schlNm`에서 온다는 점도 source flag로 명시한다.

출력 데이터:

```text
data/processed/modeling_master_current_school_features.csv
data/processed/modeling_master_school_scenario_2026_2040.csv
outputs/reports/modeling_master_column_dictionary.csv
outputs/reports/modeling_master_dataset_summary.csv
```

주요 함수:

- `build_master_current()`: 현재학교 1행 1학교 기준 마스터셋을 만든다.
- `build_master_scenario()`: 2026~2040년 1행 1학교-연도 기준 마스터셋을 만든다.
- `build_column_dictionary(frame)`: 컬럼별 피처 그룹 사전을 만든다.
- `add_source_flags(frame)`: 학교명, EDSS 보조점수, 인구/출생/상권 피처의 원천을 표시한다.

## 이 폴더의 역할

- 학교 주변 상권 밀도 계산
- 교육/아동 관련 생활기반 확인
- “학생수가 줄어드는 학교” 중에서도 주변 기반이 약한 학교를 구분하는 보조 피처 제공
- 최종 분석용 통합 마스터 데이터셋 제공

## 발표 연결 포인트

상권 피처는 회귀 예측 성능을 크게 높이는 목적보다는, 통폐합 위험등급을 해석할 때 정책적 차별점을 주는 역할이다.

예시 설명:

> 같은 학생수 감소 학교라도 주변 상권과 생활기반이 약하면 지역 유지 가능성이 낮다고 보고 위험점수에 반영했다.

