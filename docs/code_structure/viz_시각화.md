# 시각화 코드

`src/viz`는 최종 HTML 지도를 만드는 폴더다.  
현재 핵심 파일은 최종 HTML 생성용 `build_final_interactive_school_risk_map.py`와 정책 가중치 실험용 `streamlit_policy_simulator.py`다.

## 파일별 역할

### `build_final_interactive_school_risk_map.py`

최종 대화식 HTML 지도를 생성한다.  
Leaflet, MarkerCluster, Chart.js를 포함한 단일 HTML과 지역별 lazy-load 데이터를 만든다.

최종 출력:

```text
outputs/maps/final_national_interactive_school_risk_scenario.html
outputs/maps/final_school_risk_data/
```

## 주요 함수

- `clean_float(value, digits)`: JSON payload에 넣기 좋게 실수를 정리한다.
- `clean_int(value)`: JSON payload에 넣기 좋게 정수를 정리한다.
- `extract_sgg_name(address, sido_name, sgg_code)`: 주소에서 시군구명을 추출한다.
- `make_payload(df)`: 학교별 지도 표시용 JSON 데이터를 만든다.
- `make_help_payload()`: 팝업 도움말 문구를 JSON으로 만든다.
- `make_cohort_payload()`: 그래프 탭에서 쓸 코호트 시나리오 집계 데이터를 만든다.
- `main()`: 좌표 품질 리포트, 지역별 지도 데이터, 최종 HTML을 생성한다.

### `streamlit_policy_simulator.py`

정책 가중치를 사용자가 직접 조절해 위험등급 변화를 확인하는 Streamlit what-if 대시보드다.  
최종 발표용 HTML을 대체하는 파일이 아니라, 이해관계자용 정책 시뮬레이션 보조 앱이다.

실행:

```powershell
streamlit run src/viz/streamlit_policy_simulator.py
```

주요 함수:

- `load_scenario()`: 최종 학교별 시나리오 CSV를 읽고 유효 좌표만 남긴다.
- `weighted_score(df, weights)`: 슬라이더 가중치 기준으로 위험점수를 재계산한다.
- `apply_policy(df, weights)`: 재계산한 점수로 위험등급을 다시 부여한다.
- `build_map(df)`: 현재 필터/가중치 기준 학교 점 지도를 만든다.

## 지도 UI 기능

### 기본 로딩

- 기본 지역은 충남으로 열린다.
- 전국 데이터를 처음부터 모두 넣지 않고, 지역 선택 시 필요한 데이터를 불러온다.
- `전체` 선택 시 전국 데이터를 lazy-load한다.

### 표시단위

- `시군구 감소`: 시군구별 학생수 감소 버블을 보여준다.
- `학교별 점`: 학교별 위험등급 점을 보여준다.
- `자동 전환`: 넓게 보면 시군구 감소, 확대하면 학교별 점으로 바뀐다.

### 범례

표시단위에 따라 왼쪽 아래 범례가 다르게 보인다.

- 학교별 점: 위험등급 범례
- 시군구 감소: 시군구 학생수 감소 범례

### 팝업

학교 점을 누르면 다음 정보를 보여준다.

- 2025 학생수
- 선택 연도 예측 학생수
- 학령인구 압력
- 위험등급
- 점수에 반영된 피처
- 학교 고립도
- 상권 취약도
- 학령수요 감소압력
- EDSS 유사도
- 가까운 같은 학교급 거리
- 5km 내 같은 학교급 수

### 그래프 탭

지도 상단의 `그래프` 탭에서 다음 내용을 확인할 수 있다.

- 학생수 회귀 예측
- 3개 예측 모델 비교
- 출생 코호트 장기 시나리오 비교
- 선택 피처 기준 위험등급별 학교 수

## 주의할 점

- 위험 피처 체크박스는 회귀 예측값을 바꾸지 않는다.
- 체크박스는 위험점수 산식 시뮬레이션에만 영향을 준다.
- EDSS는 고정 보조지표로 반영되며 체크박스에는 노출하지 않는다.
- 지도 결과는 확정 폐교 예측이 아니라 통폐합 검토 우선순위 시나리오다.

