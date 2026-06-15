# 오픈소스 및 공공데이터 소스 정의서 v8 (OPEN_SOURCE_AND_DATA_SOURCE_SECTION_V8.md)

본 문서는 프로젝트 수행 과정에서 활용한 오픈소스 소프트웨어, 패키지 라이브러리와 공공데이터 및 Open API의 상세 명세를 정리한 명세서입니다. 데이터 기간 2012~2025년 기준에 입각하여 작성되었습니다.

---

### 1. 오픈소스 소프트웨어 및 라이브러리 활용 명세

#### 가. 개발 환경 및 플랫폼 오픈소스
* **Python 실행 환경**: `Python 3.9+` (Python Software Foundation License) - 핵심 데이터 처리 및 모델링 백엔드 실행
* **개발 도구**: `Visual Studio Code` (MIT License) / `Jupyter Notebook` (BSD License)
* **버전 관리**: `Git` (GPL v2) 및 `GitHub` 플랫폼

#### 나. 데이터 처리 및 시각화 패키지 라이브러리
| 패키지명 | 최소 요구 버전 | 라이선스 (License) | 활용 내용 및 역할 |
| :--- | :---: | :--- | :--- |
| **pandas** | `2.0.0` 이상 | BSD-3-Clause | 대용량 학교 패널 데이터 가공, 연도별 정렬, 데이터 전처리 및 CSV 병합 관리 |
| **numpy** | `1.24.0` 이상 | BSD-3-Clause | 수치 연산, 지형 좌표 변환(라디안 연산), 시계열 차분 및 통계적 집계 계산 |
| **scikit-learn** | `1.2.0` 이상 | BSD-3-Clause | Ridge 선형 회귀, RandomForestRegressor, HistGradientBoostingRegressor 학습 및 다중 출력(`MultiOutputRegressor`) 래핑, 교차 검증(`metrics`) 평가 |
| **openpyxl** | `3.1.0` 이상 | MIT License | 감사 보고서 및 주요 성능 분석 지표 테이블의 Excel 자동 생성, 셀 서식 및 동적 너비 최적화 |
| **joblib** | `1.2.0` 이상 | BSD-3-Clause | 최종 훈련된 모델의 직렬화 저장(`.joblib` 파일 저장 및 압축 로딩) |
| **optuna** | `4.9.0` 이상 | MIT License | 앙상블 트리 계열의 하이퍼파라미터(학습률, Regularization, iteration 수 등) 격자 튜닝 |
| **matplotlib** | `3.x` | PSF License | 최종 보고서 삽입용 한글 그래프 및 관계도 PNG 이미지 생성 시각화 백엔드 |
| **seaborn** | `0.12+` | BSD-3-Clause | 시도별 학생 수 감소량/감소율 바 차트 분포 생성 |
| **Plotly.js** | `CDN (latest)` | MIT License | 웹 대시보드 및 HTML 인터랙티브 차트(Plotly scatter/bar) 구현 |

---

### 2. 공공데이터 및 외부 Open API 활용 명세

#### 가. 공공데이터 수집원
1. **통계청 국가통계포털 (KOSIS)**
   - **데이터셋**: 시군구(SGG) 단위 합계출산율, 연도별 총 출생아 수, 순인구이동수(전입/전출), 연령대별 주민등록인구(0~19세)
   - **데이터 범위**: 2012년 ~ 2025년 전국 지자체 기준 (2025년 최종 기준연도)
   - **분석 활용 목적**: 지역별 학령인구 감소압력 프록시 변수 구축 및 1학년 코호트 연계 지표(`sgg_actual_cohort_ratio_lag7`) 산출
2. **한국교육학술정보원 (KERIS) 학교알리미**
   - **데이터셋**: 학교 기본 정보(주소, 학교급, 설립구분, 분교여부), 학급 수, 교원 수, 학년별/학급별 학생 수
   - **데이터 범위**: 2012년 ~ 2025년 전국 초·중·고등학교
   - **분석 활용 목적**: 모델링 타겟 변수(`student_delta_1yr`) 시퀀스 구축 및 학년별 흐름 구조 피처(`grade_share_1~6`, `grade_imbalance_ratio`) 계산

#### 나. 외부 Open API
1. **카카오 로컬 주소 검색 API (Kakao Developers Local API)**
   - **제공 기능**: 학교 기본 주소(텍스트)를 경도(Longitude) 및 위도(Latitude) 좌표 정보로 실시간 변환(Geocoding)
   - **활용 목적**: BallTree 인덱스 구축을 통한 지리적 고립도 계산(동일급 학교 최단 거리 산출) 및 웹 GIS 대시보드 지도 상에 위치 마커 표시

---

### 3. 산출물 작성 준수 사항
* **GitHub Personal Access Token 및 API KEY 보안**: 카카오 API 호출에 활용한 API Key 및 GitHub Personal Access Token 등은 보고서 본문 및 공용 소스코드 저장소 상에 노출하지 않으며, 각각 **`[보안상 별도 관리]`**, **`[API_KEY]`**로 처리하여 유출을 방지합니다. 어떠한 경우에도 토큰 및 키의 원문을 파일 내부에 남기지 않습니다.


