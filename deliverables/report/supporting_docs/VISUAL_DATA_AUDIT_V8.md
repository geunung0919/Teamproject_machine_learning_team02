# 시각화 원천 데이터 추적성 감사서 v8 (VISUAL_DATA_AUDIT_V8.md)

본 문서는 최종 결과보고서에 삽입된 12종의 시각화 자원(그림 및 HTML 대시보드)과 8종의 표 데이터가 어떤 원천 CSV/JSON 파일의 어느 변수(Column)를 활용하여 연산·집계되었는지 추적하기 위한 데이터 리니지(Data Lineage) 감사서입니다. 모든 수치는 2012~2025년 원천 데이터 수집 범위를 기준으로 정합성 감사를 완료하였습니다.

---

### 1. [그림] 시각화 자원별 데이터 출처 및 변수 매핑 테이블

| HWP 번호 | 자원 식별자 | 그림/자료명 | 연계 원천 데이터 파일 경로 | 사용된 핵심 변수 (Columns) | 데이터 수치 집계 및 연산 기준 |
| :---: | :---: | :--- | :--- | :--- | :--- |
| **그림 1** | **fig01** | **2018~2025년 학령인구 및 학생 수 변화 추이** | 1) `data/v5_clean_dataset_patch_v1/canonical/school_year_panel.csv`<br>2) `data/v5_clean_dataset_patch_v1/canonical/sgg_year_demographics.csv` | 1) `year`, `student_count`<br>2) `year`, `school_age_population_0_19` | 2018~2025년 범위 내에서 연도(`year`)를 기준으로 그룹화하여 `student_count`의 합산값(좌측 Y축)과 `school_age_population_0_19`의 합산값(우측 Y축, 이중축)을 연산함. |
| **그림 2** | **fig02** | **시도별 학생 수 감소량 비교 (2018년 대비 2025년)** | `data/v5_clean_dataset_patch_v1/canonical/school_year_panel.csv` | `year`, `sido`, `student_count` | 2018년 시도별 학생 수 합계와 2025년 시도별 학생 수 합계를 구한 후, `감소량 = (2018년 학생수 - 2025년 학생수)`를 계산하여 내림차순 정렬함. |
| **그림 3** | **fig03** | **시도별 학생 수 감소율 비교 (2018년 대비 2025년)** | `data/v5_clean_dataset_patch_v1/canonical/school_year_panel.csv` | `year`, `sido`, `student_count` | 2018년 대비 2025년 학생 수 감소율을 `감소율(%) = (감소량 / 2018년 학생수) * 100` 수식으로 계산하여 내림차순 정렬함. |
| **그림 4** | **fig06** | **이벤트성 학교 분리 결과 요약 (총 2,173개교)** | `presentation_materials/tables/excluded_school_summary.csv` | `category`, `count` | 유효 모델링 예측 대상(안정 예측 대상 학교)을 격리하기 위해 분석 및 분리 처리된 특수 학교(총 2,173개교)의 사유별 개수를 바 차트로 시각화함. (주원인 및 처리 방식 반영) |
| **그림 5** | **fig04** | **V5 ML 학령인구 및 학생 수 감소압력 시나리오 분석 데이터 파이프라인 (2012~2025)** | (개념 아키텍처 다이어그램) | (없음) | 데이터 수집(2012~2025년) $\rightarrow$ 이벤트성 학교 분리 (2,173개교) $\rightarrow$ 마스터 패널 구축 $\rightarrow$ R-stage 확장 $\rightarrow$ 예측 모델(HistGB) 학습 및 평가 $\rightarrow$ 웹 패키지 및 대시보드 구축의 6단계 흐름을 시각화함. |
| **그림 6** | **fig05** | **R0~R5 R-stage 피처 확장 구조도** | `src/features/feature_policy.py` 및 `src/features/build_master_dataset.py` | (각 피처 단계별 핵심 개념) | R0(Baseline)에서 시작하여 R5(실제 코호트 변수)까지 피처 컬럼을 나열하는 대신 2~4줄 수준의 핵심 설계 철학 중심 요약 배치도를 시각화함. |
| **그림 7** | **fig10** | **핵심 코드 및 피처 생성 흐름 관계도 (2012~2025)** | (프로젝트 디렉토리 소스 코드 라인) | (없음) | 전처리(`build_master_dataset.py`) $\rightarrow$ 이벤트성 학교 분리 $\rightarrow$ 모델링 $\rightarrow$ 배포 패키지(`export_web_package.py`)의 소스 파일 간 종속 관계를 2012~2025년 수집 연도 기준에 맞추어 표시함. |
| **그림 8** | **fig07** | **예측 기간별 최종 HistGB 모델 성능 변화** | `data/v5_r3_r6_rf_hist_tuning_v1/tuning_validation_metrics_by_horizon.csv` | `horizon`, `level_MAE`, `delta_R2` | R3 기반 HistGB 튜닝 모델(`hgb_05_deeper_regularized`)의 Horizon 1~5년 성능을 기준으로 X축은 `horizon`으로 통일하고, 위에는 `MAE` 실점수, 아래에는 `Delta R2` 설명력 추이를 상하 2분할 서브플롯 형태로 그림. |
| **그림 9** | **fig08** | **2025~2030년 예측 총 학생 수 시나리오 (안정 예측 대상 학교 기준)** | `presentation_materials/tables/summary_national_by_year.csv` | `year`, `total_student_count` | 안정 예측 대상 학교들의 연도별 예측 총 학생 수를 X축 `year` 2025~2030년에 맞춰 선그래프로 도시하고 포인트마다 수치 텍스트 라벨을 표시함. |
| **그림 10** | **web_01** | **웹 대시보드 전체 요약 화면** | `summary_national_by_year.csv` 또는 웹 앱 UI 캡처 | `total_school_count`, `total_student_count`, `decline_rate`, `gap_school_count` 등 | 총 학교 수, 총 학생 수, 감소율, 교육공백 우려 학교 수, 2025~2030 예측 학생 수 추이 및 감소압력 상위 학교 요약 등을 반응형 웹 화면에 종합적으로 요약함. |
| **그림 11** | **web_02** | **학교 단위 지도 및 상세 진단 화면** | `final_scenario_school_web.json`, `excluded_school_web.json` 또는 웹 앱 UI 캡처 | `latitude`, `longitude`, `pred_student_count_2030`, `isolation_score`, `priority_score` | 지도 위에 필터링된 개별 학교 위치 마커를 표시하고, 클릭 시 해당 학교의 예측 학생 수, 감소율, 고립도, 최근접 학교 거리, 반경 내 학교 수, 우선점검 참고 표시 여부를 상세 패널에 표시함. |
| **그림 12** | **web_03** | **지역 단위 요약 및 시나리오 분석 화면** | `summary_national_by_year.csv` 또는 웹 앱 UI 캡처 | `sido`, `sgg`, `pred_student_count_2030` 등 | 시도 및 시군구(SGG) 지역 단위의 2025~2030년 총량 추이, 누적 감소율, 감소압력 상위 학교 명단 및 등급별 학교 분포 분포를 요약 분석함. |

---

### 2. [표] 표 데이터별 데이터 출처 및 연계 원천 명세

| 표 번호 | 표 제목 | 데이터 출처 소스 | 사용된 핵심 변수 및 연산 로직 |
| :---: | :--- | :--- | :--- |
| **표 1** | **데이터 수집 및 원천 구성표** | 통계청(KOSIS), 학교알리미(KERIS), Kakao Local API 등 | 수집 데이터 원천, 수집 범위(2012~2025년), 수집된 컬럼 항목들을 논리적 분석 목적별로 매핑한 정의표 |
| **표 2** | **이벤트성 학교 분리 및 이상치 처리 전국 요약표** | `presentation_materials/tables/excluded_school_summary.csv` | 분석에 노이즈를 유발하는 급증/급감 및 결측, 이상 주소 2,173개교의 학교급별 빈도 및 2025년 기준 학생수 요약 |
| **표 3** | **R-stage별 구성 피처 그룹 정의** | `src/features/build_master_dataset.py` 등 | R0(Baseline)부터 R5(실제 코호트)까지 각 모델링 단계에 투입된 변수군 정의와 유효성 판단 기준 명세 |
| **표 4** | **R3/R5 RF 및 HistGB 제한 후보 비교 (Top 3 성능 요약)** | `data/v5_r3_r6_rf_hist_tuning_v1/tuning_validation_metrics_by_horizon.csv` | 20개 모델 파라미터 조합 검증을 통해 최적의 모델(1위 `hgb_05_deeper_regularized`)을 탐색하고 상위 3종의 지표를 요약한 비교표 |
| **표 5** | **최종 선택 모델의 예측 기간(Horizon)별 성능 지표** | 최종 학습 검증 세트 스코어 결과 데이터 | Horizon 1~5년별 검증 스코어(MAE, RMSE, Level R2, Delta R2, Median AE, 저학생수 학교 MAE) |
| **표 6** | **2025~2030년 전국 예측 학생 수 및 기본 추이** | `presentation_materials/tables/summary_national_by_year.csv` | 2025~2030년 안정 예측 대상 학교 기준 전국 총 학생수 예측치, 학교당 평균 학생수, 누적 감소율 추이 요약 |
| **표 7** | **2030년 최종 시점 기준 감소압력 및 교육공백 우려 지표** | 2030년 시나리오 분석 최종 생성본 | 2030년 최종 시점 기준 감소압력 학교(5,425개), 고립도 높은 학교(1,462개), 교육공백 우려 학교(1,848개) 표시 요약 |
| **표 8** | **기존 적정규모학교 권고 학생수 참고 기준 및 웹 표시 방식** | 교육부 및 시도별 교육청 고시 지침안 | 학교의 지역성(면, 읍, 도시)과 학교급을 기준으로 과거 교육부의 적정규모학교 권고 학생수 참고 기준(60, 120, 180, 240, 300) 매핑 및 웹 지도 우선점검 참고 표시 사양 정의 |

---

### 3. 데이터 검증 지침 (Audit Integrity)
* 본 추적 테이블의 모든 집계 결과와 원천 데이터셋의 수치는 100% 일치하며, 2030년 예측 전국 학생수(`3,299,224.39명`), 감소율(`-14.03%`), 이벤트성 분리 대상 학교 수(`2,173개교`) 등 최종 보고서 본문에 기재된 수치들은 위 원천 데이터 파일들을 통해 교차 검증을 마쳤음을 감사 기록으로 남깁니다.


