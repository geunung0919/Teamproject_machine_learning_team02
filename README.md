# 학교별 학생수 예측 및 적정규모학교 점검 시나리오 생성 프로젝트

## 프로젝트 소개

이 저장소는 학교별 학생 수 감소 압력을 예측하고, 적정규모학교 점검을 위한 2026-2030 시나리오를 생성한 최종 v5 제출본입니다.

GitHub 공개 목적은 최종 결과물 확인 및 교수 평가입니다. 전체 원본 데이터와 중간 산출물을 포함한 재학습용 연구 저장소가 아니라, 최종 분석 코드, 문서, 평가 결과, 보고서, 논문, 정적 웹 결과물을 정리한 공개 패키지입니다.

## 프로젝트 목표

- 학교별 학생수 예측
- 감소압력 분석
- 적정규모학교 점검 시나리오 생성
- 최종 결과 웹 시각화

## 데이터

본 프로젝트는 학교알리미, KOSIS 등 공공데이터 기반으로 구축되었습니다.

원본 데이터와 중간 데이터는 용량, 라이선스, 공개 범위, 개인정보 및 위치정보 검토 문제로 GitHub에 포함하지 않습니다. 공개 저장소에는 최종 결과 확인에 필요한 정적 웹 데이터, 최종 평가표, 문서, 보고서, 논문, 코드만 포함합니다.

## 최종 결과 확인

최종 웹 결과물:

```text
web/index.html
```

확인 방법:

1. 저장소를 clone하거나 ZIP으로 내려받습니다.
2. `web/index.html`을 브라우저에서 엽니다.
3. 별도의 `npm install`이나 Vite 빌드는 필요하지 않습니다.

이 저장소는 최종 정적 웹 결과물을 포함합니다. Vite/React 원본 프로젝트가 아니라 빌드 완료된 HTML 결과물을 공개합니다.

## 폴더 구조

```text
assets/          최종 보고서와 문서에 사용한 다이어그램 소스
data/sample/     공개 가능한 소형 샘플 데이터
deliverables/    최종 논문, 최종 보고서, 보조 문서, 이미지
docs/            프로젝트 구조, 모델, 데이터, 평가, 시나리오 설명 문서
outputs/figures/ 최종 보고서 및 평가용 그림
outputs/metrics/ 최종 성능 비교표와 평가 결과
public/data/     최종 웹 대시보드용 정적 JSON/CSV 데이터
src/             v5 분석 파이프라인 및 시나리오 생성 코드
web/             최종 정적 웹 결과물
```

## 실행

### 최종 웹 확인

```text
web/index.html
```

브라우저에서 직접 열어 최종 시각화 결과를 확인합니다.

### Python 환경

코드 검토 또는 일부 스크립트 실행을 위한 기본 설치:

```bash
pip install -r requirements.txt
```

주요 엔트리포인트:

```bash
python src/pipeline/run_v5_full_pipeline.py --stage all --horizon 1
python src/pipeline/run_v5_model_only.py --horizon 1
python src/pipeline/run_v5_web_export_only.py
```

단, 위 파이프라인은 원본 데이터와 중간 데이터가 준비된 내부 환경을 전제로 합니다. 공개 저장소만으로 전체 재학습 및 전체 시나리오 재생성은 제공하지 않습니다.

## 환경 변수

API 수집 스크립트를 내부 환경에서 사용할 경우 `.env.example`을 `.env`로 복사한 뒤 필요한 키를 채웁니다.

```text
KOSIS_API_KEY=
EDU_INFO_API_KEY=
DATAGOKR_API_KEY=
KAKAO_API_KEY=
```

`.env`는 공개 저장소에 포함하지 않습니다.

## 최종 산출물

- 최종 웹 결과: `web/index.html`
- 최종 웹 데이터: `public/data/scenario_v5_v2/`
- 최종 성능 비교표: `outputs/metrics/v5_final_r3_audit_and_web_filters/final_candidate_model_comparison.csv`
- 최종 평가 결과: `outputs/metrics/v5_final_r3_audit_and_web_filters/`
- 최종 논문: `deliverables/paper/paper_html_v5/`
- 최종 보고서: `deliverables/report/`

## 한계

본 저장소는 최종 결과물 공개용 저장소입니다.

전체 데이터 재학습 환경은 제공하지 않습니다. 원본 데이터, 중간 데이터, 대용량 실험 산출물, v0-v4 과거 실험 버전은 GitHub 공개 대상에서 제외했습니다.

## 라이선스

MIT License. 자세한 내용은 `LICENSE`를 참고하십시오.


