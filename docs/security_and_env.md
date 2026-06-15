# Security and Environment Policy (docs/security_and_env.md)

본 문서는 프로젝트 소스코드의 GitHub 배포 시 로컬 보안 정보(API 자격 증명) 및 비공개 대용량 원천 데이터의 격리 처리 정책과 릴리즈 전 시행된 보안 스캔 결과를 요약 서술합니다.

---

## 1. API 자격 증명 관리 정책

* **로컬 환경 설정 파일 (.env)**:
  * Kakao Local API, KOSIS 통계 데이터 연동에 필수적인 API Key는 어떠한 경우에도 소스코드 내에 하드코딩하지 않으며, 로컬 환경의 `.env` 파일로 격리하여 보존합니다.
  * `.env` 파일은 리포지토리 루트의 `.gitignore` 규칙에 등록되어 GitHub 원격 저장소 업로드 대상에서 원천 차단됩니다.
* **환경 변수 견본 파일 (.env.example)**:
  * 대외 구동을 위해 변수명 명세만을 담은 빈 견본 템플릿 `.env.example`을 리포지토리 루트에 공개하여 로컬 설정을 조력합니다.

---

## 2. raw 및 private 데이터 제외 정책

* **제외 대상**: `data/raw/`, `data/interim/`, `data/cache/` 등
* **제외 사유**:
  1. **용량 한계**: KERIS/KOSIS로부터 적재한 14개년 학교 통계 데이터셋의 기가바이트(GB) 규모에 따른 GitHub 업로드 용량 제한 초과 방지
  2. **민감 정보 보호**: 개별 학교 주소기반 정밀 위·경도 지리 좌표, 특정 취약 소규모 학교의 정보 등 대외 공개 시 보안 위험이 발생할 수 있는 데이터 격리 처리
  3. **라이선스**: 공공데이터 API 및 통계청 데이터의 원천 재배포 라이선스 이슈 방지

---

## 3. 릴리즈 보안 스캔 결과 요약

* **점검 일시**: 2026-06-14 (최종 정리 완료 후 스캔)
* **점검 키워드**: `API_KEY`, `token`, `password`, `C:\Users\`, `file:///` 등
* **점검 결과**: **보안 무결점 적합 (SECURE)**
  * 실제 사용 중인 개인 비밀 키(Kakao API, KOSIS API) 원본 하드코딩 0건 검증 완료.
  * 소스코드 및 기술 문서 내의 개인 PC 드라이브 절대경로(`C:\Users\wltjd\...`) 완전 제거 완료. (상대경로로 변환 완료)
  * 상세 점검 결과는 루트의 [release_audit/security_scan_after.md](release_audit/security_scan_after.md) 보고서에서 확인 가능합니다.


