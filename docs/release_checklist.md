# Repository Release Checklist (docs/release_checklist.md)

본 문서는 프로젝트를 GitHub 공개 저장소에 최종 푸시(`git push`)하기 전, 대외 무결성과 보안 안전을 보증하기 위해 수행해야 하는 전수 점검 체크리스트입니다.

---

## 1. GitHub 릴리즈 무결성 체크리스트

| 점검 영역 | 상세 확인 항목 | 검토 결과 | 판정 |
| :--- | :--- | :---: | :---: |
| **인증 정보** | 소스코드 및 문서 내 카카오/KOSIS API Key 원문이 전부 제거되었는가? | 마스킹/환경변수 분리 완료 | **적합** |
| **비밀 파일** | `.env` 파일이 루트에 노출되어 있지 않으며 `.gitignore`에 등록되었는가? | `.gitignore` 적용 완료 | **적합** |
| **경로 지정** | 코드와 기술 문서 내에 `C:\Users\` 및 `file:///` 절대 경로가 없는가? | 상대 경로로 수정 완료 | **적합** |
| **대용량 파일** | `data/raw/` 및 기가바이트 규모의 원천 데이터 폴더가 업로드 대상에서 배제되었는가? | gitignore 및 아카이브 격리 | **적합** |
| **결과서 위치** | 최종 V8 보고서 및 V5 논문이 `deliverables/` 하위로 격리 배치되었는가? | `deliverables/`로 이관 완료 | **적합** |
| **기술 문서** | `docs/` 하위가 결과서가 아닌 개발자용 기술문서 13종으로 교체 완료되었는가? | 13종 수록 완료 (현재 폴더) | **적합** |
| **런타임 캐시** | 소스 내 `__pycache__/` 및 `*.pyc` 런타임 빌드 캐시 파일이 소거되었는가? | gitignore 제외 완료 | **적합** |
| **임시 파일** | 과거 레거시 스크립트, draft 보고서 및 GITHUB_RELEASE_AUDIT/ 등이 아카이브 이관되었는가? | `_archive_before_github/` 이관 | **적합** |

---

## 2. 릴리즈 직전 확인용 CLI 명령어 시퀀스

Git 저장소 add/commit 전, 아래 명령을 터미널에 가동하여 예외 규칙 상태를 최종 모니터링하십시오.

```bash
# 1. ignore 상태의 비밀 파일 검증 (출력이 나오면 정상 차단 상태)
git check-ignore -v .env

# 2. raw 데이터 디렉토리 제외 검증
git check-ignore -v data/raw

# 3. 아카이브 디렉토리 격리 검증
git check-ignore -v _archive_before_github/

# 4. python 빌드 캐시 배제 검증
git check-ignore -v __pycache__
```


