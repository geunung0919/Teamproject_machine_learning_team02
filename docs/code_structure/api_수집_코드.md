# API 수집 코드

`src/api`는 외부 API와 공개 데이터를 호출해서 원천 데이터를 저장하는 폴더다.  
대부분 결과는 `data/raw` 또는 `data/processed`에 CSV로 저장된다.

## 파일별 역할

### `check_eduinfo_availability.py`

교육정보 API에서 어떤 endpoint가 실제로 호출 가능한지 확인하는 테스트 코드다.

주요 함수:

- `request_json(session, path, data, referer)`: API에 POST 요청을 보내 JSON 응답을 받는다.
- `compact(row, keys)`: 응답에서 확인할 핵심 컬럼만 추린다.
- `main()`: 여러 endpoint를 테스트하고 응답 가능 여부를 출력한다.

### `collect_eduinfo_data.py`

현재 학교 목록과 폐교 학교 데이터를 수집한다.  
학교명, 학교급, 주소, 좌표 등 지도와 모델의 학교 단위 기본 데이터가 여기서 시작된다.

주요 함수:

- `request_json(session, path, data, referer)`: 교육정보 API 요청 공통 함수다.
- `collect_closed_schools(session)`: 폐교 학교 목록을 수집한다.
- `collect_current_school_list(session, sido_code)`: 현재 운영 중인 학교 목록을 수집한다.
- `collect_school_detail(...)`: 학교별 상세 정보를 수집한다.
- `main()`: 전국 현재학교/폐교학교 데이터를 저장한다.

### `collect_national_kosis_population.py`

KOSIS 인구 데이터를 수집해 시군구별 학령인구 피처를 만든다.

주요 함수:

- `get_json(params, timeout)`: KOSIS API 요청 공통 함수다.
- `discover_sgg_codes(api_key)`: 수집 가능한 시군구 코드를 조회한다.
- `collect_population(...)`: 시군구 인구 데이터를 수집한다.
- `collect_population_chunk(...)`: 많은 지역을 나누어 호출한다.
- `build_features(raw, codes)`: 0~19세, 15~19세 등 학령인구 관련 피처를 만든다.
- `main()`: 인구 원천 데이터와 가공 피처를 저장한다.

### `collect_national_kosis_fertility.py`

합계출산율과 연령별 출산율 데이터를 수집한다.

주요 함수:

- `get_json(params, timeout)`: KOSIS API 요청 공통 함수다.
- `collect_fertility(api_key, start, end)`: 시군구 합계출산율을 수집한다.
- `collect_age_specific_fertility(api_key, start, end)`: 연령별 출산율을 수집한다.
- `build_fertility_features(fertility)`: 출산율 증감률 피처를 만든다.
- `main()`: 출산율 원천/가공 데이터를 저장한다.

### `collect_national_kosis_birth_migration.py`

출생아수와 인구이동 데이터를 수집한다.  
장기 학령인구 시나리오와 학령수요 감소압력 점수에 중요하게 쓰인다.

주요 함수:

- `get_json(params, timeout)`: KOSIS API 요청 공통 함수다.
- `collect_birth(api_key, start, end)`: 출생아수 데이터를 수집한다.
- `build_birth_features(raw)`: 출생아수 증감률 피처를 만든다.
- `collect_migration(api_key, start, end)`: 순이동/전입/전출 데이터를 수집한다.
- `build_migration_features(raw)`: 순이동 proxy와 이동률 피처를 만든다.
- `main()`: 출생/이동 데이터를 저장한다.

### `collect_national_small_shop.py`

소상공인 상가업소 데이터를 전국 단위로 수집한다.  
학교 주변 상권 취약도 계산의 입력 데이터다.

주요 함수:

- `get_json(url, timeout)`: 상가업소 API 요청 함수다.
- `collect_sido(service_key, sido_code, sido_name, num_rows)`: 시도별 상가업소 데이터를 수집한다.
- `get_total_count(service_key, sido_code)`: 시도별 전체 업소 수를 확인한다.
- `build_sgg_summary(frame)`: 시군구별 상권 요약 피처를 만든다.
- `main()`: 시도별 원천 상권 CSV와 시군구 요약 파일을 저장한다.

