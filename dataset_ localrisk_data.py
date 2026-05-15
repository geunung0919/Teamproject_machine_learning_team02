import pandas as pd
import os

path = r"C:\Users\nhunk\OneDrive\Desktop\2026-1학기\V모듈\팀프로젝트\Teamproject_machine_learning_team02\(지역파트 데이터)"

if os.path.exists(path):
    os.chdir(path)
    print(f" 현재 작업 폴더: {os.getcwd()}")
else:
    print(f"[경로 오류] 폴더를 찾을 수 없습니다: {path}")

excel_file = '1. 2025년6월기준_지방소멸위험위험지수_시도_시군구_읍면동_수정.xlsx'

if not os.path.exists(excel_file):
    print(f"[파일 오류] '{excel_file}' 파일이 폴더에 없습니다.")
    print(f"현재 폴더 내 실제 파일 목록: {os.listdir('.')}")
else:
    df = pd.read_excel(excel_file, sheet_name='읍면동')
    print(f"파일을 성공적으로 불러왔습니다.")

    df_ta = df[df['sigun_nm'].str.contains('천안시|아산시', na=False)].copy()

    df_ta['연도'] = df_ta['time'].astype(str)
    df_ta['시군구명'] = df_ta['sigun_nm'].str.strip()
    df_ta['읍면동'] = df_ta['dong_nm'].str.strip()

    df_ta['소멸위험지수'] = df_ta['nidx_25'] / 100

    def get_risk_level(idx):
        if idx >= 1.5: return '소멸위험 매우 낮음'
        elif idx >= 1.0: return '보통'
        elif idx >= 0.5: return '주의'
        elif idx >= 0.2: return '소멸위험 지역'
        else: return '소멸고위험 지역'

    df_ta['소멸위험등급'] = df_ta['소멸위험지수'].apply(get_risk_level)

    # 5. 최종 결과 저장
    output_name = '천안아산_지방소멸위험지수_최종.csv'
    final_result = df_ta[['연도', '시군구명', '읍면동', '소멸위험지수', '소멸위험등급']]
    final_result.to_csv(output_name, index=False, encoding='utf-8-sig')

    print(f"\n✨ [성공] '{output_name}' 파일이 생성되었습니다.")
    print(final_result.head(10))