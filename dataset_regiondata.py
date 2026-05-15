import pandas as pd
import os

target_path = r"C:\Users\nhunk\OneDrive\Desktop\2026-1학기\V모듈\팀프로젝트\Teamproject_machine_learning_team02\(지역파트 데이터)"

if os.path.exists(target_path):
    os.chdir(target_path)
    print(f"현재 작업 폴더 설정 완료: {os.getcwd()}")
else:
    print(f"[경로 오류] 다음 경로를 찾을 수 없습니다: {target_path}")
    print("탐색기에서 해당 폴더 주소를 다시 복사해서 붙여넣어 보세요.")

file_name = '아파트(매매)_실거래가_20260515153221.csv'

if not os.path.exists(file_name):
    print(f"[파일 오류] '{file_name}' 파일이 설정한 폴더 안에 없습니다.")
    print(f"현재 폴더 내 실제 파일들: {os.listdir('.')}")
else:
    print(f"파일을 찾았습니다. 분석을 시작합니다.")
    
    try:
        df = pd.read_csv(file_name, encoding='cp949', skiprows=15)
    except:
        df = pd.read_csv(file_name, encoding='utf-8-sig', skiprows=15)

    df_ta = df[df['시군구'].str.contains('천안|아산', na=False)].copy()

    df_ta['읍면동'] = df_ta['시군구'].apply(lambda x: x.split()[-1])

    df_ta['건물노후도'] = 2026 - df_ta['건축년도']

    def get_size_type(area):
        if area <= 60: return '소형'
        elif area <= 85: return '중형'
        else: return '대형'

    df_ta['평형구분'] = df_ta['전용면적(㎡)'].apply(get_size_type)

    summary = df_ta.groupby('읍면동').agg({
        '건물노후도': 'mean',          # 평균 나이
        '단지명': 'count',             # 아파트 거래 활성도(밀집도)
        '평형구분': lambda x: (x == '대형').sum() / len(x) * 100 # 대형 평수 비중
    }).reset_index()

    summary.columns = ['읍면동', '평균건물노후도', '아파트단지밀집도', '대형평수비율(%)']
    output_name = '천안아산_주거특성_최종요약.csv'
    summary.to_csv(output_name, index=False, encoding='utf-8-sig')

    print(f"\n'{output_name}' 파일이 해당 폴더에 생성되었습니다.")
    print(summary.head(10))