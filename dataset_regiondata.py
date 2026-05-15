import pandas as pd
import os

path = r"C:\Users\nhunk\OneDrive\Desktop\2026-1학기\V모듈\팀프로젝트\Teamproject_machine_learning_team02\(주거파트 데이터)"
os.chdir(path)
file_name = '아파트(매매)_실거래가_20260515153221'

try:
    df = pd.read_csv(file_name, encoding='cp949', skiprows=15)
except UnicodeDecodeError:
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
    '건물노후도': 'mean',
    '단지명': 'count',
    '평형구분': lambda x: (x == '대형').sum() / len(x) * 100
}).reset_index()

summary.columns = ['읍면동', '평균건물노후도', '아파트단지밀집도', '대형평수비율(%)']

output_name = '천안아산_주거특성_최종요약.csv'
summary.to_csv(output_name, index=False, encoding='utf-8-sig')

print(f"{output_name} 파일생성")
print(summary.head(10))