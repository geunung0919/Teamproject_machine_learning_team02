import pandas as pd
import os

path = r"C:\Users\nhunk\OneDrive\Desktop\2026-1학기\V모듈\팀프로젝트\Teamproject_machine_learning_team02\(공간파트 데이터)"
os.chdir(path)

df = pd.read_csv('천안아산_학교현황_2025.csv', encoding='utf-8-sig')

def fill_eup_myeon_dong(row):
    if pd.isna(row['읍면동']):
        addr_parts = row['주소'].split()
        for part in addr_parts:
            if any(suffix in part for suffix in ['읍', '면', '동']):
                return part
    return row['읍면동']

df['읍면동'] = df.apply(fill_eup_myeon_dong, axis=1)
df.loc[df['학교명'] == '나사렛새꿈학교', '읍면동'] = '쌍용동'
df.loc[df['학교명'] == '천안늘해랑학교', '읍면동'] = '병천면'

print(f"정제 후 결측치 개수")
print(df['읍면동'].isnull().sum())

df.to_csv('천안아산_학교현황_2025_최종.csv', index=False, encoding='utf-8-sig')
print("\n[성공] 모든 빈칸이 채워진 '천안아산_학교현황_최종.csv' 파일이 생성되었습니다.")