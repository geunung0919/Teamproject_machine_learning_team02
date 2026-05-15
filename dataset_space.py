import pandas as pd
import os

path = r"C:\Users\nhunk\OneDrive\Desktop\2026-1학기\V모듈\팀프로젝트\Teamproject_machine_learning_team02\(공간파트 데이터)"
os.chdir(path)
filename = '충청남도교육청_충남학교현황_20250301.csv'

try:
    df_school = pd.read_csv(filename, encoding='utf-8-sig')
except UnicodeDecodeError:
    df_school = pd.read_csv(filename, encoding='cp949')

df_filtered = df_school[df_school['시군명'].str.contains('천안|아산', na=False)].copy()

print(f"--- 필터링 완료 ---")
print(f"전체 학교 수: {len(df_school)}개 -> 천안/아산 학교 수: {len(df_filtered)}개")

print(df_filtered[['시군명', '읍면동', '학교명', '학생수']].head())

output_filename = '천안아산_학교현황_2025.csv'
df_filtered.to_csv(output_filename, index=False, encoding='utf-8-sig')

print(f"\n'{output_filename}' 파일로 저장되었습니다.")