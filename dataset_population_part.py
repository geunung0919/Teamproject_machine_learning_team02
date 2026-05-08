import pandas as pd
import glob
import os

path = r"C:\Users\nhunk\OneDrive\Desktop\2026-1학기\V모듈\팀프로젝트\Teamproject_machine_learning_team02\(인구파트 데이터)"
os.chdir(path)

all_files= glob.glob("*.csv")

li = []
for filename in all_files:
    df = pd.read_csv(filename, index_col= None, header = 0, encoding='cp949')
    li.append(df)

frame = pd.concat(li, axis = 0, ignore_index=True)

frame.to_csv("통합_인구데이터_2016_2026.csv", index=False, encoding='utf-8')

print(f"총 {len(all_files)}개의 파일 합치기")