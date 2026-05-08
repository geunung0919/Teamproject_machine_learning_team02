import pandas as pd
import os

path = r"C:\Users\nhunk\OneDrive\Desktop\2026-1학기\V모듈\팀프로젝트\Teamproject_machine_learning_team02\(지역파트 데이터)"
os.chdir(path)

cols = ['시군구명', '행정동명', '상권업종중분류명', '상권업종소분류명', '상호명']
df = pd.read_csv('소상공인시장진흥공단_상가(상권)정보_충남_202603.csv', encoding='utf-8', usecols=cols)

#천안/아산 지역 필터링
df_target = df[df['시군구명'].str.contains('천안|아산', na=False)].copy()

#Kids(어린이) Silver(어르신)
kids_keywords = '소아과|키즈|문구|어린이집|유치원|학원|소아청소년과'
silver_keywords = '요양원|주간보호|투석|노인|실버|재활저널|요양병원'

df_target['category'] = '기타'
df_target.loc[df_target['상권업종소분류명'].str.contains(kids_keywords, na=False) | 
              df_target['상호명'].str.contains(kids_keywords, na=False), 'category'] = 'Kids'

df_target.loc[df_target['상권업종소분류명'].str.contains(silver_keywords, na=False) | 
              df_target['상호명'].str.contains(silver_keywords, na=False), 'category'] = 'Silver'

analysis = df_target.groupby(['시군구명', '행정동명', 'category']).size().unstack(fill_value=0)

if 'Kids' not in analysis.columns: analysis['Kids'] = 0
if 'Silver' not in analysis.columns: analysis['Silver'] = 0

#지수가 높을수록 고령화된 상권 (Silver 시설이 Kids보다 많은 동네) 분모가 0이 되는 것을 방지하기 위해 +1
analysis['상권교체지수'] = (analysis['Silver'] + 1) / (analysis['Kids'] + 1)

analysis.reset_index(inplace=True)
analysis.to_csv('천안아산_상권교체지수_결과.csv', index=False, encoding='utf-8-sig')

print(analysis.head(100))