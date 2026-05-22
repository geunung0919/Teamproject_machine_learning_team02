import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rc('font', family='Malgun Gothic') 
plt.rcParams['axes.unicode_minus'] = False

df_metrics = pd.read_csv('schooldata_closure_classifier_metrics.csv')
df_panel_report = pd.read_csv('schooldata_modeling_panel_report.csv')
df_risk_2025 = pd.read_csv('schooldata_current_closure_risk_2025.csv')

plt.figure(figsize=(8, 6))

orig_rate = df_panel_report['closed_next_year_positive_rate'].iloc[0] * 100 # 약 0.98%
new_rate = df_metrics[df_metrics['model']=='tuned_histgb_schooldata_closure']['positive_rate_train'].iloc[0] * 100 # 3.43%

scarcity_data = pd.DataFrame({
    '구분': ['기존 (전체/1년)', '재정의 (후보군/3년)'],
    '폐교 발생 비율 (%)': [orig_rate, new_rate]
})

ax1 = sns.barplot(x='구분', y='폐교 발생 비율 (%)', data=scarcity_data, palette=['#d1d5db', '#ef4444'])
for p in ax1.patches:
    ax1.annotate(f'{p.get_height():.2f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=12, color='black', xytext=(0, 10), textcoords='offset points', fontweight='bold')

plt.title('데이터 희소성 극복: 문제 재정의를 통한 학습 효율화', fontsize=14, pad=15)
plt.ylim(0, 5)
plt.show()

plt.figure(figsize=(10, 6))

sns.regplot(x='radius_1_0km_all_shops', y='tuned_histgb_schooldata_closure_probability', 
            data=df_risk_2025, scatter_kws={'alpha':0.2, 'color':'gray'}, line_kws={'color':'#2563eb'})

plt.title('상권 활성화 정도와 모델 예측 폐교 위험도의 상관관계', fontsize=14)
plt.xlabel('학교 반경 1km 내 전체 상점 수', fontsize=12)
plt.ylabel('모델 예측 폐교 확률 (Probability)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(10, 7))

sns.scatterplot(x='school_isolation_score', y='tuned_histgb_schooldata_closure_probability', 
                hue='final_policy_category', data=df_risk_2025, 
                palette={'통폐합 검토 후보': '#ef4444', '교육공백 보호대상': '#756bb1', '저위험': '#2ca25f'},
                alpha=0.6)

plt.axvline(x=df_risk_2025['school_isolation_score'].mean(), color='gray', linestyle='--', alpha=0.5)
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

plt.title('고립도의 역설: 위험도와 고립도를 결합한 정책 분류', fontsize=14)
plt.xlabel('학교 고립도 점수 (Isolation Score)', fontsize=12)
plt.ylabel('모델 예측 폐교 확률 (Probability)', fontsize=12)
plt.legend(title='최종 정책 분류', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()