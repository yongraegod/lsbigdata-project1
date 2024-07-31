# p.225 chap.09 한국복지패널 데이터
!pip install pyreadstat

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 불러오기
raw_welfare = pd.read_spss('data/koweps/Koweps_hpwc14_2019_beta2.sav')

# 복사본 만들기
welfare = raw_welfare.copy()
welfare.shape
welfare.info()
# welfare.describe()

# 변수명 바꾸기
welfare = welfare.rename(
    columns = {'h14_g3'     : 'sex',
               'h14_g4'     : 'birth',
               'h14_g10'    : 'marriage_type',
               'h14_g11'    : 'religion',
               'p1402_8aq1' : 'income',
               'h14_eco9'   : 'code_job',
               'h14_reg7'   : 'code_region'}
)

# 분석에 필요한 부분만 추출해서 저장
welfare = welfare[['sex','birth','marriage_type','religion',
                   'income','code_job','code_region']]

# 변수 검토하기
welfare['sex'].dtypes
welfare['sex'].value_counts()
# welfare['sex'].isna().sum()

# 성별 항목에 이름 부여
welfare['sex'] = np.where(welfare['sex'] == 1, 'male', 'female')
welfare['sex'].value_counts()
============================
# 월급 변수 검토
# welfare['income'].describe()
welfare['income'].isna().sum()

# 성별 월급 평균표 만들기
sex_income = welfare.dropna(subset = 'income') \
                    .groupby('sex', as_index = False) \
                    .agg(mean_income = ('income', 'mean'))

sns.barplot(data = sex_income, x = 'sex', y = 'mean_income', hue = 'sex')
plt.show()
plt.clf()
============================
# 나이와 월급의 관계

welfare['birth'].describe()
sns.histplot(data = welfare, x = 'birth')
plt.show()
plt.clf()

welfare['birth'].isna().sum()

welfare = welfare.assign(age = 2019 - welfare['birth'] + 1)
welfare['age']
sns.histplot(data = welfare, x = 'age')
plt.show()
plt.clf()

age_income = welfare.dropna(subset = 'income') \
                    .groupby('age', as_index = False) \
                    .agg(mean_income = ('income', 'mean'))

sns.lineplot(data = age_income, x = 'age', y = 'mean_income')
plt.show()
plt.clf()

# 나이별 income 칼럼 na 개수 세기!
a = welfare.assign(income_na = welfare['income'].isna()) \
           .groupby('age', as_index = False) \
           .agg(n = ('income_na', 'sum'))
a
sns.barplot(data = a, x = 'age', y = 'n')
plt.show()
plt.clf()
=============================
# 연령대에 따른 월급 차이
welfare['age'].head()
welfare = welfare.assign(ageg = np.where(welfare['age'] < 30, 'young',
                                np.where(welfare['age'] <=59, 'middle',
                                                              'old')))
welfare['ageg']

sns.countplot(data = welfare, x = 'ageg', hue = 'ageg')
plt.show()
plt.clf()

ageg_income = welfare.dropna(subset = 'income') \
                     .groupby('ageg', as_index = False) \
                     .agg(mean_income = ('income', 'mean'))

sns.barplot(data = ageg_income, x = 'ageg', y = 'mean_income', hue = 'ageg',
            order = ['young', 'middle', 'old'])
plt.show()
plt.clf()
------------------------------------------
# # 응용하기! 0~9, 10~19, 20~29, ...
# welfare['age'].head()
# welfare = welfare.assign(new_ageg = np.where(welfare['age'] < 10, '0s',
#                                     np.where(welfare['age'] < 20, '10s',
#                                     np.where(welfare['age'] < 30, '20s',
#                                     np.where(welfare['age'] < 40, '30s',
#                                     np.where(welfare['age'] < 50, '40s',
#                                     np.where(welfare['age'] < 60, '50s',
#                                     np.where(welfare['age'] < 70, '60s',
#                                     np.where(welfare['age'] < 80, '70s',
#                                     np.where(welfare['age'] < 90, '80s',
#                                     np.where(welfare['age'] < 100, '90s',
#                                     np.where(welfare['age'] < 110, '100s','110s'
#                                     ))))))))))))
# 
# welfare['new_ageg']
# ------------------------------------------
# # 나이대 범주를 정의(gpt 사용)
# bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, np.inf]
# labels = ['0s', '10s', '20s', '30s', '40s', '50s', '60s', '70s', '80s', '90s', '100s']
# 
# # cut 함수로 나이대를 범주화
# welfare['new_ageg'] = pd.cut(welfare['age'], bins=bins, labels=labels, right=False)
# 
# # 결과 확인
# print(welfare[['age', 'new_ageg']])
------------------------------------------
# 나이대별 수입 분석(수업에서 사용)
# plt.rcParams.update({'font.family' : 'Malgun Gothic'}) # 한글 폰트 추가
# cut 사용
bin_cut = np.array([0,9,19,29,39,49,59,69,79,89,99,109,119])
welfare = welfare.assign(age_group = pd.cut(welfare['age'],
                         bins = bin_cut,
                         labels = (np.arange(12) * 10).astype(str) + '대'))

age_income = welfare.dropna(subset = 'income') \
                    .groupby('age_group', as_index = False) \
                    .agg(mean_income = ('income', 'mean'))

sns.barplot(data = age_income, x = 'age_group', y = 'mean_income')

plt.show()
plt.clf()
----------------------------------------------
# 판다스 데이터 프레임을 다룰 때, 변수의 타입이 카테고리로 설정되어 있는 경우,
# groupby + agg 콤보가 안먹힘.
# 그래서 object 타입으로 바꿔 준 후 수행
welfare['age_group'] = welfare['age_group'].astype('object')

sex_age_income = welfare.dropna(subset = 'income') \
                        .groupby(['age_group', 'sex'], as_index = False) \
                        .agg(mean_income = ('income','mean'))

sns.barplot(data = sex_age_income, x = 'age_group', y = 'mean_income', hue ='sex')
plt.show()
plt.clf()
----------------------------------------------
# 연령대별, 성별 상위 4% 수입 찾아보기!
# quantile
# x = np.arange(10)
# np.quantile(x, q=0.96) 96% 위치 값을 찾아줌

sex_age_income = welfare.dropna(subset = 'income') \
                        .groupby(['age_group', 'sex'], as_index = False) \
                        .agg(top4per_income = ('income', lambda x: np.quantile(x, q=0.96)))
sex_age_income

sns.barplot(data = sex_age_income, x = 'age_group', y = 'top4per_income', hue ='sex')
plt.show()
plt.clf()
===============================================
# 9-6장
welfare['code_job']
welfare['code_job'].value_counts()

# !pip install openpyxl
list_job = pd.read_excel("data/koweps/Koweps_Codebook_2019.xlsx", sheet_name = '직종코드')
list_job.head()

# welfare에 list_job 결합하기
welfare = welfare.merge(list_job, how = 'left', on = 'code_job')

welfare.dropna(subset = ['job', 'income'])[['income', 'job']]

# 직업별 월급 평균표 만들기
job_income = welfare.dropna(subset = ['job', 'income']) \
                    .groupby('job', as_index = False) \
                    .agg(mean_income = ('income','mean'))
job_income.head()

# 직업별 월급 평균 Top 10 추출
top10 = job_income.sort_values('mean_income', ascending = False).head(10)
top10

# 국문 폰트 설정
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family' : 'Malgun Gothic'})

# 막대 그래프 만들기
sns.barplot(data = top10, x='mean_income', y='job', hue='job')
plt.show()
plt.clf()
-----------------------------------------------
# 통으로 하기
df = welfare.dropna(subset=['job','income']) \
            .query('sex=="female"') \              #query도 사용 가능함!!(9-7장 내용)
            .groupby('job', as_index = False) \
            .agg(mean_income = ('income','mean')) \
            .sort_values('mean_income', ascending = False) \
            .head(10)
            
# 막대 그래프 만들기
sns.barplot(df, x='mean_income', y='job', hue='job')
plt.tight_layout()
plt.show()
plt.clf()            
===============================================
## 9-8장(p.263)
welfare.info()
welfare["marriage_type"]
df = welfare.query("marriage_type != 5") \
            .groupby('religion', as_index = False) \
            ["marriage_type"] \
            .value_counts(normalize=True) # <- proportion을 구해줌!!(핵심)
df
-----------------------------------------------
df.query('marriage_type == 1') \
  .assign(proportion = df['proportion'] * 100) \
  .round(1)
























