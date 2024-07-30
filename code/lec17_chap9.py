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
welfare.describe()

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
welfare['income'].describe()
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
# ------------------------------------------
# 나이대별 수입 분석(수업에서 사용)
# cut
bin_cut = np.array([0,9,19,29,39,49,59,69,79,89,99,109,119])
welfare = welfare.assign(age_group = pd.cut(welfare['age'],
                         bins = bin_cut,
                         labels = (np.arange(12) * 10).astype(str) + '대'))

age_income = welfare.dropna(subset = 'income') \
                    .groupby('age_group', as_index = False) \
                    .agg(mean_income = ('income', 'mean'))

sns.barplot(data = age_income, x = 'age_group', y = 'mean_income')

plt.rcParams.update({'font.family' : 'Malgun Gothic'}) # 한글 폰트 추가

plt.show()
plt.clf()
===============================================
# 연령대 및 성별 월급 차이
sex_income = welfare.dropna(subset = 'income') \
                    .groupby(['ageg', 'sex'], as_index = False) \
                    .agg(mean_income = ('income','mean'))
sex_income

sns.barplot(data = sex_income, x = 'ageg', y = 'mean_income', hue ='sex',\
            order = ['young','middle','old'])
plt.show()
plt.clf()
