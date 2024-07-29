# 교재 8장, p.212, 선그래프

import pandas as pd
import numpy as np
import seaborn as sns

#economics 데이터 불러오기
economics = pd.read_csv('data/economics.csv')
economics.head()

economics.info() #date는 object => 문자열, 범주형 정보

sns.lineplot(data = economics, x = 'date', y = 'unemploy')
plt.show()

# 날짜 시간 타입 변수 만들기
economics['date2'] = pd.to_datetime(economics['date'])
economics.info()

economics[['date','date2']]

# 연,월,일 등 추출
economics['date2'].dt.year #연
economics['date2'].dt.month #월
economics['date2'].dt.day #일
economics['date2'].dt.month_name() #월의 이름
economics['date2'].dt.quarter #분기
economics['date2'].dt.day_name() #요일

economics['quarter'] = economics['date2'].dt.quarter
economics[['date2','quarter']]

# 날짜에 연산하기
economics['date2'] + pd.DateOffset(days=30) # 일자에 덧셈
economics['date2'] + pd.DateOffset(months=-1) #월에 뺄셈
economics['date2'] + pd.DateOffset(years=1) #연에 덧셈

#윤년 체크
economics['date2'].dt.is_leap_year

economics['year'] = economics['date2'].dt.year

sns.lineplot(data = economics, x = 'year', y = 'unemploy')
sns.scatterplot(data = economics, x = 'year', y = 'unemploy', s=2)
sns.lineplot(data = economics, x = 'year', y = 'unemploy',
             errorbar = None) #신뢰구간 제거
plt.show()
plt.clf()
--------------------------------------------------------------
# 연도별 표본평균, 표준편차 구하기
my_df = economics.groupby('year', as_index = False) \
                 .agg(mean_year = ('unemploy','mean'),
                      std_year = ('unemploy', 'std'),
                      n_year = ('unemploy', 'count'))
my_df['mean_year']

# 신뢰구간 구하기
my_df['left_ci']  = my_df['mean_year'] - (1.96 * my_df['std_year'] / np.sqrt(my_df['n_year']))
my_df['right_ci'] = my_df['mean_year'] + (1.96 * my_df['std_year'] / np.sqrt(my_df['n_year']))
my_df.head()

import matplotlib.pyplot as plt
x = my_df['year']
y = my_df['mean_year']
plt.plot(x,y,color='black')
plt.scatter(x, my_df['left_ci'], color='blue', s=1.5)
plt.scatter(x, my_df['right_ci'], color='red', s=1.5)

plt.show()
plt.clf()
===============================================================
