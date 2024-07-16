import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#(단기체류외국인) 월별 단기체류외국인 국적(지역)별 현황
df_foreigner = pd.read_csv('data/foreigner.csv',  encoding='euc-kr')
df_foreigner.columns


#항목 열을 영문으로 변경 (1.변수 이름 변경 했는지?)
df_foreigner = df_foreigner.rename(columns={
    '년': 'year',
    '월': 'month',
    '국적지역': 'nationality',
    '단기체류외국인 수': 'visitors'
})

df_foreigner.describe()


#연도 오름차순 정렬
df_foreigner.sort_values('year')
df_foreigner.head()


#국적별 빈도수 확인
count = df_foreigner['nationality'].value_counts()
count


#2023년만의 데이터 추출
df_2023 = df_foreigner.query('year == 2023')
df2023 = (df_foreigner['year']==2023)

df_foreigner[df_2023].set_index
df_foreigner[df2023].set_index

#연도별 방문자 평균
#yr_mean = df_foreigner.groupby('year')['visitors'].mean()

yr_mean = df_foreigner.groupby('year') \
                      .agg(mean_visit = ('visitors', 'mean'))

#2023년 방문자 평균
avg_2023 = yr_mean.loc[2023]
# year_filter=df_foreigner['year']==2023
# average_visitors_2023 = df_foreigner.loc[year_filter, 'visitors'].mean()


#2023년 방문자 평균보다 더 많이 방문한 국가 상위 5개




