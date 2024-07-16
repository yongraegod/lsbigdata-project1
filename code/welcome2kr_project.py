import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# (단기체류외국인) 월별 단기체류외국인 국적(지역)별 현황
# 단기체류 외국인 : 관광, 친지방문 등의 목적으로 입국, 90일 이내에서 단기간 국내 체류
df = pd.read_csv('data/foreigner.csv',  encoding='euc-kr')
df

# df_new로 복사본 생성
df_new = df.copy() 
df_new

#항목 열 이름을 영문으로 변경
df_new = df_new.rename(columns={
    '년': 'year',
    '월': 'month',
    '국적지역': 'nationality',
    '단기체류외국인 수': 'visitors'
                            })
df_new


#2022년에 어떤 나라가 우리나라를 많이 방문할까?? Top 5! 구해보기~

#2022년으로 데이터 한정하기
df_2022 = df_new.query('year == 2022')
df_2022

#
df_new.sort_values(['year','visitors'], ascending=[False, False])

#2022~2024 나라별 방문자 수 합친 새로운 열 생성
# 나라별 visitors 합계 계산
df_country_totals = df_exam2.groupby('nationality')['visitors'].sum().reset_index()

# 새로운 열 추가
df_exam2['country_totals'] = df_exam2['nationality'].map(df_country_totals.set_index('nationality')['visitors'])
df_exam2

#평균방문자
mean_visitors = df_exam2["country_totals"].mean()
mean_visitors
# 평균보다 많은 방문자를 갖고 있는 국가들 중 5국가 선택

alot_of_visitors = df_exam2[df_exam2['country_totals'] > mean_visitors]['nationality']
alot_of_visitors.head(5)


