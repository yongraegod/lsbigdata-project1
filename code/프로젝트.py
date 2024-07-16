import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#(단기체류외국인) 월별 단기체류외국인 국적(지역)별 현황


# 0. 데이터 불러오기
df = pd.read_csv('data/foreigner.csv',  encoding='euc-kr')
df.columns


# 1.변수 영어로 변경
df = df.rename(columns={
                         '년': 'year',
                         '월': 'month',
                         '국적지역': 'nation',
                         '단기체류외국인 수': 'visitors'
                        })

df.head()


# 2. 2023년도만 뽑아오기
df_23 = df.query("year == 2023")
df_23.head()

# 월 삭제
df_23 = df_23.drop(columns = 'month')
df_23.head()


# 3. 23년도 방문자 평균값 확인
avg_23 = df_23['visitors'].mean()
avg_23


# 4. (월 제외) 23년도 국가별 방문자 수 구하기
df_23 = df_23.groupby('nation') \
             .agg(sum_visitor = ('visitors', 'sum'))
print(df_23, type(df_23))


# 5. avg_23(23년도 방문자 평균)과 비교하기
df_23['compare'] = np.where(df_23['sum_visitor'] >= 2962.0, 'high', 'low')
df_23

# 내림차순 & 상위 값 5개 추출
df_23.sort_values('sum_visitor', ascending = False).head(5)
