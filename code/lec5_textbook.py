#교재 p.81
import numpy as np
import pandas as pd

pd.DataFrame()

df = pd.DataFrame({'name'   : ['김지훈','이유진','박동현','김민지'],
                   'english' : [90,80,60,70],
                   'math'    : [60,60,100,20]})
df                   

type(df)
df['name']

sum(df["english"])

#p.84 문제

df_1 = pd.DataFrame({'제품'   : ['사과','딸기','수박'],
                     '가격'   : [1800,1500,3000],
                     '판매량' : [24,38,13]})
df_1                    

sum(df_1["가격"]) / 3
sum(df_1["판매량"]) / 3

#p.85
!pip install openpyxl

df_exam = pd.read_excel('data/excel_exam.xlsx')
df_exam

sum(df_exam['math'])    / 20
sum(df_exam['english']) / 20
sum(df_exam['science']) / 20

len(df_exam)
df_exam.shape
df_exam.size

#없는 column도 만들 수 있음
df_exam['total'] = df_exam['math'] + df_exam['english'] + df_exam['science']
df_exam['mean'] = df_exam['total'] / 3
df_exam

df_exam[(df_exam['math'] > 50) & (df_exam['english'] > 50)]

#수학 점수가 평균보다 높은 사람 중, 영어는 평균보다 낮은 사람을 구하시오.
mean_m = np.mean(df_exam["math"])
mean_e = np.mean(df_exam["english"])

df_exam[(df_exam["math"] > mean_m) & (df_exam["english"] < mean_e)]

df_nc3 = df_exam[df_exam["nclass"] == 3]
df_nc3[["math", "english", "science"]]
df_nc3[0:]

df_exam[::2]

#p.151 순서대로 정렬하기
df_exam.sort_values("math")
df_exam.sort_values("math", ascending = False)
df_exam.sort_values(["nclass","math"], ascending = [True,False])


np.where(a > 3) #위치를 찾아서 튜플로 반환
type(np.where(a > 3, "Up", "Down")) #array로 반환

df_exam["updown"] = np.where(df_exam["math"] > 50, "Up", "Down")


