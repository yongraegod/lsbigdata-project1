#교재 p.81
import numpy as np
import pandas as pd

pd.DataFrame()

df = pd.DataFrame({'name '   : ['김지훈','이유진','박동현','김민지'],
                   'english' : [90,80,60,70],
                   'math'    : [60,60,100,20]})
df                   

type(df)
df["name"]

sum(df["english"])

#p.84 문제

df_1 = pd.DataFrame({'제품'   : ['사과','딸기','수박'],
                     '가격'   : [1800,1500,3000],
                     '판매량' : [24,38,13]})
df_1                    

sum(df_1["가격"]) / 3
sum(df_1["판매량"]) / 3

#p.85

