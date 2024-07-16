import pandas as pd
import numpy as np

#p.99

# 데이터 탐색 함수
# head() : 데이터의 앞부분
# tail() : 데이터의 끝부분
# shape
# info()
# describe()

exam = pd.read_csv("data/exam.csv")
exam.head
exam.tail(5)
exam.shape
exam.info()
exam.describe()

type(exam)
var=[1,2,3]
type(var)
exam.head()
# var.head()

#p.113 변수명 바꾸기
exam2 = exam.copy()
exam2 = exam2.rename(columns = {"nclass" : "class"})
exam2.head()

#p.116 파생변수 만들기
exam2["total"] = exam2["math"] + exam2["english"] + exam2["science"]
exam2.head()

#p.120 합격 판정 변수 만들기
exam2["test"] = np.where(exam2["total"] >= 200, 'pass', 'fail')
exam2.head()

exam2["test"].value_counts()

#p.122 막대 그래프로 빈도 표현하기
import matplotlib.pyplot as plt
count_test = exam2["test"].value_counts()
count_test.plot.bar(rot=0)
plt.show()
plt.clf()


#200 이상: A
#100 이상: B
#100 미만: C
exam2["test2"] = np.where(exam2["total"] >= 200, 'A',
                np.where(exam2["total"] >= 100, 'B', 'C'))

exam2.head()

#p.128 따라해보기
exam2["test2"].isin(["A"])


#ch6 - p.132

#데이터 전처리 함수
# query()
# df[]
# sort_values()
# groupby()
# assign()
# agg()
# merge()
# concat()

# 조건에 맞는 행을 걸러내는 .query()
# exam[exam["nclass"] == 1]
exam.query('nclass == 1')

exam.query('nclass != 1')
exam.query('math > 50')
exam.query('math < 50')
exam.query('english <= 80')
exam.query('nclass == 2 & math >= 80')
exam.query('nclass == 5 and math >= 80')
exam.query('nclass == 1 | nclass == 5')
exam.query('nclass == 1 or nclass == 4')
exam.query('nclass in [1, 3, 5]')
exam.query('nclass not in [1, 3, 5]')


exam["nclass"]
exam[["nclass"]]
exam[["id","nclass"]]

#p.147 변수 제거하기
exam.drop(columns = "math")
exam.drop(columns = ["math", "english"])
exam

#p.148
exam.query("nclass == 1")[["math", "english"]]

#가독성 있게 코드 줄 바꾸기
exam.query("nclass == 1") \
    [["math", "english"]] \
    .head()


#p.151 정렬하기
exam.sort_values("math")
exam.sort_values("math", ascending = False) #오름차순
exam.sort_values(["nclass", "english"], ascending = [True, False])

#p.154  파생변수 추가
exam = exam.assign(
    total = exam["math"] + exam["science"] + exam["english"],
    mean = (exam["math"] + exam["science"] + exam["english"])/3
    ) \
    .sort_values("total", ascending = False) #여러 함수를 동시에 추가 가능
exam.head()


#p.157 lambda 함수 사용하기
exam2 = pd.read_csv("data/exam.csv")

exam2 = exam2.assign(
    total = lambda x: x["math"] + x["science"] + x["english"],
    mean = lambda x: x["total"]/3 #앞에서 만든 함수 바로 사용 가능
    ) \
    .sort_values("total", ascending = False)
exam.head()

#p.159 그룹을 나눠 요약을 하는 .groupby() + .agg() 콤보
exam2.agg(mean_math = ('math', "mean"))
exam2.groupby("nclass") \
     .agg(mean_math = ('math', "mean"))

#반별 과목 평균 구하기
exam2.groupby("nclass") \
     .agg(
         mean_math = ('math', "mean"),
         mean_eng = ('english', "mean"),
         mean_sci = ('science', "mean"
         )


#p.165
import pandas as pd
from pydataset import data

# 데이터셋 로드
df = data("mpg")

# 'class' 열이 'compact'인 행들을 필터링
compact_df = df[df['class'] == 'compact']

# 결과 출력
print(compact_df)

