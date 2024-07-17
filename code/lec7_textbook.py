# p.167 데이터 합치기
import pandas as pd
import numpy as np

#5명 학생의 중간고사 데이터 만들기
test1 = pd.DataFrame({'id'      : [1,2,3,4,5],
                      'midterm' : [60,80,70,90,85]})

test2 = pd.DataFrame({'id'      : [1,2,3,4,5],
                      'final'   : [70, 83, 65, 95, 80]})

test1
test2

# Left Join
total = pd.merge(test1, test2, how="left", on = "id")
total

# Right Join
total = pd.merge(test1, test2, how="right", on = "id")
total


# Inner Join - 공통으로 가지고 있는 것만
total = pd.merge(test1, test2, how="inner", on = "id")
total

# Outer Join - 모두 다 있음
total = pd.merge(test1, test2, how="outer", on = "id")
total


# p.169
#exam = pd.read_csv("data/exam.csv")
name = pd.DataFrame({'nclass'  : [1,2,3,4,5],
                     'teacher' : ["kim", "lee", "park", "choi", "jung"]})
name

pd.merge(exam, name, how="left", on = "nclass")


# 데이터를 세로로 쌓는 방법
score1 = pd.DataFrame({'id'     : [1,2,3,4,5],
                      'score' : [60,80,70,90,85]})

score2 = pd.DataFrame({'id'     : [6,7,8,9,10],
                      'score'   : [70, 83, 65, 95, 80]})
score1
score2

pd.concat([score1, score2])

test1
test2
pd.concat([test1, test2], axis=1)


# p.178 결측치
df = pd.DataFrame({'sex'   : ['M', 'F', np.nan, 'M', 'F'],
                   'score' : [5, 4, 3, 4, np.nan]})
df

df["score"] + 1

pd.isna(df) #np.nan 값 자리에만 True 출력
pd.isna(df).sum() #결측치 몇개 있는지 확인


#결측치 제거
df.dropna(subset = 'score') #score 결측치 제거
df.dropna(subset = ['score', 'sex']) #score, sex 결측치 제거
df.dropna() #묻지도 따지지 않고 결측치 다 제거


#p.183 결측치 대체
exam = pd.read_csv("data/exam.csv")

#데이터 프레임 location을 사용한 인덱싱
# exam.loc[행 인덱스, 열 인덱스]
exam.loc[[0], ["id", "nclass"]]
exam.iloc[0:2, 0:5]

type(exam)

exam.loc[[2,7,14], ['math']] = np.nan #2,7,14 행의 'math'열을 nan으로 만들기
exam.iloc[[2,7,14], 2] = 3 #2,7,14 행의 2열을 3으로 만들기


#수학점수 50점 이하인 학생들 점수를 50점으로 상향 조정하기! (loc 사용)
exam.loc[exam["math"] <= 50, "math"] = 50
exam

#영어점수 90점 이상인 학생의 점수를 90으로 하향 조정하기! (iloc 사용)
exam.iloc[exam["english"] >= 90, 3]                    #실행 안됨
exam.iloc[np.array(exam["english"] >= 90), 3]          #실행 됨
exam.iloc[np.where(exam["english"] >= 90)[0], 3] = 90  #np.where도 튜플이라 [0] 사용해서 꺼내오면 됨
exam.iloc[exam[exam["english"] >= 90].index, 3] = 90   #index 벡터도 작동


# math 50점 이하를 "-"로 변경
exam = pd.read_csv("data/exam.csv")
exam.loc[exam["math"] <= 50 , "math"] = "-"
exam

# "-" 결측치를 수학점수 평균으로 바꾸고 싶다!

# 1.
math_mean = exam.loc[(exam["math"] != "-"), "math"].mean()
exam.loc[exam["math"] == '-', "math"] = math_mean
exam

# 2.
math_mean = exam.query('math not in ["-"]')['math'].mean()
exam.loc[exam['math'] == '-', 'math'] = math_mean
exam

# 3.
math_mean = exam[exam["math"] != "-"]["math"].mean()
exam.loc[exam["math"] == '-', "math"] = math_mean
exam

# 4.
exam.loc[exam["math"] == '-', ["math"]] = np.nan
math_mean = exam["math"].mean()
exam.loc[pd.isna(exam['math']), ['math']] = math_mean

# 5.
# np.nanmean() : nan 무시 함수
vector = np.nanmean(np.array([np.nan if x == '-' else float(x) for x in exam["math"]]))
vector = np.nanmean(np.array([float(x) if x != '-' else np.nan for x in exam["math"]]))
exam["math"] = np.where(exam["math"] == "-", vector, exam["math"])
exam

# 6. 
math_mean = exam[exam["math"] != "-"]["math"].mean()
exam["math"] = exam["math"].replace("-", math_mean)
exam

df.loc[df["score"] == 3.0, ["score"]] = 4
df


















