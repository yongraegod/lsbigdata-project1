# 강사님의 houseprice_3.py 부분임||
==================================
# 회귀분석을 활용해서 house_price 예측하기~~!!

import pandas as pd
import numpy as np
# adp교재 p.139

# 직선의 방정식
# y = ax + b
# 예) y = 2x+3의 그래프를 그려보세요!
a = 2
b = 3

x = np.linspace(-5, 5, 100)
y = a * x + b

plt.plot(x, y, color='blue')
plt.axvline(0, color='black') # 수직(vertical)선 그리기
plt.axhline(0, color='black') # 수평(horizontal)선 그리기
plt.xlim(-5,5)
plt.ylim(-5,5)
plt.show()
plt.clf()

#house_price 데이터 불러와서 해보기
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family' : 'Malgun Gothic'})

plt.clf()

a = 53
b = 45

x = np.linspace(0, 5, 100)
y = a * x + b
plt.plot(x, y, color='blue')

house_train = pd.read_csv("data/house_price/train.csv")
my_df = house_train[['BedroomAbvGr','SalePrice']].head(10)
my_df['SalePrice'] = my_df['SalePrice'] / 1000
plt.scatter(x=my_df['BedroomAbvGr'], y=my_df['SalePrice'])

plt.xlabel("방 개수", fontsize=10)
plt.ylabel("가격", fontsize=10)
plt.show()

------------------------------------
# test 집 정보 가져오기 
house_test = pd.read_csv("data/house_price/test.csv")
a=53; b=45
(a * house_test['BedroomAbvGr'] + b) * 1000

#sub 데이터 불러오기
sub_df = pd.read_csv("data/house_price/sample_submission.csv")

# SalePrice 바꿔치기
sub_df['SalePrice'] = (a * house_test['BedroomAbvGr'] + b) * 1000
sub_df

#파일 바꿔치기: csv로 내보내기
sub_df.to_csv("data/house_price/sample_submission4.csv", index = False)
------------------------------------
# 직선 성능 평가
# house_train = pd.read_csv("data/house_price/train.csv")
a = 53
b = 45

# y_hat을 어떻게 구할까?
y_hat = (a * house_train['BedroomAbvGr'] + b) * 1000

# y는 어디에 있는가?
y = house_train['SalePrice']

# (y - y_hat)의 절대값 구하기 
np.abs(y - y_hat) # 절대값(절대거리)
# np.sum((y - y_hat)**2) # 제곱합

np.sum(np.abs(y - y_hat))
======================================
!pip install scikit-learn

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 예시 데이터 (x와 y 벡터)
x = np.array([1, 3, 2, 1, 5]).reshape(-1, 1)  # x 벡터 (특성 벡터는 2차원 배열이어야 합니다)
y = np.array([1, 2, 3, 4, 5])  # y 벡터 (레이블 벡터는 1차원 배열입니다)

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y) # 자동으로 기울기, 절편 값을 구해준다

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

slope = model.coef_[0]
intercept = model.intercept_
print(f"기울기 (slope a): {slope}")
print(f"절편 (intercept b): {intercept}")

# 예측값 계산
y_pred = model.predict(x)

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='data')
plt.plot(x, y_pred, color='red', label='regression line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()
===============================================================
# 회귀모델을 통한 집값 예측

#필요한 패키지 불러오기
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 필요한 데이터 불러오기
house_train = pd.read_csv("data/house_price/train.csv")
house_test = pd.read_csv("data/house_price/test.csv")
sub_df = pd.read_csv("data/house_price/sample_submission.csv")

# 회귀분석 적합(fit)하기
x = np.array(house_train['BedroomAbvGr']).reshape(-1,1)
y = house_train['SalePrice'].values / 1000

# # 데이터프레임에서 .values 속성을 사용하면, 
# # 데이터프레임의 데이터를 NumPy 배열 형식으로 변환
# x = house_train['BedroomAbvGr'].values.reshape(-1, 1)
# y = house_train['SalePrice'].values / 1000

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y) # 자동으로 기울기, 절편 값을 구해준다

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

slope = model.coef_[0]
intercept = model.intercept_
print(f"기울기 (slope a): {slope}")
print(f"절편 (intercept b): {intercept}")

# 예측값 계산
y_pred = model.predict(x)

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='data')
plt.plot(x, y_pred, color='red', label='regression line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()

test_x = np.array(house_test['BedroomAbvGr']).reshape(-1,1)
test_x

pred_y = model.predict(test_x) #test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기
sub_df['SalePrice'] = pred_y * 1000
sub_df

#파일 바꿔치기: csv로 내보내기
sub_df.to_csv("data/house_price/sample_submission6.csv", index = False)
=========================================
# ====================
# =====   옵션   =====
# ====================

import numpy as np
from scipy.optimize import minimize

# 최소값을 찾을 다변수 함수 정의
def my_f(x):
    return (x[0] - 1) ** 2 + (x[1] - 2) ** 2

# 초기 추정값
initial_guess = [0, 0]

# 최소값 찾기
result = minimize(my_f, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)


# 회귀직선 구하기
import numpy as np
from scipy.optimize import minimize

def line_perform(par):
    y_hat=(par[0] * house_train["BedroomAbvGr"] + par[1]) * 1000
    y=house_train["SalePrice"]
    return np.sum(np.abs((y-y_hat)))

line_perform([36, 68])

# 초기 추정값
initial_guess = [0, 0]

# 최소값 찾기
result = minimize(line_perform, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)

--------------------------
# test 집 정보 가져오기 
house_test = pd.read_csv("data/house_price/test.csv")
a=16.38101698; b=133.96602049739172
(a * house_test['BedroomAbvGr'] + b)

#sub 데이터 불러오기
sub_df = pd.read_csv("data/house_price/sample_submission.csv")

# SalePrice 바꿔치기
sub_df['SalePrice'] = (a * house_test['BedroomAbvGr'] + b) * 1000
sub_df

#파일 바꿔치기: csv로 내보내기
sub_df.to_csv("data/house_price/sample_submission5.csv", index = False)
-----------------------------

# ====================
# ==옵션에 대한 설명==
# ====================

import numpy as np
from scipy.optimize import minimize

# 최솟값을 찾을 다변수 함수 정의
def my_f(x):
    return x**2 +3

my_f(3)

# 초기 추정값
initial_guess = [0]

# 최솟값 찾기
result = minimize(my_f, initial_guess)

# 결과 출력
print("최솟값:", result.fun)
print("최솟값을 갖는 x 값:", result.x)

-------------------------------
# z = x^2 + y^2 +3
# minimize가 x를 리스트 형식으로 받아오기 때문에 y를 사용할 수 없음.
# 그래서 x[0], x[1]과 같이 사용!

def my_f2(x):
    return x[0]**2 + x[1]**2 +3

my_f2([1, 3])

# 초기 추정값
initial_guess = [-10,3]

# 최솟값 찾기
result = minimize(my_f2, initial_guess)

# 결과 출력
print("최솟값:", result.fun)
print("최솟값을 갖는 x 값:", result.x)
----------------------------------
# f(x,y,z) = (x-1)^2 + (y-2)^2 + (z-4)^2 + 7
def my_f3(x):
    return (x[0]-1)**2 + (x[1]-2)**2 + (x[2]-4)**2 + 7

my_f2([1,2,3])

# 초기 추정값
initial_guess = [-10,3,4]

# 최솟값 찾기
result = minimize(my_f3, initial_guess)

# 결과 출력
print("최솟값:", result.fun)
print("최솟값을 갖는 x 값:", result.x)

=================================
===============================================================
# 회귀모델을 통한 집값 예측

#필요한 패키지 불러오기
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 필요한 데이터 불러오기
house_train = pd.read_csv("data/house_price/train.csv")
house_test = pd.read_csv("data/house_price/test.csv")
sub_df = pd.read_csv("data/house_price/sample_submission.csv")

# 회귀분석 적합(fit)하기
x = np.array(house_train['BedroomAbvGr']).reshape(-1,1)
y = house_train['SalePrice'].values / 1000

# # 데이터프레임에서 .values 속성을 사용하면, 
# # 데이터프레임의 데이터를 NumPy 배열 형식으로 변환
# x = house_train['BedroomAbvGr'].values.reshape(-1, 1)
# y = house_train['SalePrice'].values / 1000

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y) # 자동으로 기울기, 절편 값을 구해준다

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

slope = model.coef_[0]
intercept = model.intercept_
print(f"기울기 (slope a): {slope}")
print(f"절편 (intercept b): {intercept}")

# 예측값 계산
y_pred = model.predict(x)

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='data')
plt.plot(x, y_pred, color='red', label='regression line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()

test_x = np.array(house_test['BedroomAbvGr']).reshape(-1,1)
test_x

pred_y = model.predict(test_x) #test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기
sub_df['SalePrice'] = pred_y * 1000
sub_df

#파일 바꿔치기: csv로 내보내기
sub_df.to_csv("data/house_price/sample_submission6.csv", index = False)
