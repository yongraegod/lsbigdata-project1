# 수학함수
import math
x=4
math.sqrt(x)

#지수 계산
exp_val = math.exp(5)
exp_val

#로그 계산
log_val = math.log(10, 10)
log_val

#팩토리얼 계산
fact_val = math.factorial(5)
fact_val


def my_normal_pdf(x, mu, sigma):
  part_1 = (sigma * math.sqrt(2 * math.pi))**-1
  part_2 = math.exp((-(x-mu)**2) / (2*sigma**2))
  return part_1 * part_2

my_normal_pdf(3,3,1)

def normal_pdf(x, mu, sigma):
  sqrt_two_pi = math.sqrt(2 * math.pi)
  factor = 1 / (sigma * sqrt_two_pi)
  return factor * math.exp(-0.5 * ((x - mu) / sigma) ** 2)

import math

def my_function(x, y, z):
    return (x**2 + math.sqrt(y) + math.sin(z)) * math.exp(x)

   
def my_g(x):
    return math.cos(x) + math.sin(x) * math.exp(x)
      
my_g(math.pi)


def fname(input):
    contents
    return
    
import pandas as pd

import numpy as np


#Numpy
#Ctrl + Shift + c : 커멘트 처리
#!pip install numpy
import pandas as pd
import numpy as np
   
# 벡터 생성하기 예제
a = np.array([1, 2, 3, 4, 5]) # 숫자형 벡터 생성
b = np.array(["apple", "banana", "orange"]) # 문자형 벡터 생성
c = np.array([True, False, True, True]) # 논리형 벡터 생성
print("Numeric Vector:", a)
print("String Vector:", b)
print("Boolean Vector:", c)

type(a) #numpy에서 새로 만든 데이터 타입
a[3]
a[2:]
a[1:4]

# 빈 배열 생성
b = np.empty(3)
b
print("빈 벡터 생성하기:", x)

# 배열 채우기
b[0] = 1
b[1] = 7
b[2] = 358
print("채워진 벡터:", x)
b
b[2]

#np.arrange()
vec1 = np.arange(100)
vec1
vec2 = np.arange(1,101)
vec2
vec3 = np.arange(1,100.2, 0.5)
vec3

# -100부터 0까지
vec5=np.arange(-100,1)
vec5

vec7= -np.arange(0,101)
vec7

# 0부터 -100까지
vec6 = np.arange(0,-101,-1)
vec6

#np.linspace()
l_space1 = np.linspace(0, 1, 5)
l_space1

l_space2 = np.linspace(0, 1, 5, endpoint = False)
l_space2


#np.repeat() vs np.tile()
np.repeat(3, 5)

vec4=np.arange(5)
np.repeat(vec4, 5)

np.tile(vec4, 3)

vec4 +vec4 #같은 위치끼리 연산
max(vec4)
sum(vec4)

# Q. 35672 이하 홀수들의 합은?
vec_odd = np.arange(1,35673,2)
vec_odd
sum(vec_odd)

sum(np.arange(1, 35673, 2))
np.arange(1, 35673, 2).sum()

#len 함수 사용하기
len(vec_odd)
vec_odd.shape


# 2차원 배열
b = np.array([[1, 2, 3], [4, 5, 6]])
length = len(b) # 첫 번째 차원의 길이
shape = b.shape # 각 차원의 크기
size = b.size # 전체 요소의 개수
length, shape, size


#Broadcast
a=np.array([1,2])
b=np.array([1,2,3,4])

a+b #둘의 길이가 다르기에 연산 불가

np.tile(a,2) + b #길이가 맞으니깐 연산 가능
np.repeat(a,2) + b

b == 3

#Q.35672 보다 작은 수 중에서 7로 나눠서 나머지가 3인 숫자들의 갯수는??
c=np.arange(0, 35673)
answer = (c % 7) ==3
sum(answer)

sum(np.arange(1, 35672) % 7 == 3)

#Q.10보다 작은 수 중에서 7로 나눠서 나머지가 3인 숫자들의 갯수는??
sum((np.arange(1,10) % 7) == 3)

#1차원 배열 생성
a = np.array([1.0, 2.0, 3.0])
b = 2.0
a * b

a.shape
b.shape

# 2차원 배열 생성
matrix = np.array([[ 0.0, 0.0, 0.0],
                   [10.0, 10.0, 10.0],
                   [20.0, 20.0, 20.0],
                   [30.0, 30.0, 30.0]])
matrix.shape
# 1차원 배열 생성
vector = np.array([1.0, 2.0, 3.0])
vector.shape
# 브로드캐스팅을 이용한 배열 덧셈
result = matrix + vector
print("브로드캐스팅 결과:\n", result)


import numpy as np
# 2차원 배열 생성
matrix = np.array([[ 0.0, 0.0, 0.0],
                   [10.0, 10.0, 10.0],
                   [20.0, 20.0, 20.0],
                   [30.0, 30.0, 30.0]])
# 벡터 생성
vector = np.array([1.0, 2.0, 3.0, 4.0])
# 브로드캐스팅을 이용한 배열 덧셈
result = matrix + vector
print("브로드캐스팅 결과:\n", result)

vector = np.array([1.0,2.0,3.0,4.0]).reshape(4,1)
vector
vector.shape
result = matrix + vector
result












