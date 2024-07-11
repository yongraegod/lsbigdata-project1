#ctrl + enter
#shift + 화살표: 블록

a=1
a

#파워쉘 명령어 리스트
# ls: 파일 목록
# cd: 폴더 이동
#. 현재폴더
#.. 상위폴더

#show folder in new window: 해당위치 탐색기
#Tab/shift tab 자동완성/옵션선택
#cls = clear 화면정리

a=14

a = "안녕하세요!"
a

a = '안녕~!'
a
a = '"안녕하세요!"라고 아빠가 말했다.'
a

a = [1, 2,3]
a

var1 = [1,2,3]
var2 = [4,5,6]
var1 + var2


a = "안녕하세요!"
b = "LS빅데이터 스쿨~!"

a+b
a+" "+b

print(var1)
print(a)

1num =3 
num1=3
print(num1)
num2=5
num1+num2


a=10
b=3.3
print("a + b=", a+b)
print("a - b =", a-b)
print("a * b =", a*b)
print("a / b =", a/b)
print("a % b =", a%b)
print("a // b =", a//b)
print("a ** b =", a**b)

a+b
(a**3) // 7
(a**3) & 7

a == b
a != b
a < b
a > b
a <= b
a >= b

#2에 4승과 12453을 7로 나눈 몫을 더해서 8로 나눴을 때 나머지
a = ((2**4) + (12453 // 7)) % 8
a

#9의 7승을 12로 나누고, 36452를 253로 나눈 나머지에 곱한 수
b = ((9 ** 7) / 12) * (36452 % 253)
b

#중 큰 것은??
a < b

user_age = 25
is_adult = user_age >= 18
print("성인입니까?", is_adult)

# False = 3
# TRUE = 2

true = (12,3,5,7)

a = "True"
b = TRUE
c = true
d = True

c
b

# True, False
a = True
b = False

a and b
a or b

# True : 1 / Fasle : 0
True + True
True + False
False + False

# and 연산자
True and False
True and True
False and False
False and True

True  * False
True  * True
False * False
False * True

# or 연산자
True  or True
True  or False
False or True
False or False

a = False
b = False
a or b
min(a + b, 1)

#복합 대입 연산자
a = 3
a += 10
# a = a + 10
a

a -= 4
a

a %= 3 
a

a += 12
a

a **= 2
a

a/= 7
a


# 문자열 변수 할당
str1 = "Hello! "

# 문자열 반복
repeated_str = str1 * 3
print("Repeated string:", repeated_str)

str1 * 2.5

# 정수: int(eger)
# 실수: float (double)

#단항 연산자
x = 5
+x
-x
~x

#binary
x=0
~x
bin(0)
bin(3)

var1
sum(var1)

#책 p.71 패키지 설치
pip install pydataset

import pydataset
pydataset.data()

df = pydataset.data("AirPassengers")
df

import pandas as pd
pd
df = pd.DataFrame({'제품명': ['머그컵'],
                   '원산지': ['국산'],
                   '판매가': [10000]
                   }); df

import numpy as np
np

#테스트~~d hhh
