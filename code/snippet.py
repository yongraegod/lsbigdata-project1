import numpy as np

# -----------------------------

def g(x=3):
    result = x + 1
    return result

g()

# 함수의 내용 확인
import inspect
print(inspect.getsource(g)) # F12 누르면 함수가 정의된 곳으로 이동함!

# -----------------------------

# if...else 정식
x = 3
if x > 4:
    y = 1
else:
    y = 2
print(y)

# if else 축약
y = 1 if x > 4 else 2

# -----------------------------

# 리스트 컴프리헨션
x = [1, -2, 3, -4, 5]
result = ["양수" if value > 0 else "음수" for value in x]
print(result)

# -----------------------------

# 조건 3개 이상의 경우 elif() 사용
x = 0
if x > 0:
    result = "양수"
elif x == 0:
    result = "0"
else:
    result = "음수"
print(result)

# -----------------------------

# 조건 3가지 넘파이 ver.
import numpy as np
x = np.array([1, -2, 3, -4, 0])
conditions = [
    x > 0, x == 0, x < 0
]
choices = [
    "양수", "0", "음수"
]
result = np.select(conditions, choices, x)
print(result)

# -----------------------------

# for loop
for i in range(1, 4):
    print(f"Here is {i}")

# for loop 리스트 컴프리헨션
print([f"Here is {i}" for i in range(1, 4)])

# -----------------------------

name = "John"
age = 30
greeting = f"Hello, my name is {name} and I am {age} years old."
print(greeting)

import numpy as np
names = ["John", "Alice"]
ages = np.array([25, 30])

greetings = [f"Hello, my name is {name} and I am {age} years old." for name, age
in zip(names, ages)]

for greeting in greetings:
    print(greeting)

# -----------------------------

# zip 함수
import numpy as np

names = ["John", "Alice"]
ages = np.array([25, 30])

# zip() 함수로 names와 ages를 병렬적으로 묶음
zipped = zip(names, ages)

# 각 튜플을 출력
for name, age in zipped:
    print(f"Name: {name}, Age: {age}")

# -----------------------------

# while
i = 0
while i <= 10:
    i += 3
    print(i)

# while & break 루프 구문
i = 0
while True:
    i += 3
    if i > 10:
        break
    print(i)

# -----------------------------
# apply 함수 이해하기

import pandas as pd
data = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
    })
data

data.apply(max, axis=0)
data.apply(max, axis=1)

# 사용자 함수 apply에 적용하기
def my_func(x, const=3):
    return max(x)**2 + const

my_func([3,4,10], 5)

data.apply(my_func, axis=1, const=5)

# 넘파이(Numpy) 배열에 apply 함수 적용
import numpy as np
array_2d = np.arange(1, 13).reshape((3, 4), order='F')
print(array_2d)

np.apply_along_axis(max, axis=0, arr=array_2d)

# -----------------------------

# 함수 환경

y = 2

def my_func(x):
    global y
    y = y + 1
    result = x + y
    return result
# 함수 선언할 때는 함수가 실행되는 건 아님

my_func(3) # global y로 지정하니 my_func() 돌릴때 마다 y가 1씩 증가함
print(y)

# 함수 안에 함수를 또 만들 수 있음
def my_func(x):
    global y

    def my_f(k):
        return k**2

    y = my_f(x) + 1
    result = x + y

    return result

my_f(3) # my_func() 안에 선언되어 있기에 밖에서는 부를 수가 없다!!

my_func(3) # 13
print(y) # 10

# -----------------------------

# 입력값이 몇 개일지 모를땐 별표(*)를 붙임
def add_many(*args): 
    result = 0 
    for i in args: 
        result = result + i   # *args에 입력받은 모든 값을 더한다.
    return result

add_many(1,2,3)


def first_many(*args): 
    return args[0]

first_many(1,2,3)
first_many(4,1,2,3)

# -----------------------------

def add_mul(choice, *my_input): 
    if choice == "add":   # 매개변수 choice에 "add"를 입력받았을 때
        result = 0 
        for i in my_input: 
            result = result + i 
    elif choice == "mul":   # 매개변수 choice에 "mul"을 입력받았을 때
        result = 1 
        for i in my_input: 
            result = result * i 
    return result 

add_mul("add", 5,4,3,1)
add_mul("mul", 5,4,3,1)

# -----------------------------

# 별표 두개 (**)는 입력값을 딕셔너리로 만들어줌!
def my_twostars(choice, **kwargs):
    if choice == "first":
        return print(kwargs["age"])
    elif choice == "second":
        return print(kwargs["name"])
    else:
        return print(kwargs)
    
my_twostars("first", age=30, name="issac")
my_twostars("second", age=30, name="issac")
my_twostars("all", age=30, name="issac")

dict_a = {'age': 30, 'name': 'issac'}
dict_a["age"]
dict_a["name"]