# 균일확률변수 만들기

import numpy as np

np.random.rand(1)

def X(i):
    return np.random.rand(i)

X(3)

# 베르누이 확률변수 모수: p 만들어보세요!
# 베르누이는 가질 수 있는 값이 2개 0, 1
def Y(p):
    x = np.random.rand(1)
    return np.where(x < p, 1, 0)
 
Y(0.5)
 

def Y(num, p):
    x = np.random.rand(num)
    return np.where(x < p, 1, 0)

#sum(Y(100, 0.5)) / 100
Y(1000000, 0.5).mean() #대수의 법칙


#새로운 확률변수 | 가질 수 있는 값: 0, 1, 2 | 확률 20%, 50%, 30%
def Z(num):
    x = np.random.rand(num)
    return np.where(x < 0.2, 0, np.where( x < 0.7, 1, 2))

Z(3)

#cumsum 사용해서 해보기
def Z(num, p):
    x = np.random.rand(num)
    p_cumsum = p.cumsum()
    return np.where(x < p_cumsum[0], 0, np.where( x < p_cumsum[1], 1, 2))
p = np.array([0.2, 0.5, 0.3])
Z(7, p)
