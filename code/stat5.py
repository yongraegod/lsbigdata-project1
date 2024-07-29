import pandas as pd
import numpy as np

np.random.seed(20240729)

old_seat = np.arange(1,29)
new_seat = np.random.choice(a,28,False)

result = pd.DataFrame({"old_seat": old_seat,
                       "new_seat": new_seat})

pd.DataFrame.to_csv(result, "result.csv")

========================================================
# y = 2x 그래프 그리기
import matplotlib.pyplot as plt

x = np.linspace(0,8,2)
y = 2 * x
plt.scatter(x,y,s=2) #원래 파이썬은 이렇게 점만 보는데
plt.plot(x,y) #plot을 사용해서 두 점을 이어줌

plt.show()
plt.clf()

# y = x^2
x = np.linspace(-8,8,100)
y = x**2
# plt.scatter(x,y,s=2,color="red")
plt.plot(x,y,color="blue")

# x축, y축 범위 설정
plt.xlim(-10,10)
plt.ylim(0,40)
plt.gca().set_aspect('equal',adjustable='box')

#비율 맞추기
# plt.axis('equal')는 xlim, ylim과 같이 사용 불가

plt.show()
plt.clf()
=========================================================
# adp책 p.57 신뢰구간 구하기

# 2) 작년 남학생 3학년 전체 분포의 표준편차는 6kg 이었다고 합니다. 
# 이 정보를 이번 년도 남학생 분포의 표준편차로 대체하여 모평균에 대한 90% 신뢰구간을 구하세요.
x = np.array([79.1, 68.8, 62.0, 74.4, 71.0, 60.6, 98.5, 86.4, 73.0, 
                 40.8, 61.2, 68.7, 61.6, 67.7, 61.7, 66.8])
x.mean()
len(x)
from scipy.stats import norm

z_005 = norm.ppf(0.95, loc=0, scale=1)

x.mean() + z_005 *6 / np.sqrt(16)
x.mean() - z_005 *6 / np.sqrt(16)

# 1) 모평균에 대한 95% 신뢰구간을 구하세요.


==========================================================
## 몬테카를로 적분: 확률변수 기대값을 구할때
# 표본을 많이 뽑은 후, 원하는 형태로 변형, 
# 평균을 계산해서 기대값을 구하는 방법

# 데이터로부터 E[X^2] 구하기
x=norm.rvs(loc=3, scale=5, size=10000)
np.mean(x**2)

# X~N(3,5^2) | E[X^2]
# E[X^2] = Var(X) + E[X]^2
sum(x**2) / (len(x)-1)
==========================================================
# E[(X-X^2)/(2X)] = ?
x=norm.rvs(loc=3, scale=5, size=100000)
# (x-x**2) / (2*x)
np.mean((x-x**2) / (2*x))
==========================================================
# X~N(3,5^2), 표본분산 S^2 구하기
np.random.seed(20240729)

x = norm.rvs(loc=3, scale=5, size=100000)
x_bar = x.mean() #x의 평균  구하기
s_2 = sum((x - x_bar)**2) / (100000-1)
s_2

np.var(x, ddof=1) # n-1로 나눈 값 (표본 분산)
# np.var(x) 사용하면 안됨 주의~!! # n으로 나눈 값

# n-1 vs n
x = norm.rvs(loc=3, scale=5, size=20)
np.var(x)
np.var(x, ddof=1)
