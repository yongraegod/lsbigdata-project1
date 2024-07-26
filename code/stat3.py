import numpy as np
from scipy.stats import bernoulli

!pip install scipy

# 확률질량함수(pmf)
# 확률변수가 갖는 값에 해당하는 확률을 저장하고 있는 함수
# bernoulli.pmf(k, p)
# P(X=1)
bernoulli.pmf(1, 0.3)
# P(X=0)
bernoulli.pmf(0, 0.3)


# 이항분포 X ~ P(X = k | n, p)
# n: 베르누이 확률변수 더한 갯수
# p: 1이 나올 확률
# binom.pmf(k, n, p)

from scipy.stats import binom
binom.pmf(0, n=2, p=0.3)
binom.pmf(1, n=2, p=0.3)
binom.pmf(2, n=2, p=0.3)


# X ~ B(n,p)
binom.pmf(np.arange(31), n=30, p=0.3) #리스트 comp.
[binom.pmf(x, n=30, p=0.3) for x in range(31)] #numpy 사용

import math
math.factorial(54) / (math.factorial(26) * math.factorial(28))
math.comb(54, 26)

================몰라도 됨=====================================
# np.cumprod(np.arange(1, 55))[-1]
# ln
log(a * b) = log(a) + log(b)
log(1 * 2 * 3 * 4) = log(1) + log(2) + log(3) + log(4)

math.log(24)
sum(np.log(np.arange(1,5)))

math.log(math.factorial(54))
log_54f = sum(np.log(np.arange(1,55)))
log_26f = sum(np.log(np.arange(1,27)))
log_28f = sum(np.log(np.arange(1,29)))
log_54f - (log_26f + log_28f)

# math.comb(54, 26)
np.exp(log_54f - (log_26f + log_28f))

math.comb(2,0) * 0.3**0 *(1-0.3)**2
math.comb(2,1) * 0.3**1 *(1-0.3)**1
math.comb(2,2) * 0.3**2 *(1-0.3)**0
=============================================================
# pmf: probability mass fuction(확률질량함수)
binom.pmf(0, 2, 0.3)
binom.pmf(1, 2, 0.3)
binom.pmf(2, 2, 0.3)

## Quiz_1. X ~ B(n=10, p=0.36)
# P(X=4)
binom.pmf(4, 10, 0.36)

# P(X<=4)
binom.pmf(np.arange(5), 10, 0.36).sum()

# P(2<X<=8)
binom.pmf(np.arange(3,9), 10, 0.36).sum()

## Quiz_2. X ~ B(n=30, p=0.2)
# P(X<4 or 25<=X)
# 1) 1 - P(4<=X<25)
1 - binom.pmf(np.arange(4,25), 30, 0.2).sum()

# 2) P(X<4) + P(X>=25)
binom.pmf(np.arange(4), 30, 0.2).sum() + \
binom.pmf(np.arange(25,31), 30, 0.2).sum()
=============================================================
# rvs 함수 (random variates sample)
# 표본 추출 함수
# X1 ~ Bernoulli(p=0.3)
bernoulli.rvs(p=0.3)
# X2 ~ Bernoulli(p=0.3)
bernoulli.rvs(p=0.3)
# X ~ B(n=2, p=0.3)
bernoulli.rvs(p=0.3) + bernoulli.rvs(p=0.3)
binom.rvs(n=2, p=0.3, size=1)

binom.pmf(0, n=2, p=0.3)
binom.pmf(1, n=2, p=0.3)
binom.pmf(2, n=2, p=0.3)

# Quiz~!
# X ~ B(30, 0.26) | 표본 30개를 뽑아보자~
binom.rvs(n=30, p=0.26, size=30)
# E[X] = n * p
-----------------------------------------------------
# Quiz~!
# X ~ B(30, 0.26)을 시각화!
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
-----------------------------------------------------
# 그래프 그리기_1 (ChatGPT)
# 파라미터 설정
n = 30  # 시행 횟수
p = 0.26  # 성공 확률

# X 값 범위 설정
x = np.arange(0, n+1)

# 이항 분포의 PMF 계산
pmf = binom.pmf(x, n, p)

# 그래프 그리기
plt.figure(figsize=(20, 6))
plt.bar(x, pmf, color='skyblue')
plt.xlabel('Number of Successes')
plt.ylabel('Probability')
plt.title('Binomial Distribution B(30, 0.26)')
plt.xticks(np.arange(0, n+1, step=2))
plt.grid(axis='y')
plt.show()
plt.clf()
---------------------------------------------
# 그래프 그리기_2
import seaborn as sns
prob_x =  binom.pmf(np.arange(31), n=30, p=0.26)
sns.barplot(prob_x)
plt.show()
plt.clf()
---------------------------------------------
# 그래프 그리기_3 (교재 p.207)
import pandas as pd
x = np.arange(31)
prob_x =  binom.pmf(np.arange(31), n=30, p=0.26)
df = pd.DataFrame({"x": x, "prob": prob_x})
df

sns.barplot(data = df, x = "x", y = "prob")
plt.show()
===============================================
# cdf: cumulative dsit. fuction
# 누적확률분포 함수
# F_X(x) = P(X <= x)

# Quiz. P(X<=4)
binom.cdf(4, n=30, p=0.26)
 
# Quiz. P(4<X<=18)
binom.cdf(18, n=30, p=0.26) - binom.cdf(4, n=30, p=0.26) 

# Quiz. P(13<X<20)
binom.cdf(19, n=30, p=0.26) - binom.cdf(13, n=30, p=0.26)
--------------

import numpy as np
import seaborn as sns

x_1 = binom.rvs(n=30, p=0.26, size=10)
binom.rvs(n=2, p=0.3, size=3)
x_1
x=np.arange(31)
prob_x = binom.pmf(x, n=30, p=0.26)
sns.barplot(prob_x, color="yellow")

# Add a point
plt.scatter(x_1, np.repeat(0.002,10), color='red', zorder=10, s=5)

# 기댓값 표현
plt.axvline(7.8, color='green', linestyle ='--', linewidth=2)

plt.show()
plt.clf()
===================================
# 퀀타일 함수(ppf)
# P(X < ?) = 0.5 : ?를 찾는 함수
# X~B(n=30, p=0.26)
binom.ppf(0.5, n=30, p=0.26)
binom.cdf(8, n=30, p=0.26)
binom.cdf(7, n=30, p=0.26)

# P(X<?) = 0.7
binom.ppf(0.7, n=30, p=0.26)
binom.cdf(9, n=30, p=0.26)
binom.cdf(8, n=30, p=0.26)
===================================
## norm = normal distribution(정규분포)
## 𝑓(𝑥; 𝜇, 𝜎)

#ex1. 𝑥 = 0, 𝜇 = 0, 𝜎 = 1
1 / np.sqrt(2 * math.pi) # 0.3989...

from scipy.stats import norm #norm.pdf를 사용해보자!
norm.pdf(0, loc=0, scale=1) #loc 뮤, scale 시그마, 위와 같은 값이 나옴!!

# ex2. 𝑥 = 5, 𝜇 = 3, 𝜎 = 4
norm.pdf(5, loc=3, scale=4)

# ex3. 정규분포 pdf 그리기
k = np.linspace(-5, 5, 100)
y = norm.pdf(k, loc=0, scale=1)

plt.plot(k, y, color='black')
plt.scatter(k, y, color='red', s=0.5)
plt.show()
plt.clf()

# 𝜇(loc, 뮤, 평균): 분포의 `중심`을 결정하는 모수(특징을 결정하는 수)
k = np.linspace(-5, 5, 100)
y = norm.pdf(k, loc=0, scale=1)

plt.plot(k, y, color='black')
plt.show()
plt.clf()

# 𝜎(scale, 시그마, 표준편차): 분포의 `퍼짐`을 결정하는 모수
k = np.linspace(-5, 5, 100)
y = norm.pdf(k, loc=0, scale=1)
y2 = norm.pdf(k, loc=0, scale=2)
y3 = norm.pdf(k, loc=0, scale=0.5)

plt.plot(k, y, color='red')
plt.plot(k, y2, color='blue')
plt.plot(k, y3, color='yellow')
plt.show()
plt.clf()

# P(X<=0)
norm.cdf(0, loc=0, scale=1)
# P(X<=100)
norm.cdf(100, loc=0, scale=1)

# P(-2<X<0.54)
norm.cdf(0.54, loc=0, scale=1) - norm.cdf(-2, loc=0, scale=1)

# P(X<1 or X>3)
# 1 - P(1<=X<=3) 
1 - (norm.cdf(3, loc=0, scale=1) - norm.cdf(1, loc=0, scale=1))
-------------------------------------------
## X~N(3,5^2) | P(3<X<5) = ? = 15.54...%
norm.cdf(5, 3, 5) - norm.cdf(3, 3, 5)

# 위 확률변수에서 표본 100개 뽑기
x = norm.rvs(loc=3, scale=5, size=1000)
sum((x > 3) & (x < 5))/1000 # 15.7% !!!

# ex) 평균:0, 표준편차:1
# 표본 1000개 뽑아서 0보다 작은 비율 확인
x = norm.rvs(loc=0, scale=1, size=1000)
np.mean(x < 0) # 48.6%
(x < 0).mean()
------------------------------------------

x = norm.rvs(loc=3, scale=2, size=1000)
x

sns.histplot(x, stat="density")

# plot the normal distribution PDF
xmin, xmax = (x.min(), x.max())
x_values = np.linspace(xmin, xmax, 100)
pdf_values = norm.pdf(x_values, loc=3, scale=2)
plt.plot(x_values, pdf_values, color='red', linewidth = 2)

plt.show()
plt.clf()

