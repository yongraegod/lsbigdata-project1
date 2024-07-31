import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

# X ~ N(3, 7^2)

# 하위 25%에 해당하는 x는?
x = norm.ppf(0.25, loc=3, scale=7)

# 하위 25%에 해당하는 z는?
z = norm.ppf(0.25, loc=0, scale=1)

# (x - mu) / sigma == z
x == 3 + z * 7 # True
x - 3 == z * 7 # True
(x - 3) / 7 == z # True
--------------------------------
# X ~ N(3, 7^2) -> Z ~ N(0 , 1^2)
norm.cdf(5, loc=3, scale=7)
norm.cdf(2/7, loc=0, scale=1) # 둘의 값이 같음!

=================================
표본정규분포
표본 z 1000개, 히스토그램 -> pdf 겹쳐서 그리기

z = norm.rvs(loc=0, scale=1, size=1000)
x = z * np.sqrt(2) + 3 # X ~ N(3, 루트2^2)

sns.histplot(z, stat="density", color = 'grey')
sns.histplot(x, stat="density", color = 'green')

# plot the normal distribution PDF
zmin, zmax = (z.min(), x.max())
z_values = np.linspace(zmin, zmax, 100)
pdf_values = norm.pdf(z_values, loc=0, scale=1)
pdf_values2 = norm.pdf(z_values, loc=3, scale=np.sqrt(2))
plt.plot(z_values, pdf_values, color='red', linewidth = 2)
plt.plot(z_values, pdf_values2, color='blue', linewidth = 2)

plt.show()
plt.clf()
==================================
# 정규분포 X ~ N(5, 3^2)
# z = (x - 5) / 3 가 표준정규분포를 따르나요?

# 1. 표준화 확인
x = norm.rvs(loc=5, scale=3, size=1000)

# 2. 표준화
z = (x - 5) / 3
sns.histplot(z, stat="density", color = 'green')

# plot the normal distribution PDF
zmin, zmax = (z.min(), z.max())
z_values = np.linspace(zmin, zmax, 100)
pdf_values = norm.pdf(z_values, loc=0, scale=1)
plt.plot(z_values, pdf_values, color='red', linewidth = 2)

plt.show()
plt.clf()
===================================
# 정규분포 X ~ N(5, 3^2)
# 1. X 표본을 10개 뽑아서 표본분산 값 계산
# 2. X에서 표본 1000개 뽑음
# 3. 1에서 계산한 S^2으로 시그마^2을 대체한 표준화를 진행
# 표준화 : z = (X - mu) / 시그마
# 4. z의 히스토그램 그리기, 표준정규분포 pdf 확인
# 표본표준편차 나눠도 표준정규분포가 될까?
# 결론: 더이상 표준정규분포가 아니다.

# 1.
x = norm.rvs(loc=5, scale=3, size=20)
# s = np.std(x, ddof=1)
s_2 = np.var(x,ddof=1)
s

# 2.
x = norm.rvs(loc=5, scale=3, size=1000)

# 3.
# z = (x - 5) / s
z = (x - 5) / np.sqrt(s_2)
sns.histplot(z, stat="density", color = 'green')

# plot the normal distribution PDF
zmin, zmax = (z.min(), z.max())
z_values = np.linspace(zmin, zmax, 100)
pdf_values = norm.pdf(z_values, loc=0, scale=1)
plt.plot(z_values, pdf_values, color='red', linewidth = 2)

plt.show()
plt.clf()
================================
# t 분포에 대해서 알아보자!
# X ~ t(df) # df = degree of freedom
# 연속형 확률변수이고, 정규분포랑 비슷하게 생김
# 종모양, 대칭분포, 중심 0
# 모수 df: 자유도라고 부름, 분산에 영향을 미침(퍼짐을 나타내는 모수)
# df이 작으면 분산 커짐.
# df이 무한대로 가면 표준정규분포가 된다.
from scipy.stats import t

# t.pdf
# t.ppf
# t.cdf
# t.rvs

plt.rcParams.update({'font.family' : 'Malgun Gothic'})

# 자유도가 4인 t분포의 pdf를 그려보자.
t_values = np.linspace(-4, 4, 100)
pdf_values = t.pdf(t_values, df=10)
plt.plot(t_values, pdf_values, color='red', linewidth = 2, label = "t분포")

# 표준정규분포 겹치기
pdf_values = norm.pdf(t_values, loc=0, scale=1)
plt.plot(t_values, pdf_values, color='blue', linewidth = 2, label = "표준정규분포")

plt.legend()
plt.show()
plt.clf()
---------------------------
# X ~ ?(mu, sigma^2)
# X bar ~ N(mu, sigma^2/n)
# X bar ~= t(x_bar, s^2/n) 자유도가 df-1인 t 분포
---------------------------

# example
x = norm.rvs(loc=15, scale=3, size=16, random_state=42)
x
n=len(x)
x_bar = x.mean()

# 모분산을 모를때: 모평균에 대한 95% 신뢰구간을 구해보자!
x_bar + t.ppf(0.975, df=n-1) * np.std(x, ddof=1) / np.sqrt(n)
x_bar - t.ppf(0.975, df=n-1) * np.std(x, ddof=1) / np.sqrt(n)

# 모분산(3^2)을 알때: 모평균에 대한 95% 신뢰구간을 구해보자!
x_bar + norm.ppf(0.975, loc=0, scale=1) * 3 / np.sqrt(n)
x_bar - norm.ppf(0.975, loc=0, scale=1) * 3 / np.sqrt(n)











