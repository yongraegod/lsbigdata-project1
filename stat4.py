# Uniform 균일분포 (𝑎, 𝑏) p.28
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import uniform

# Uniform에서 loc은 시작점, scale은 구간 길이
x = uniform.rvs(loc=2, scale=4, size=1)

# X ~ U(2,6)
x_values = np.linspace(0, 8, 100)
pdf_values = uniform.pdf(x_values, loc=2, scale=4)
plt.plot(x_values, pdf_values, color='red', linewidth = 2)
plt.show()
plt.clf()

# P(X<3.25) = ?
uniform.cdf(3.25,loc=2,scale=4)

# P(5<X<8.39) = ?
uniform.cdf(8.39, 2, 4) - uniform.cdf(5, 2, 4)

# 상위 7% 값은?
uniform.ppf(0.93, 2, 4)
-------------------------------------------------------
# 표본 20개를 뽑고, 표본평균을 구하시오.
x = uniform.rvs(loc=2, scale=4, size=20, random_state=42)
x.mean()

x = uniform.rvs(loc=2, scale=4, size=20*1000, random_state=42)
x = x.reshape(-1,20) # x.reshape(-1,20)
x.shape
blue_x = x.mean(axis=1)
blue_x

sns.histplot(blue_x, stat="density")
plt.show()
plt.clf()

# X bar ~ N(mu, sigma^2/n)
# X bar ~ N(4, 1.33333/20)
uniform.var(loc=2, scale=4) #분산
uniform.expect(loc=2, scale=4) #기댓값

# plot the normal distribution PDF
from scipy.stats import norm
xmin, xmax = (blue_x.min(), blue_x.max())
x_values = np.linspace(xmin, xmax, 100)
pdf_values = norm.pdf(x_values, loc=4, scale=np.sqrt(1.3333/20))
plt.plot(x_values, pdf_values, color='red', linewidth = 2)

plt.show()
plt.clf()
------------------------------------------------------------
# 신뢰구간

# X bar ~ N(mu, sigma^2/n)
# X bar ~ N(4, 1.33333/20)

# plot the normal distribution PDF
from scipy.stats import norm
x_values = np.linspace(3, 5, 100)
pdf_values = norm.pdf(x_values, loc=4, scale=np.sqrt(1.3333/20))
plt.plot(x_values, pdf_values, color='red', linewidth = 2)

# 표본평균(파란벽돌) 점 찍기
blue_x = uniform.rvs(loc=2, scale=4, size=20).mean()
a = blue_x + 0.665 # 99% 신뢰구간
b = blue_x - 0.665 # 99% 신뢰구간
# norm.ppf(0.95,loc=0,scale=1) == 1.96
# a = blue_x + 1.96 * np.sqrt(1.3333/20) : 95% 신뢰구간
# b = blue_x - 1.96 * np.sqrt(1.3333/20) : 95% 신뢰구간

plt.scatter(blue_x, 0.002, color='blue', zorder=10, s=5)
plt.axvline(4, color='green', linestyle ='--', linewidth=2)
plt.axvline(a, color='blue', linestyle ='--', linewidth=1)
plt.axvline(b, color='blue', linestyle ='--', linewidth=1)



# 기댓값 표현
plt.axvline(4, color='green', linestyle ='-', linewidth=2)

plt.show()
plt.clf()

# 95% 신뢰구간(a,b)
norm.ppf(0.025, loc=4, scale=np.sqrt(1.3333/20))
norm.ppf(0.975, loc=4, scale=np.sqrt(1.3333/20))

# 99% 신뢰구간(a,b)
norm.ppf(0.005, loc=4, scale=np.sqrt(1.3333/20))
4-norm.ppf(0.995, loc=4, scale=np.sqrt(1.3333/20))



sns.histplot(blue_x, stat="density")
plt.show()
plt.clf()

