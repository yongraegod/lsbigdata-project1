from scipy.stats import binom

# Y ~ B(3, 0.7)
# Y가 갖는 값에 대응하는 확률은?
# P(Y=0) = ?
binom.pmf(0, 3, 0.7)

import numpy as np
binom.pmf(np.array([0,1,2,3]), 3, 0.7)

# Y ~ B(20, 0.45)
# P(6 < Y <= 14) = ?
sum(binom.pmf(np.arange(7, 15), 20, 0.45))

binom.cdf(14, 20, 0.45) - binom.cdf(6, 20, 0.45)

# X ~ N(30, 4^2)
# P(X > 24) = ?
from scipy.stats import norm
1 - norm.cdf(24, loc=30, scale=4)

# 표본은 8개를 뽑아서 표본평균 X_bar
# X ~ N(30, 4^2)
# P(28 < X_bar < 29.7) = ?

# X_bar ~ N(30, 4^2/8)
a = norm.cdf(29.7, loc=30, scale=np.sqrt(4**2/8))
b = norm.cdf(28, loc=30, scale=np.sqrt(4**2/8))
a-b

# 표준화 사용 방법
mean = 30
s_var = 4/np.sqrt(8) # 표준편차
right_x = (29.7-mean) / s_var # 표준화
left_x = (28-mean) / s_var

a = norm.cdf(right_x, 0, 1) # 표준정규분포기준
b = norm.cdf(left_x, 0, 1)
a-b

# 자유도가 7인 카이제곱분포 확률밀도 함수 그리기
from scipy.stats import chi2
import matplotlib.pyplot as plt

k = np.linspace(-2, 40, 500)
y = chi2.pdf(k, 7)
plt.plot(k, y, color="black")