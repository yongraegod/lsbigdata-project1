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
