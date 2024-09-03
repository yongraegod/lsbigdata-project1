import numpy as np

mat_a=np.array([14, 4, 0, 10]).reshape(2,2)
mat_a

# 귀무가설: 두 변수 독립
# 대립가설: 두 변수가 독립 X
from scipy.stats import chi2_contingency

chi2, p, df, expected = chi2_contingency(mat_a)
chi2.round(3) # 검정통계량
p.round(4) # p-value

# 유의수준이 0.05라면, p-value가 0.04로 귀무가설을 기각
# 즉, 두 변수는 독립이 아니다.

# X~chi2(1)일 때, P(X>12.6) = ?
from scipy.stats import chi2

1-chi2.cdf(12.6, df=1)