# p. 98 운동선수 예제
import numpy as np

mat_a=np.array([14, 4, 0, 10]).reshape(2,2)
mat_a

# 귀무가설: 두 변수 독립
# 대립가설: 두 변수가 독립 X
from scipy.stats import chi2_contingency

chi2, p, df, expected = chi2_contingency(mat_a)
chi2.round(3) # 검정통계량
p.round(4) # p-value

np.sum((mat_a - expected)**2 / expected )

# 유의수준이 0.05라면, p-value가 0.04로 귀무가설을 기각
# 즉, 두 변수는 독립이 아니다.

# X~chi2(1)일 때, P(X> 15.556) = ?
from scipy.stats import chi2

1-chi2.cdf(15.556, df=1)


# 귀무가설: 두 도시에서의 음료 선호도가 동일하다.
# 대립가설: 두 도시에서의 음료 선호도가 동일하지 않다.
mat_b=np.array([[50,30,20],
               [45, 35, 20]])
chi2, p, df, expected = chi2_contingency(mat_b)
chi2.round(3) # 검정통계량
p.round(4) # p-value
expected

# p.234 휴대전화 사용자들의 정치 성향은 다를까?
# 귀무가설: 휴대전화 사용 여부와 정당 지지 성향은 독립이다.
# 대립가설: 휴대전화 사용 여부와 정당 지지 성향은 독립이 아니다.
mat_c = np.array([[49, 47],
                  [15,27],
                  [32, 30]])
mat_c

chi2, p, df, expected = chi2_contingency(mat_c)
chi2.round(3) # 검정통계량
p.round(4) # p-value
# 유의수준 0.05보다 p값이 크므로, 귀무가설을 기각할 수 없다.
expected

# p.104 요일별 출생아 수 
from scipy.stats import chisquare

observed = np.array([13, 23,24, 20,27, 18, 15])
expected = np.repeat(20, 7)
statistic, p_value = chisquare(observed, f_exp=expected)

print("Test statistic: ", statistic.round(3))
print("p-value: ", p_value.round(3))

round((1-chi2.cdf(7.6, df=6)), 3)

# p.112 지역별 대선 후보의 지지율
mat_d = np.array([[176, 124],
                  [193, 107],
                  [159, 141]])
chi2, p, df, expected = chi2_contingency(mat_d)
chi2.round(3) # 검정통계량
p.round(4) # p-value
expected