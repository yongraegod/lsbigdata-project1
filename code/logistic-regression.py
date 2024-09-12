import pandas as pd

# 워킹 디렉토리 설정
import os
cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
os.chdir(parent_dir)

admission_data = pd.read_csv("./data/admission.csv")
print(admission_data)

# GPA: 학점
# GRE: 대학원 입학시험(영어, 수학)

# 합격을 한 사건: Admit
# Admit의 확률 오즈(odds)는?
# P(Admit) = 합격인원 / 전체학생
p_hat = admission_data['admit'].mean()
p_hat / (1 - p_hat)

# p(A): 0.5보다 큰 경우 -> 오즈비: 무한대에 가까워짐
# p(A): 0.5 -> 오즈비: 1
# p(A): 0.5보다 작은 경우 -> 오즈비: 0에 가까워짐
# 확률의 오즈비가 갖는 값의 범위: 0~무한대

admission_data['rank'].unique()

grouped_data = admission_data \
    .groupby('rank', as_index=False) \
    .agg(p_admit=('admit', 'mean'))
grouped_data['odds'] = grouped_data['p_admit'] / (1 - grouped_data['p_admit'])
print(grouped_data)


# ==========================
# admission 데이터 산점도 그리기
# x: gre, y: admit

import seaborn as sns

sns.stripplot(data=admission_data, x='rank', y='admit', jitter=0.3, alpha=0.3)

sns.scatterplot(data=grouped_data, x='rank', y='p_admit')

sns.regplot(data=grouped_data, x='rank', y='p_admit')

# ==========================
import numpy as np

odds_data = admission_data.groupby('rank').agg(p_admit=('admit', 'mean')).reset_index()
odds_data['odds'] = odds_data['p_admit'] / (1 - odds_data['p_admit'])
odds_data['log_odds'] = np.log(odds_data['odds'])
print(odds_data)

sns.regplot(data=odds_data, x='rank', y='log_odds')

# ==========================

import statsmodels.api as sm
model = sm.formula.ols("log_odds ~ rank", data=odds_data).fit()
print(model.summary())

# ==========================

selected_data = odds_data[['rank', 'p_admit', 'odds']]
selected_data['odds_frac'] = selected_data['odds'] / selected_data['odds'].shift(1, fill_value=selected_data['odds'])


# ==========================
import statsmodels.api as sm

admission_data = pd.read_csv("./data/admission.csv")

# admission_data['rank'] = admission_data['rank'].astype('category')
admission_data['gender'] = admission_data['gender'].astype('category')

model = sm.formula.logit("admit ~ gre + gpa + rank + gender", data=admission_data).fit()

print(model.summary())


# ==========================
# 여학생
# GRE: 450
# GPA: 3.0
# Rank: 2

# 이 학생의 합격 확률은???

# odds = exp(-3.4075 + -0.0576 * x1 +  0.0023 * x2 + 0.7753 * x3 -0.5614 * x4)
my_odds = np.exp(-3.4075 + -0.0576 * 0 +  0.0023 * 450 + 0.7753 * 3.0 -0.5614 * 2)

p_hat = my_odds / (my_odds + 1)

p_hat / (1 - p_hat)

# 이 상태(GAP 3.0)에서 GPA가 1 증가하면 합격 확률이 어떻게 변하는가?
my_odds = np.exp(-3.4075 + -0.0576 * 0 +  0.0023 * 450 + 0.7753 * 4.0 -0.5614 * 2)

p_hat = my_odds / (my_odds + 1)

# ==========================

from scipy.stats import norm

2* (1-norm.cdf(2.123, 0, 1))