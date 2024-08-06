import numpy as np
import pandas as pd

tab3=pd.read_csv("./data/tab3.csv")
tab3

tab1 = pd.DataFrame({"id"    : np.arange(1,13),
                     "score" : tab3["score"]})

tab2 = tab1.assign(gender=["female"]*7 + ["male"]*5)
tab2
-----------------------------------------------------------
## 1표본 t 검정 (그룹 1개)
# 귀무가설 vs 대립가설
# H0: 𝜇 = 10 vs Ha: 𝜇 =/= 10
# 유의수준 5%로 설정

from scipy.stats import ttest_1samp

result = ttest_1samp(tab1['score'], popmean=10, alternative='two-sided')
result
t_value = result[0] # t 검정통계량
p_value = result[1] # 유의확률 (p-value)
tab1['score'].mean() # 표본평균

result.pvalue
result.statistic
result.df
# 귀무가설이 참(𝜇=10)일 때, 표본평균(11.53)이 관찰될 확률이 6.48%(유의확률)이므로,
# 이것은 우리가 생각하는 보기 힘들다고 판단하는 기준인
# 0.05 = 5%(유의수준)보다 크므로, 귀무가설을 거짓이라 판단하기 힘들다.
# 유의확률 0.0648이 유의수준 0.05보다 크므로 귀무가설을 기각하지 못한다.

# 95% 신뢰구간 구하기
ci = result.confidence_interval(0.95)
ci[0]
ci[1]
-----------------------------------------------------------
## 2표본 t 검정 (그룹 2개) - 분산 같고, 다를때
# 분산이 같은 경우: 독립 2표본 t 검정
# 분산이 다른 경우: 웰치스 t 검정
# 귀무가설 vs 대립가설
# H0: 𝜇_m = 𝜇_f vs Ha: 𝜇_m > 𝜇_f
# 유의수준 1%로 설정, 두 그룹의 분산은 같다고 가정한다

from scipy.stats import ttest_ind

male = tab2[tab2['gender'] == 'male']
female = tab2[tab2['gender'] == 'female']

# alternative='less'의 의미는 대립가설이 
# 첫번째 입력그룹의 평균이 두번째 입력 그룹 평균보다 작다고 설정된 경우를 나타냄.
# ttest_ind(male['score'], female['score'], alternative="greater", equal_var=True)
result = ttest_ind(female['score'], male['score'], alternative="less", equal_var=True)
result.pvalue
result.statistic

# 95% 신뢰구간 구하기
ci = result.confidence_interval(0.95)
ci[0]
ci[1]
-----------------------------------------------------------
## 대응표본 t 검정 (짝지을 수 있는 표본)
# 귀무가설 vs 대립가설
# H0: 𝜇_before = 𝜇_after vs Ha: 𝜇_after > 𝜇_before
# H0: 𝜇_d = 0 vs Ha: 𝜇_d > 0
# 𝜇_d = 𝜇_after - 𝜇_before
# 유의수준 1%로 설정

# 𝜇_d에 대응하는 표본으로 변환
tab3 = pd.read_csv('./data/tab3.csv')
tab3_data = tab3.pivot_table(index='id',columns='group',values='score').reset_index()

tab3_data['score_diff'] = tab3_data['after'] - tab3_data['before']
test3_data = tab3_data[['score_diff']]
test3_data

from scipy.stats import ttest_1samp

result = ttest_1samp(test3_data['score_diff'], popmean=0, alternative='greater')
result
t_value = result[0] # t 검정통계량
p_value = result[1] # 유의확률 (p-value)
----------------------------------------------------------
# 연습 pivot&melt: long to wide, wide to long

df = pd.DataFrame({"id" : [1,2,3],
                   "A" : [10,20,30],
                   "B" : [40,50,60]})

df_long = df.melt(id_vars = "id",
                  value_vars = ['A','B'],
                  var_name = 'group',
                  value_name = 'score')

df_long.pivot_table(columns = 'group',
                    values = 'score')

df_long.pivot_table(columns = 'group',
                    values = 'score',
                    aggfunc = "sum")

df_wide = df_long.pivot_table(
                    index = 'id',
                    columns = 'group',
                    values = 'score',
                    ).reset_index()

# 연습 2
# !pip install seaborn
import seaborn as sns
tips = sns.load_dataset('tips')

# 요일별로 펼치고 싶은 경우
tips.reset_index(drop=False) \
    .pivot_table(index = 'index',
                 columns = 'day',
                 values = 'tip').reset_index()
                 
tips.pivot_table(columns = 'day',
                 values = 'tip').reset_index()

df2 = tips.pivot_table(index = tips.index, columns = 'day', values = 'tip', aggfunc = 'sum').reset_index()
                 
df1 = tips.drop(columns = 'day').reset_index()
pd.merge(df2, df1, how = 'left', on = 'index')
