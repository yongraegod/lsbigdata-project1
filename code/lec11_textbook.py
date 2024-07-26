p.199

import pandas as pd

mpg = pd.read_csv("data/mpg.csv")
mpg.head()
mpg.shape

!pip install seaborn
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(data=mpg,
                x="displ",
                y="hwy",
                hue="drv") \
    .set(xlim = [3,6], ylim = [10,30])

plt.show()
plt.clf()

# 막대그래프
# mpg["drv"].unique() #유니크 값만 보기
df_mpg = mpg.groupby("drv", as_index = False) \
            .agg(mean_hwy = ('hwy','mean'))
            
sns.barplot(data = df_mpg.sort_values("mean_hwy", ascending = False),
            x = 'drv',
            y = 'mean_hwy',
            hue = 'drv')
plt.show()
plt.clf()

#p.208 Do it! 실습
df_mpg = mpg.groupby('drv', as_index = False) \
            .agg(n = ('drv','count'))

sns.barplot(data = df_mpg, x = 'drv', y = 'n')

sns.countplot(data = mpg, x = 'drv', order = ['4','f','r'])

plt.show()
plt.clf()

mpg
