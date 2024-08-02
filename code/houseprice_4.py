import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
==========================================================

# 1. 'OverallQual'(전체적인 재료와 마감 품질) & 'OverallCond'(전체적인 상태 등급)가 'SalePrice'에 미치는 영향
house_train = pd.read_csv("data/house_price/train.csv")

# 데이터 준비
tot = house_train.groupby('OverallQual') \
           .agg(mean_price=('SalePrice', 'mean'),
                Cond_mean=('OverallCond', 'mean'))

# 그래프 그리기
fig, ax1 = plt.subplots()

# 첫 번째 y축 (counts)
sns.lineplot(data=tot, x='OverallQual', y='mean_price', color='black', linestyle='-', ax=ax1)
ax1.set_xlabel('OverallQual')
ax1.set_ylabel('Mean_price', color='black')
ax1.tick_params(axis='y', labelcolor='black')

# 두 번째 y축 (mean_area)
ax2 = ax1.twinx()  # 공유 x축을 가지는 두 번째 y축
sns.lineplot(data=tot, x='OverallQual', y='Cond_mean', color='red', linestyle='-', ax=ax2)
ax2.set_ylabel('Cond_mean', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# 그래프 제목 및 레이아웃 조정
plt.title('Mean_price and Mean Area by Total Rooms Above Ground')
fig.tight_layout()

# 그래프 표시
plt.show()
plt.clf()

=======================================================
# 2. 'LotArea'(대지면적) & 'TotRmsAbvGrd'(지상층 방 개수)가 'SalePrice'에 미치는 영향

house_train = pd.read_csv("data/house_price/train.csv")
house_train = house_train[['LotArea','TotRmsAbvGrd','SalePrice']]

# 데이터 준비
tot = house_train.groupby('TotRmsAbvGrd') \
           .agg(mean_price=('SalePrice', 'mean'),
                mean_area=('LotArea', 'mean'))

# 그래프 그리기
fig, ax1 = plt.subplots()

# 첫 번째 y축 (counts)
sns.lineplot(data=tot, x='TotRmsAbvGrd', y='mean_price', color='black', linestyle='-', ax=ax1)
ax1.set_xlabel('TotRmsAbvGrd')
ax1.set_ylabel('Mean_price', color='black')
ax1.tick_params(axis='y', labelcolor='black')

# 두 번째 y축 (mean_area)
ax2 = ax1.twinx()  # 공유 x축을 가지는 두 번째 y축
sns.lineplot(data=tot, x='TotRmsAbvGrd', y='mean_area', color='red', linestyle='-', ax=ax2)
ax2.set_ylabel('Mean Area', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# 그래프 제목 및 레이아웃 조정
plt.title('Mean_price and Mean Area by Total Rooms Above Ground')
fig.tight_layout()

# 그래프 표시
plt.show()
========================================================
수정하기!!
# 'OverallQual'와 'OverallCond'별 'SalePrice' 평균
house_train = pd.read_csv("data/house_price/train.csv")

# OverallQual별 평균 SalePrice 계산
overallqual_mean = house_train.groupby('OverallQual')['SalePrice'].mean()

# OverallCond별 평균 SalePrice 계산
overallcond_mean = house_train.groupby('OverallCond')['SalePrice'].mean()

# 라인 그래프 그리기
fig, ax1 = plt.subplots()

ax1.plot(overallqual_mean.index, overallqual_mean.values, label='OverallQual', marker='o')
ax1.plot(overallcond_mean.index, overallcond_mean.values, label='OverallCond', marker='x')

ax1.set_xlabel('Quality/Condition Rating')
ax1.set_ylabel('Average Sale Price')
ax1.set_title('Average Sale Price by OverallQual and OverallCond')
ax1.legend()

plt.show()


# 그래프 표시
plt.show()
plt.clf()


========================================================
# pairplot
df = house_train[['SalePrice', 'YearBuilt', 'LotFrontage', 'LotArea', 'OverallQual', \
                 'MasVnrArea', 'BsmtQual', 'TotalBsmtSF', 'GrLivArea', 'FullBath', \
                 'TotRmsAbvGrd', 'GarageArea', 'PoolArea']]

plt.figure(figsize=(12, 10))
sns.pairplot(df, plot_kws={'s': 5})
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.show()
plt.clf()
========================================================
train =  pd.read_csv('data/house_price/train.csv')
#train = train[['Id', 'SaleType', 'ExterCond', 'GarageCars', 'LandContour', 'LandSlope', 'Neighborhood','SalePrice']]

# SaleType
SaleType_mean = train.groupby('SaleType', as_index = False) \
                     .agg(S_price_mean = ('SalePrice', 'mean'))
SaleType_mean = SaleType_mean.sort_values('S_price_mean', ascending = False)
sns.barplot(data = SaleType_mean, x = 'SaleType', y = 'S_price_mean', hue = 'SaleType')
plt.show()
plt.clf()

# ExterCond
ExterCond_mean = train.groupby('ExterCond', as_index=False) \
                      .agg(mean_sale=('SalePrice', 'mean'))

ExterCond_mean = ExterCond_mean.sort_values('mean_sale', ascending=False)

sns.barplot(data=ExterCond_mean, y='mean_sale', x='ExterCond', hue='ExterCond')
plt.show()
plt.clf()


# Ex > TA > Good> 각 개수가 몰려 있다.

# GarageCars
GarageCars_mean = train.groupby('GarageCars', as_index = False) \
                             .agg(mean_price = ('SalePrice', 'mean')) \
                             .sort_values('mean_price', ascending = False)

sns.barplot(data = GarageCars_mean, x = 'GarageCars', y = 'mean_price', hue = 'GarageCars')
plt.show()
plt.clf()

# scatter 찍어보기

LandContour_scatter1 = train[["LandContour", "SalePrice"]]
plt.scatter(data = LandContour_scatter1, x="LandContour", y= "SalePrice")
plt.show()
#
plt.clf()
LandContour_scatter2 = train[["LandSlope", "SalePrice"]]
plt.scatter(data = LandContour_scatter2, x="LandSlope", y= "SalePrice")
plt.show()


plt.clf()
LandContour_scatter3 = train[["SaleType", "SalePrice"]]
plt.scatter(data = LandContour_scatter3, x="SaleType", y= "SalePrice")
plt.show()



plt.clf()
LandContour_scatter5 = train[["Condition1", "SalePrice"]]
plt.scatter(data = LandContour_scatter5, x="Condition1", y= "SalePrice")
plt.show()

####
Neighborhood_mean = train.groupby('Neighborhood', as_index = False) \
                         .agg(N_price_mean = ('SalePrice', 'mean'))

plt.clf()
plt.grid()
sns.barplot(data = Neighborhood_mean, y = 'Neighborhood', x = 'N_price_mean', hue = 'Neighborhood')

LandContour_scatter4 = train[["Neighborhood", "SalePrice"]]
plt.scatter(data = LandContour_scatter4, y="Neighborhood", x= "SalePrice", s = 1, color = 'red')

plt.xlabel("price", fontsize=10)
plt.ylabel("n", fontsize=10)
plt.yticks(rotation=45,fontsize=8)
plt.show()


plt.clf()
sns.barplot(data = SaleType_mean, x = 'SaleType', y = 'S_price_mean', hue = 'SaleType')
plt.scatter(data = LandContour_scatter3, x="SaleType", y= "SalePrice", color = 'red')
plt.show()
===============================================


