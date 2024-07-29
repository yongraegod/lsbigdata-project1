import numpy as np
import pandas as pd
house_df = pd.read_csv("data/house_price/train.csv")
house_df.shape
house_df.info()

price_mean = house_df['SalePrice'].mean()

sub_df = pd.read_csv("data/house_price/sample_submission.csv")
sub_df['SalePrice'] = price_mean
sub_df

#파일 바꿔치기: csv로 내보내기
sub_df.to_csv("data/house_price/sample_submission.csv", index = False)
-----------------------------------------------------------------------
# train.csv에서 연도별 평균을 구해서, test.csv의 집값을 예측

# 연도 범위, 평균 확인
house_df['YearBuilt'].describe()

# 년도 별 그룹바이, 가격 평균
new = house_df.groupby('YearBuilt',as_index = False) \
              .agg(new_price = ('SalePrice','mean'))
new

# test.csv 불러오기
test = pd.read_csv('data/house_price/test.csv')

# test랑 가격 평균을 new2로 합체
new2 = pd.merge(test, new, how = 'left', on = 'YearBuilt')
new2.to_csv('data/house_price/new2.csv', index = False)
new2

# new price 결측치에 전체 평균값넣기
new2 = new2.fillna(new2['new_price'].mean())
new2

# 제출용데이터 불러오기
sub2 = pd.read_csv('data/house_price/sample_submission.csv')

# 제출용 데이터에 년도별 그룹합치기
sub2['SalePrice'] = new2['new_price']
sub2.to_csv('data/house_price/sub2.csv', index = False)

