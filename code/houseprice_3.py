import numpy as np
import pandas as pd
house_train = pd.read_csv("data/house_price/train.csv")
house_train = house_train[["Id","OverallQual","SalePrice"]]
house_train.info()

# OverallQual별 평균
house_mean = house_train.groupby("OverallQual", as_index = False) \
                        .agg(mean_OverallQual = ("SalePrice", "mean"))
house_mean

# test 불러오기 
house_test = pd.read_csv("data/house_price/test.csv")
house_test = house_test[["Id","OverallQual"]]
house_test

# test에 연도별 평균 붙이기
house_test = pd.merge(house_test, house_mean,
                      how = "left", on = "OverallQual")
house_test = house_test.rename(columns = {"mean_OverallQual":"SalePrice"})

# 결측값이 얼마나 있는지 확인
house_test["SalePrice"].isna().sum()
# sum(house_test["SalePrice"].isna())

# 비어있는 테스트 세트 집 확인
house_test.loc[house_test["SalePrice"].isna()]

# 비어있는 값을 평균으로 채우기
house_mean = house_train["SalePrice"].mean()
house_test['SalePrice'] = house_test['SalePrice'].fillna(house_mean)

#sub 데이터 불러오기
sub_df = pd.read_csv("data/house_price/sample_submission.csv")

# SalePrice 바꿔치기
sub_df['SalePrice'] = house_test['SalePrice']
sub_df

#파일 바꿔치기: csv로 내보내기
sub_df.to_csv("data/house_price/sample_submission3.csv", index = False)
