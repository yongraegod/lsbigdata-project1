# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import os
os.getcwd()
os.chdir('..')

## 필요한 데이터 불러오기
house_train=pd.read_csv("./data/house_price/train.csv")
house_test=pd.read_csv("./data/house_price/test.csv")
sub_df=pd.read_csv("./data/house_price/sample_submission.csv")

## NaN 채우기
house_train.isna().sum()
house_test.isna().sum()

## 수치형 채우기
# 각 수치형 변수는 평균으로 채우기
quantitative = house_train.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    house_train[col].fillna(house_train[col].mean(), inplace=True)
house_train[quant_selected].isna().sum()

# 각 범주형 변수는 Unknwon으로 채우기
qualitative = house_train.select_dtypes(include = [object])
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    house_train[col].fillna("unknown", inplace=True)
house_train[qual_selected].isna().sum()

# ====================================================
# test 데이터 채우기
## 수치형 채우기
# 각 수치형 변수는 평균으로 채우기
quantitative = house_test.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    house_test[col].fillna(house_train[col].mean(), inplace=True)
house_test[quant_selected].isna().sum()

# 각 범주형 변수는 Unknwon으로 채우기
qualitative = house_test.select_dtypes(include = [object])
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    house_test[col].fillna("unknown", inplace=True)
house_test[qual_selected].isna().sum()
# ====================================================
house_train.shape
house_test.shape
train_n=len(house_train)
# ====================================================


# 통합 df 만들기 + 더미코딩
df = pd.concat([house_train, house_test], ignore_index=True)
# df.info()
df.select_dtypes(include=[object]).columns
df = pd.get_dummies(
    df,
    columns= df.select_dtypes(include=[object]).columns,
    drop_first=True
    )
df

# train / test 데이터셋
train_df = df.iloc[:train_n,]
test_df = df.iloc[train_n:,]

## 이상치 탐색 및 제거
train_df=train_df.query("GrLivArea <= 4500") # 이상치 두 개가 빠짐

## train
train_x=train_df.drop("SalePrice", axis=1)
train_y=train_df["SalePrice"]

## test
test_x=test_df.drop("SalePrice", axis=1)


# Elastic Net
from sklearn.linear_model import ElasticNet
model = ElasticNet()

param_grid={
    'alpha': [0.1, 1.0, 10.0, 100.0],
    'l1_ratio': [0, 0.1, 0.5, 1.0]
}

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5
)

grid_search.fit(train_x, train_y)

grid_search.best_params_
best_model=grid_search.best_estimator_

pred_y = best_model.predict(test_x) # predict 함수 사용가능
pred_y

# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y
sub_df

# csv 파일로 내보내기
sub_df.to_csv("./data/houseprice/sample_submission20240828.csv", index=False)