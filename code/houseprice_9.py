# 회귀모델을 통한 집값 예측

# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

## 필요한 데이터 불러오기
house_train=pd.read_csv("./data/house_price/train.csv")
house_test=pd.read_csv("./data/house_price/test.csv")
sub_df=pd.read_csv("./data/house_price/sample_submission.csv")

## 이상치 탐색
# house_train=house_train.query("GrLivArea <= 4500")

# dummies 작업(입력값이 범주형인 neighborhood를 추가해보자)
house_train['Neighborhood']
neighborhood_dummies = pd.get_dummies(
    house_train["Neighborhood"],
    drop_first=True)

x = pd.concat([house_train[["GrLivArea", "GarageArea"]],
              neighborhood_dummies], axis = 1)
y = house_train["SalePrice"]

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y)  # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b
----------------------------------------------------------
# test에도 dummies 작업
neighborhood_dummies_test = pd.get_dummies(
    house_test["Neighborhood"],
    drop_first=True)

test_x = pd.concat([house_test[["GrLivArea", "GarageArea"]],
              neighborhood_dummies_test], axis = 1)
test_x

# 결측치 확인
test_x["GrLivArea"].isna().sum()
test_x["GarageArea"].isna().sum()
test_x=test_x.fillna(house_test["GarageArea"].mean())

pred_y=model.predict(test_x) # test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y
sub_df

# csv 파일로 내보내기
sub_df.to_csv("./data/house_price/sample_submission15.csv", index=False)


