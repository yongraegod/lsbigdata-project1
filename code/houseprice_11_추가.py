# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# 워킹 디렉토리 설정
import os
cwd=os.getcwd()
parent_dir = os.path.dirname(cwd)
os.chdir(parent_dir)

## 필요한 데이터 불러오기
house_train=pd.read_csv("./data/house_price/train.csv")
house_test=pd.read_csv("./data/house_price/test.csv")
sub_df=pd.read_csv("./data/house_price/sample_submission.csv")

## NaN 채우기
# 각 숫치형 변수는 평균 채우기
# 각 범주형 변수는 Unknown 채우기
house_train.isna().sum()
house_test.isna().sum()

## 숫자형 채우기
quantitative = house_train.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    house_train[col].fillna(house_train[col].mean(), inplace=True)
house_train[quant_selected].isna().sum()

## 범주형 채우기
qualitative = house_train.select_dtypes(include = [object])
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    house_train[col].fillna("unknown", inplace=True)
house_train[qual_selected].isna().sum()


# test 데이터 채우기
## 숫자형 채우기
quantitative = house_test.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    house_test[col].fillna(house_train[col].mean(), inplace=True)
house_test[quant_selected].isna().sum()

## 범주형 채우기
qualitative = house_test.select_dtypes(include = [object])
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    house_test[col].fillna("unknown", inplace=True)
house_test[qual_selected].isna().sum()


house_train.shape
house_test.shape
train_n=len(house_train)

# 통합 df 만들기 + 더미코딩
# house_test.select_dtypes(include=[int, float])

df = pd.concat([house_train, house_test], ignore_index=True)
# df.info()
df = pd.get_dummies(
    df,
    columns= df.select_dtypes(include=[object]).columns,
    drop_first=True
    )
df

# train / test 데이터셋
train_df=df.iloc[:train_n,]
test_df=df.iloc[train_n:,]

## 이상치 탐색
train_df=train_df.query("GrLivArea <= 4500")

## train
train_x=train_df.drop("SalePrice", axis=1)
train_y=train_df["SalePrice"]

## test
test_x=test_df.drop("SalePrice", axis=1)

# -------------------------------------------------------

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
# !pip install xgboost
import xgboost as xgb

# 모델 정의
models = {
    'ElasticNet': ElasticNet(),
    'RandomForest': RandomForestRegressor(n_estimators=100),
    'KNeighbors': KNeighborsRegressor(),
    'XGBoost': xgb.XGBRegressor()  # XGBoost 추가
}

# ElasticNet 하이퍼파라미터
param_grid_eln = {
    'alpha': [0.1, 1.0, 10.0, 100.0],
    'l1_ratio': [0, 0.1, 0.5, 1.0]
}

# RandomForest 하이퍼파라미터
param_grid_rf = {
    'max_depth': [3, 5, 7],
    'min_samples_split': [20, 10, 5],
    'min_samples_leaf': [5, 10, 20, 30],
    'max_features': ['sqrt', 'log2', None]
}

# KNeighbors 하이퍼파라미터
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9]
}

# XGBoost 하이퍼파라미터
param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

# 모델 그리드 서치 / 랜덤 서치
best_models = {}
param_grids = {
    'ElasticNet': param_grid_eln,
    'RandomForest': param_grid_rf,
    'KNeighbors': param_grid_knn,
    'XGBoost': param_grid_xgb
}

for name, model in models.items():
    param_grid = param_grids[name]
    
    if name == 'XGBoost':  # XGBoost는 RandomizedSearchCV 사용
        search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, 
                                    n_iter=20, scoring='neg_mean_squared_error', cv=5, random_state=42)
    else:  # 그 외 모델은 GridSearchCV 사용
        search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)

    search.fit(train_x, train_y)
    best_models[name] = search.best_estimator_
    print(f"{name} - Best Params: {search.best_params_}")

# 스택킹
preds = []
for name, model in best_models.items():
    preds.append(model.predict(train_x))

train_x_stack = pd.DataFrame(preds).T

# Ridge 모델로 스태킹
from sklearn.linear_model import Ridge
rg_model = Ridge()
param_grid_rg = {'alpha': np.arange(0, 10, 0.01)}
grid_search = GridSearchCV(estimator=rg_model, param_grid=param_grid_rg, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(train_x_stack, train_y)
blander_model = grid_search.best_estimator_

# 테스트 데이터 예측
test_preds = []
for name, model in best_models.items():
    test_preds.append(model.predict(test_x))

test_x_stack = pd.DataFrame(test_preds).T

# 최종 예측
final_pred_y = blander_model.predict(test_x_stack)

# 결과 저장
sub_df["SalePrice"] = final_pred_y
sub_df.to_csv("./data/house_price/sample_submission12.csv", index=False)

# 모델 성능 평가 (훈련 데이터에 대해)
print("\nModel Performance on Training Data:")
for name, model in best_models.items():
    train_pred = model.predict(train_x)
    mse = mean_squared_error(train_y, train_pred)
    r2 = r2_score(train_y, train_pred)
    print(f"{name} - MSE: {mse:.4f}, R2: {r2:.4f}")

# 최종 스태킹 모델 성능 평가
train_stack_pred = blander_model.predict(train_x_stack)
stack_mse = mean_squared_error(train_y, train_stack_pred)
stack_r2 = r2_score(train_y, train_stack_pred)
print(f"\nStacking Model - MSE: {stack_mse:.4f}, R2: {stack_r2:.4f}")
