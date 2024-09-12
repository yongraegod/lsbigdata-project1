import numpy as np
import pandas as pd
pip install xgboost
pip install lightgbm
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
import os

# 데이터 불러오기
# cwd = os.getcwd()
# parent_dir = os.path.dirname(cwd)
# os.chdir(parent_dir)
house_train = pd.read_csv("./data/house_price/train.csv")
house_test = pd.read_csv("./data/house_price/test.csv")
sub_df = pd.read_csv("./data/house_price/sample_submission.csv")


# NaN 처리 (학습 데이터에서)
quantitative_train = house_train.select_dtypes(include=[int, float])
for col in quantitative_train.columns:
    if col != 'SalePrice':  # 'SalePrice'는 타겟 변수이므로 제외
        house_train[col].fillna(house_train[col].mean(), inplace=True)

# NaN 처리 (테스트 데이터에서)
quantitative_test = house_test.select_dtypes(include=[int, float])
for col in quantitative_test.columns:
    house_test[col].fillna(house_test[col].mean(), inplace=True)

qualitative = house_train.select_dtypes(include=[object])
for col in qualitative.columns:
    house_train[col].fillna("unknown", inplace=True)
    house_test[col].fillna("unknown", inplace=True)

# 데이터 통합 후 더미 처리
df = pd.concat([house_train, house_test], ignore_index=True)
df = pd.get_dummies(df, drop_first=True)

# Train/Test 나누기
train_n = len(house_train)
train_df = df.iloc[:train_n, :]
test_df = df.iloc[train_n:, :]

# 타겟 로그 변환
train_x = train_df.drop("SalePrice", axis=1)
train_y = np.log1p(train_df["SalePrice"])  # 로그 변환
test_x = test_df.drop("SalePrice", axis=1)

# 특성 스케일링
scaler = StandardScaler()
train_x_scaled = scaler.fit_transform(train_x)
test_x_scaled = scaler.transform(test_x)

# 특성 선택
selector = SelectKBest(f_regression, k=150)  # 상위 150개의 특성 선택
train_x_selected = selector.fit_transform(train_x_scaled, train_y)
test_x_selected = selector.transform(test_x_scaled)

# K-Fold Cross Validation 설정
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# XGBoost 모델
xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror", n_estimators=500, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8
)

# LightGBM 모델
lgb_model = lgb.LGBMRegressor(
    objective="regression", n_estimators=500, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8
)

# Ridge 모델
ridge_model = Ridge(alpha=10)

# 스태킹 모델 정의
stacking_model = StackingRegressor(
    estimators=[
        ("xgb", xgb_model),
        ("lgb", lgb_model),
    ],
    final_estimator=ridge_model,
    cv=kf
)

# 모델 학습
stacking_model.fit(train_x_selected, train_y)

# 테스트 데이터 예측
final_pred_log = stacking_model.predict(test_x_selected)
final_pred = np.expm1(final_pred_log)  # 로그 변환을 원래 값으로 복구

# 결과 저장
sub_df["SalePrice"] = final_pred
output_dir = "./data/house_price/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
sub_df.to_csv(os.path.join(output_dir, "sample_submission_ver5.csv"), index=False)
