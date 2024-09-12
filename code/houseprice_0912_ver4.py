import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import os

# 워킹 디렉토리 설정 및 데이터 불러오기
cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
os.chdir(parent_dir)
house_train = pd.read_csv("./data/house_price/train.csv")
house_test = pd.read_csv("./data/house_price/test.csv")
sub_df = pd.read_csv("./data/house_price/sample_submission.csv")

# NaN 채우기 (평균, 'unknown'으로 채우기)
quantitative = house_train.select_dtypes(include=[int, float])
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]
for col in quant_selected:
    house_train[col].fillna(house_train[col].mean(), inplace=True)

qualitative = house_train.select_dtypes(include=[object])
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]
for col in qual_selected:
    house_train[col].fillna("unknown", inplace=True)

# test 데이터도 동일하게 처리
quantitative = house_test.select_dtypes(include=[int, float])
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]
for col in quant_selected:
    house_test[col].fillna(house_train[col].mean(), inplace=True)

qualitative = house_test.select_dtypes(include=[object])
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]
for col in qual_selected:
    house_test[col].fillna("unknown", inplace=True)

# train/test 데이터 나누기 및 스케일링 적용
train_n = len(house_train)
df = pd.concat([house_train, house_test], ignore_index=True)
df = pd.get_dummies(df, columns=df.select_dtypes(include=[object]).columns, drop_first=True)
train_df = df.iloc[:train_n,]
test_df = df.iloc[train_n:,]
train_x = train_df.drop("SalePrice", axis=1)
train_y = np.log(train_df["SalePrice"])  # 로그 변환 적용
test_x = test_df.drop("SalePrice", axis=1)

# 스케일러 적용 (스케일링)
scaler = StandardScaler()
train_x_scaled = scaler.fit_transform(train_x)
test_x_scaled = scaler.transform(test_x)

# 부트스트랩 데이터 생성
def bootstrap_data(X, y, n_samples=1000):
    idx = np.random.choice(np.arange(len(X)), n_samples, replace=True)
    return X[idx, :], y[idx]

bts_train_x1, bts_train_y1 = bootstrap_data(train_x_scaled, train_y)
bts_train_x2, bts_train_y2 = bootstrap_data(train_x_scaled, train_y)
bts_train_x3, bts_train_y3 = bootstrap_data(train_x_scaled, train_y)

# ElasticNet 모델 및 그리드 서치
eln_model = ElasticNet()
param_grid_eln = {'alpha': np.arange(10.0, 100.0, 10), 'l1_ratio': [0.9, 0.95, 1.0]}
grid_search = GridSearchCV(estimator=eln_model, param_grid=param_grid_eln, scoring='neg_mean_squared_error', cv=5)

bts_eln_models = []
for bts_x, bts_y in zip([bts_train_x1, bts_train_x2, bts_train_x3],
                        [bts_train_y1, bts_train_y2, bts_train_y3]):
    grid_search.fit(bts_x, bts_y)
    bts_eln_models.append(grid_search.best_estimator_)

# 부스팅 모델 (GradientBoosting)
gb_model = GradientBoostingRegressor()
param_grid_gb = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}
grid_search_gb = GridSearchCV(estimator=gb_model, param_grid=param_grid_gb, scoring='neg_mean_squared_error', cv=5)

grid_search_gb.fit(train_x_scaled, train_y)
best_gb_model = grid_search_gb.best_estimator_

# 예측 스택킹 (부트스트랩된 모델들의 예측값 모으기)
train_stack_preds = []
for model in bts_eln_models:
    train_stack_preds.append(model.predict(train_x_scaled))

# GradientBoosting 모델 예측값 추가
train_stack_preds.append(best_gb_model.predict(train_x_scaled))

train_stack_preds = np.array(train_stack_preds).T  # 열을 모델별 예측값으로 정렬

# 블렌더 모델 (Ridge 사용)
blender_model = Ridge(alpha=10)
blender_model.fit(train_stack_preds, train_y)

# 테스트 데이터에 대한 예측
test_stack_preds = []
for model in bts_eln_models:
    test_stack_preds.append(model.predict(test_x_scaled))

test_stack_preds.append(best_gb_model.predict(test_x_scaled))
test_stack_preds = np.array(test_stack_preds).T

# 최종 예측
final_pred_y_log = blender_model.predict(test_stack_preds)
final_pred_y = np.exp(final_pred_y_log)  # 로그 취소 (exp로 원래 값 복원)

# 결과 저장
sub_df["SalePrice"] = final_pred_y
output_dir = "./data/house_price/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
sub_df.to_csv(os.path.join(output_dir, "sample_submission_ver4.csv"), index=False)
