import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# 데이터 불러오기
house_train = pd.read_csv("./data/house_price/train.csv")
house_test = pd.read_csv("./data/house_price/test.csv")
sub_df = pd.read_csv("./data/house_price/sample_submission.csv")

# 결측치 처리
quantitative = house_train.select_dtypes(include=[np.number])
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]
for col in quant_selected:
    house_train[col].fillna(house_train[col].mean(), inplace=True)
    house_test[col].fillna(house_test[col].mean(), inplace=True)

qualitative = house_train.select_dtypes(include=[object])
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]
for col in qual_selected:
    house_train[col].fillna("Unknown", inplace=True)
    house_test[col].fillna("Unknown", inplace=True)

# 데이터 합치기
train_n = len(house_train)
df = pd.concat([house_train, house_test], ignore_index=True)

# 더미 코딩
df = pd.get_dummies(df, columns=df.select_dtypes(include=[object]).columns, drop_first=True)

# train/test 분리
train_df = df.iloc[:train_n,].copy()
test_df = df.iloc[train_n:,].copy()

# 학습을 위한 데이터 준비
train_x = train_df.drop("SalePrice", axis=1)
train_y = train_df["SalePrice"]

# 테스트 데이터에서 SalePrice가 없다면 NaN이 포함될 수 있으므로 추가
test_x = test_df.drop("SalePrice", axis=1)
test_x = test_x.fillna(test_x.mean())  # 테스트 데이터에서 결측치 처리

# 스케일링
scaler = StandardScaler()
train_x_scaled = scaler.fit_transform(train_x)
test_x_scaled = scaler.transform(test_x)

# 결측치 확인
print("NaN in train_x_scaled:", np.isnan(train_x_scaled).sum())
print("NaN in test_x_scaled:", np.isnan(test_x_scaled).sum())

# 교차 검증 설정
kf = KFold(n_splits=3, shuffle=True, random_state=2024)

# RMSE 계산 함수
def rmse(model, X, y):
    score = np.sqrt(-cross_val_score(model, X, y, cv=kf, scoring="neg_mean_squared_error").mean())
    return score

# 최적의 alpha 값 찾기
alpha_values = np.arange(0.01, 2, 0.1)
mean_scores = []

for alpha in alpha_values:
    lasso = Lasso(alpha=alpha, max_iter=5000)
    mean_scores.append(rmse(lasso, train_x_scaled, train_y))

# 결과를 DataFrame으로 저장
df_scores = pd.DataFrame({
    'alpha': alpha_values,
    'validation_error': mean_scores
})

# 최적의 alpha 값 찾기
optimal_alpha = df_scores['alpha'][np.argmin(df_scores['validation_error'])]
print("최적의 alpha 값:", optimal_alpha)

# 최적의 라쏘 모델 학습
lasso_best = Lasso(alpha=optimal_alpha, max_iter=5000)
lasso_best.fit(train_x_scaled, train_y)

# 예측 및 결과 저장
pred_y = lasso_best.predict(test_x_scaled)
sub_df["SalePrice"] = pred_y
sub_df.to_csv("./data/house_price/sample_submission_lasso.csv", index=False)
