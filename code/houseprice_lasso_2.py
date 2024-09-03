import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# 데이터 불러오기
house_train = pd.read_csv("./data/house_price/train.csv")
house_test = pd.read_csv("./data/house_price/test.csv")
sub_df = pd.read_csv("./data/house_price/sample_submission.csv")

# NaN 채우기
# 수치형 변수는 평균으로 채우기
quantitative = house_train.select_dtypes(include=[int, float])
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]
for col in quant_selected:
    house_train[col].fillna(house_train[col].mean(), inplace=True)
    house_test[col].fillna(house_test[col].mean(), inplace=True)

# 범주형 변수는 'Unknown'으로 채우기
qualitative = house_train.select_dtypes(include=[object])
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]
for col in qual_selected:
    house_train[col].fillna("Unknown", inplace=True)
    house_test[col].fillna("Unknown", inplace=True)

train_n = len(house_train)

# 통합 df 만들기 + 더미코딩
df = pd.concat([house_train, house_test], ignore_index=True)
df = pd.get_dummies(df, columns=df.select_dtypes(include=[object]).columns, drop_first=True)

# train / test 데이터셋 분리
train_df = df.iloc[:train_n,].copy()
test_df = df.iloc[train_n:,].copy()

# Validation 셋 나누기
np.random.seed(42)
val_index = np.random.choice(np.arange(train_n), size=438, replace=False)

valid_df = train_df.loc[val_index]  # 30%
train_df = train_df.drop(val_index) # 70%

# 이상치 탐색 및 제거
train_df = train_df.query("GrLivArea <= 4500") # 이상치 두 개가 빠짐

# x, y 나누기
train_x = train_df.drop("SalePrice", axis=1)
train_y = train_df["SalePrice"]
valid_x = valid_df.drop("SalePrice", axis=1)
valid_y = valid_df["SalePrice"]
test_x = test_df.drop("SalePrice", axis=1)

# 결측치가 있는지 확인
print("NaN values in train_x:", train_x.isna().sum().sum())
print("NaN values in valid_x:", valid_x.isna().sum().sum())
print("NaN values in test_x:", test_x.isna().sum().sum())

# 결측치 처리 (수치형 변수는 평균으로 채우기)
train_x.fillna(train_x.mean(), inplace=True)
valid_x.fillna(valid_x.mean(), inplace=True)
test_x.fillna(test_x.mean(), inplace=True)

# 스케일링
scaler = StandardScaler()
train_x_scaled = scaler.fit_transform(train_x)
valid_x_scaled = scaler.transform(valid_x)
test_x_scaled = scaler.transform(test_x)

# 결측치가 있는지 다시 확인
print("NaN values in train_x_scaled:", np.isnan(train_x_scaled).sum())
print("NaN values in valid_x_scaled:", np.isnan(valid_x_scaled).sum())
print("NaN values in test_x_scaled:", np.isnan(test_x_scaled).sum())

# 교차 검증 설정
kf = KFold(n_splits=5, shuffle=True, random_state=2024)

# RMSE 계산 함수
def rmse(model, X, y):
    score = np.sqrt(-cross_val_score(model, X, y, cv=kf, n_jobs=-1, scoring="neg_mean_squared_error").mean())
    return score

# 각 알파 값에 대한 교차 검증 점수 저장
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

# 검증 데이터에 대한 성능 측정
y_hat = lasso_best.predict(valid_x_scaled)
rmse_valid = np.sqrt(mean_squared_error(valid_y, y_hat))
print("Validation RMSE:", rmse_valid)

# 테스트 데이터에 대한 예측
pred_y = lasso_best.predict(test_x_scaled)
sub_df["SalePrice"] = pred_y

# 결과 csv 파일로 저장
sub_df.to_csv("./data/house_price/sample_submission_lasso2.csv", index=False)

# 결과 시각화
plt.plot(df_scores['alpha'], df_scores['validation_error'], label='Validation Error', color='red')
plt.xlabel('Alpha')
plt.ylabel('Root Mean Squared Error')
plt.legend()
plt.title('Lasso Regression Validation Error vs Alpha')
plt.show()
