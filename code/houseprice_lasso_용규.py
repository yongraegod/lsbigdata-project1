import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

# 데이터 불러오기
house_train = pd.read_csv("../data/house_price/train.csv")
house_test = pd.read_csv("../data/house_price/test.csv")
sub_df = pd.read_csv("../data/house_price/sample_submission.csv")

#train NaN 채우기 (수치형 변수)
quantitative = house_train.select_dtypes(include=[int, float])
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    house_train[col].fillna(house_train[col].mean(), inplace=True)

qualitative = house_train.select_dtypes(include=[object])
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    house_train[col].fillna("unknown", inplace=True)

# 통합 데이터프레임 생성 및 더미 코딩
train_n = len(house_train)
df = pd.concat([house_train, house_test], ignore_index=True)

columns_to_drop = ['Street', 'Alley', 'CentralAir', 'Utilities', 'LandSlope', 'PoolQC', 'PavedDrive']

df = df.drop(columns=columns_to_drop)
unknown_counts = (df == "unknown").sum()
columns_to_drop2 = unknown_counts[unknown_counts > 1000].index.tolist()

df = df.drop(columns=columns_to_drop2)

df = pd.get_dummies(df, columns=df.select_dtypes(include=[object]).columns, drop_first=True)

#df속 test 값 채우기 -> 고민 중
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Train/Test 데이터셋 분리
train_df = df_imputed.iloc[:train_n, :]
test_df = df_imputed.iloc[train_n:, :]

# X와 y 변수 분리 (Train 데이터)
X = train_df.drop(columns=['SalePrice'])
y = train_df['SalePrice']

# 데이터 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 교차 검증 설정
kf = KFold(n_splits=5, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model, X_scaled, y, cv=kf,
                                     n_jobs=-1, scoring="neg_mean_squared_error").mean())
    return(score)

# 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(148, 150, 0.1)
mean_scores = np.zeros(len(alpha_values))

for k, alpha in enumerate(alpha_values):
    lasso = Lasso(alpha=alpha)
    mean_scores[k] = rmse(lasso)

# 결과를 DataFrame으로 저장
df2 = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})

# 최적의 alpha 값 찾기
optimal_alpha = df2['lambda'][np.argmin(df2['validation_error'])]
print("Optimal lambda:", optimal_alpha)

# 최적의 alpha 값으로 Lasso 모델 학습
lasso_optimal = Lasso(alpha=optimal_alpha)
lasso_optimal.fit(X_scaled, y)

# Test 데이터셋 준비
X_test = test_df.drop(columns=['SalePrice'])
X_test_scaled = scaler.transform(X_test)

# 예측 수행
predictions = lasso_optimal.predict(X_test_scaled)

# 결과를 DataFrame으로 저장
output = pd.DataFrame({'Id': house_test['Id'], 'SalePrice': predictions})

# CSV 파일로 저장
output.to_csv('../data/submission.csv', index=False)
print("Predictions saved to submission.csv")