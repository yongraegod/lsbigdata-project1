import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# 데이터 불러오기
train_df=pd.read_csv("../data/blueberry/train.csv")
test_df=pd.read_csv("../data/blueberry/test.csv")
sub_df=pd.read_csv("../data/blueberry/sample_submission.csv")

train_df.columns

# 데이터 전처리: 결측치 확인 및 처리
train_df.fillna(train_df.mean(), inplace=True)
test_df.fillna(test_df.mean(), inplace=True)

# 훈련 및 테스트 데이터 준비
train_x = train_df.drop(["id", "yield"], axis=1)
train_y = train_df["yield"]
test_x = test_df.drop("id", axis=1)

# 데이터 분할: 훈련 세트와 검증 세트
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

# 파이프라인 설정: 스케일링 -> 다항식 특성 생성 -> Lasso
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # 특성 스케일링
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),  # 다항식 특성 생성
    ('lasso', Lasso(max_iter=10000, random_state=42))  # Lasso 회귀 모델
])

# 하이퍼파라미터 그리드 설정
param_grid = {
    'lasso__alpha': np.logspace(-4, 1, 50)  # 0.0001 ~ 10 사이의 50개의 로그 스케일 알파 값
}

# GridSearchCV를 사용한 최적화
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(train_x, train_y)

# 최적의 모델 및 하이퍼파라미터
best_model = grid_search.best_estimator_
best_alpha = grid_search.best_params_['lasso__alpha']
print("최적의 alpha 값:", best_alpha)

# 검증 데이터에 대한 성능 평가
val_y_pred = best_model.predict(val_x)
val_rmse = np.sqrt(mean_squared_error(val_y, val_y_pred))
print("검증 데이터 RMSE:", val_rmse)

# 테스트 데이터에 대한 예측
test_y_pred = best_model.predict(test_x)

# 예측 결과를 sample_submission.csv에 저장
sub_df["yield"] = test_y_pred
sub_df.to_csv("../data/blueberry/sample_submission_lasso_optimized.csv", index=False)
print("최적화된 예측 결과가 sample_submission_lasso_optimized.csv에 저장되었습니다.")
