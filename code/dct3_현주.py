# 펭귄 데이터 부리길이 예측 모형 만들기
# ElasticNet & DecisionTree 회귀 모델 사용
# 모든 변수 자유롭게 사용!
# 종속변수: bill_length_mm

import numpy as np
import pandas as pd
from palmerpenguins import load_penguins
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

# 1. 펭귄 데이터 로드 및 전처리
penguins = load_penguins()
df = penguins.dropna()  # 결측치 제거
df = df.rename(columns={"bill_length_mm": "y"})  # 종속변수 이름 변경

# 범주형 변수 더미 코딩 (drop_first=True를 통해 다중공선성 방지)
df = pd.get_dummies(df, drop_first=True)

# 독립변수 X와 종속변수 y 분리
X = df.drop(columns='y')
y = df['y']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#--------------------------------------------------------------
# 2. ElasticNet 회귀 모델  GridSearchCV
elastic_model = ElasticNet()
param_grid = {
    'alpha': [0.1, 1.0, 10.0, 100.0],
    'l1_ratio': [0, 0.1, 0.5, 1.0]
}

# GridSearchCV 설정
grid_search = GridSearchCV(
    estimator=elastic_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',  # MSE
    cv=5  # 5-fold 
)

# 모델 학습 및 최적 하이퍼파라미터 탐색
grid_search.fit(X_train, y_train)

# 최적 하이퍼파라미터 및 성능 출력
print("ElasticNet 최적 하이퍼파라미터:", grid_search.best_params_)
print("ElasticNet 최적 성능 (MSE):", grid_search.best_score_)

# 최적 모델을 사용한 예측 및 성능 평가
best_elastic_model = grid_search.best_estimator_
y_pred_elastic = best_elastic_model.predict(X_test)

# 성능 평가
e_mse = mean_squared_error(y_test, y_pred_elastic)
e_r2 = r2_score(y_test, y_pred_elastic)
print(f"ElasticNet MSE: {e_mse}")
print(f"ElasticNet R-squared: {e_r2}")

#--------------------------------------------------------------
# 3. Decision Tree 회귀 모델
tree_model = DecisionTreeRegressor(random_state=42, max_depth=6, min_samples_split=10)
tree_model.fit(X_train, y_train)

# 예측 및 성능 평가
y_pred_tree = tree_model.predict(X_test)
t_mse = mean_squared_error(y_test, y_pred_tree)
t_r2 = r2_score(y_test, y_pred_tree)

# 성능 출력
print(f"Decision Tree MSE: {t_mse}")
print(f"Decision Tree R-squared: {t_r2}")

#-------------------------------------------------------------