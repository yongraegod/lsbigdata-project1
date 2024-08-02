pip install plotly

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# 데이터 로드
house_train = pd.read_csv("data/house_price/train.csv")
house_test = pd.read_csv("data/house_price/test.csv")
sub_df = pd.read_csv("./data/house_price/sample_submission.csv")

# 이상치 제거
house_train = house_train.query('GrLivArea <= 4500')

# 피처와 타겟 변수 선택
x = house_train[['GrLivArea', 'GarageArea']]
y = house_train['SalePrice']

# 모델 학습
model = LinearRegression()
model.fit(x, y)

# 기울기와 절편 출력
slope_grlivarea = model.coef_[0]
slope_garagearea = model.coef_[1]
intercept = model.intercept_

print(f"GrLivArea의 기울기 (slope): {slope_grlivarea}")
print(f"GarageArea의 기울기 (slope): {slope_garagearea}")
print(f"절편 (intercept): {intercept}")

# 회귀 평면
GrLivArea_vals = np.linspace(x['GrLivArea'].min(), x['GrLivArea'].max(), 100)
GarageArea_vals = np.linspace(x['GarageArea'].min(), x['GarageArea'].max(), 100)
GrLivArea_vals, GarageArea_vals = np.meshgrid(GrLivArea_vals, GarageArea_vals)
SalePrice_vals = intercept + slope_grlivarea * GrLivArea_vals + slope_garagearea * GarageArea_vals

# Plotly 그래프 생성
fig = go.Figure()

# 데이터 포인트
fig.add_trace(go.Scatter3d(
    x=x['GrLivArea'],
    y=x['GarageArea'],
    z=y,
    mode='markers',
    marker=dict(size=5, color='blue'),
    name='Data points'
))

# 회귀 평면
fig.add_trace(go.Surface(
    x=GrLivArea_vals,
    y=GarageArea_vals,
    z=SalePrice_vals,
    colorscale='reds',
    opacity=0.5,
    name='Regression plane'
))

# 축 라벨 설정
fig.update_layout(
    scene=dict(
        xaxis_title='GrLivArea',
        yaxis_title='GarageArea',
        zaxis_title='SalePrice'
    ),
    title='3D Regression Plane with Data Points'
)

# 그래프 표시
fig.show()
