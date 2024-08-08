# plotly 라이브러리 모듈 로딩
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# plotly.express 라이브러리 로딩
import plotly.express as px

# 데이터 패키지 설치
# !pip install palmerpenguins

import pandas as pd
import numpy as np
from palmerpenguins import load_penguins

penguins = load_penguins()
penguins.head()
penguins['species'].unique()

fig = px.scatter(
    penguins,
    x="bill_length_mm",
    y="bill_depth_mm",
    color="species",
    size_max=20,  # 점 크기 최대값 설정
    trendline = 'ols'
)

# 점의 투명도 설정
fig.update_traces(marker=dict(opacity=0.6))

# 레이아웃 업데이트
fig.update_layout(
    title=dict(text="팔머펭귄 종별 부리 길이 vs. 깊이", font=dict(color="white", size=20)),
    paper_bgcolor="black",
    plot_bgcolor="black",
    font=dict(color="white"),
    xaxis=dict(
        title=dict(text="부리 길이 (mm)", font=dict(color="white", size=16)), 
        tickfont=dict(color="white"),
        gridcolor='rgba(255, 255, 255, 0.2)'  # 그리드 색깔 조정
    ),
    yaxis=dict(
        title=dict(text="부리 깊이 (mm)", font=dict(color="white", size=16)), 
        tickfont=dict(color="white"),
        gridcolor='rgba(255, 255, 255, 0.2)'  # 그리드 색깔 조정
    ),
    legend=dict(title=dict(text="펭귄 종류", font=dict(color="white"))),  # 범례 제목 한글로 변경
)

fig.show()
=================================================
# 선형회귀 모델
from sklearn.linear_model import LinearRegression

model = LinearRegression()

penguins = penguins.dropna() # Nan 제거

x=penguins[["bill_length_mm"]]
y=penguins["bill_depth_mm"]

model.fit(x,y)
model.coef_      # model 기울기
model.intercept_ # model 절편
linear_fit = model.predict(x)

fig.add_trace(
    go.Scatter(
        mode="lines",
        x=penguins["bill_length_mm"], y=linear_fit,
        name="선형회귀직선",
        line=dict(dash='dot', color='white')
    )
)

fig.show()
===============================================
# 범주형 변수로 회귀분석 진행하기
# 범주형 변수인 'species'를 더미 변수로 변환
penguins_dummies = pd.get_dummies(penguins, 
                                  columns=['species'],
                                  drop_first=False)
penguins_dummies.columns
penguins_dummies.iloc[:,-3:]

# x와 y 설정
x = penguins_dummies[["bill_length_mm", "species_Chinstrap", "species_Gentoo"]]
y = penguins_dummies["bill_depth_mm"]

# 모델 학습
model = LinearRegression()
model.fit(x, y)

model.coef_
model.intercept_

# 선형회귀 직선의 방정식
# y = 0.2 * bill_lenght -1.93 * species_Chinstrap -5.1 * species_Gentoo + 10.56

# species     island     bill_length_mm  ...  body_mass_g     sex  year
# Adelie      Torgersen            39.5  ...       3800.0  female  2007
# Chinstrap   Torgersen            40.5  ...       3800.0  female  2007
# Gentoo      Torgersen            40.5  ...       3800.0  female  2007

# x1,  x2, x3
# 39.5, 0, 0
# 40.5, 1, 0

regline_y = model.predict(x)

import matplotlib.pyplot as plt
import seaborn as sns

sns.scatterplot(x=penguins["bill_length_mm"], y=regline_y, color = 'black')
sns.scatterplot(x=penguins["bill_length_mm"], y=y, hue = penguins['species'], palette='deep')
plt.show()
plt.clf()



