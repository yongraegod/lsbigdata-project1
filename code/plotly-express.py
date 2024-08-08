# plotly.express 라이브러리 로딩
import plotly.express as px

# 데이터 패키지 설치
!pip install palmerpenguins

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
    size_max=20  # 점 크기 최대값 설정
)

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
    legend=dict(title=dict(text="펭귄 종류", font=dict(color="white")), font=dict(color="white")),  # 범례 제목 한글로 변경
)

fig.show()

