# 패키지 설치 및 데이터 로드
import plotly.express as px
import pandas as pd
import numpy as np
from palmerpenguins import load_penguins

penguins = load_penguins()
penguins.head()

fig = px.scatter(
    penguins,
    x="bill_length_mm",
    y="bill_depth_mm",
    color="species",
    size_max=20,  # 점 크기 최대값 설정
    # trendline = 'ols'
)

fig.show()

fig.update_layout(
    title = {'text' : "<span style = 'color:blue;font-weight:bold'> 팔머펭귄 </span>",
             'x' : 0.5}
)
fig

# css
<span>
    <span style = 'font-weight:bold'> ... </span>
    

</span>