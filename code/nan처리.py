import pandas as pd
import numpy as np

# NaN 포함
data = {
    '나이': np.random.randint(10, 60, size=10),
    '성적': np.random.choice(['상', '중', '하', np.nan], size=10)  # '상', '중', '하', NaN 중에서 랜덤 선택
}

df1 = pd.DataFrame(data)

df1 = pd.get_dummies(
    df1,
    columns= df1.select_dtypes(include=[object]).columns,
    drop_first=True
    )

# ------------------------------------------------------
# NaN 없음
data = {
    '나이': np.random.randint(10, 60, size=10),
    '성적': np.random.choice(['상', '중', '하'], size=10)  # '상', '중', '하' 중에서 랜덤 선택
}

df2 = pd.DataFrame(data)

df2 = pd.get_dummies(
    df2,
    columns= df2.select_dtypes(include=[object]).columns,
    drop_first=True
    )