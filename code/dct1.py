import pandas as pd
import numpy as np
from palmerpenguins import load_penguins

penguins = load_penguins()
penguins.head()

df=penguins.dropna()
df = df[["bill_length_mm", "bill_depth_mm"]]
df = df.rename(columns={'bill_length_mm': 'y',
                        'bill_depth_mm': 'x'})

df

# 원래 MSE는?
np.mean((df["y"] -df["y"].mean())**2) # 29.81

# -----------------------------------

# x=15 기준으로 나눴을때, 데이터 포인트가 몇개씩 나뉘나요?
n1=df.query("x < 15").shape[0] # 1번 그룹
n2=df.query("x >= 15").shape[0] # 2번 그룹

# 1번 그룹은 얼마로 예측하나요?
y_hat1=df.query("x < 15").mean()[0]

# 2번 그룹은 얼마로 예측하나요?
y_hat2=df.query("x >= 15").mean()[0]

# 각 그룹 MSE는 얼마인가요?
mse1=np.mean((df.query("x < 15")["y"] - y_hat1)**2)
mse2=np.mean((df.query("x >= 15")["y"] - y_hat2)**2)

# x=15 의 MSE 가중평균은?
# (mse1 + mse2)*0.5 가 아닌
(mse1*n1 + mse2*n2)/(n1+n2) # 29.23

# ------------------------------------

# x=20 일때 MSE 가중평균은?
n1=df.query("x < 20").shape[0] # 1번 그룹
n2=df.query("x >= 20").shape[0] # 2번 그룹

y_hat1=df.query("x < 20").mean()[0]
y_hat2=df.query("x >= 20").mean()[0]

mse1=np.mean((df.query("x < 20")["y"] - y_hat1)**2)
mse2=np.mean((df.query("x >= 20")["y"] - y_hat2)**2)

(mse1*n1 + mse2*n2)/(n1+n2) # 29.73

# ------------------------------------

# df=df.query("x >= 16.41")

# 기준값 x를 넣으면 MSE값이 나오는 함수는?
def my_mse(x):
    n1=df.query(f"x < {x}").shape[0] # 1번 그룹
    n2=df.query(f"x >= {x}").shape[0] # 2번 그룹

    y_hat1=df.query(f"x < {x}").mean()[0]
    y_hat2=df.query(f"x >= {x}").mean()[0]

    mse1=np.mean((df.query(f"x < {x}")["y"] - y_hat1)**2)
    mse2=np.mean((df.query(f"x >= {x}")["y"] - y_hat2)**2)

    return(mse1*n1 + mse2*n2)/(n1+n2)

my_mse(16.6)

# ------------------------------------

df["x"].min()
df["x"].max()
# 13.1~21.5 사이값 중 0.01 간격으로 MSE 계산을 해서 가장 작은 MSE가 나오는 x는? (minimize 사용)

# x_values = np.linspace(start=13.2, stop=21.4, num=100)
x_values = np.arange(16.51, 21.5, 0.01)
result = np.repeat(0.0, len(x_values))

for i in range(len(x_values)):
    result[i]=my_mse(x_values[i])

result
x_values[np.argmin(result)] # 16.409

# ------------------------------------

# 14.01 | 16.42 | 19.4

# ------------------------------------

# x, y 산점도를 그리고, 빨간 평행선 4개 그려주세요!
import matplotlib.pyplot as plt

df.plot(kind="scatter", x='x', y='y')

# 간격 구분 선 찍기
plt.axvline(x=14.01, color='black', linestyle='--', linewidth=2)
plt.axvline(x=16.42, color='black', linestyle='--', linewidth=2)
plt.axvline(x=19.4, color='black', linestyle='--', linewidth=2)

thresholds=[14.01, 16.42, 19.4]
df["group"] = np.digitize(df["x"], thresholds)
y_mean=df.groupby("group").mean()["y"]

k1=np.linspace(13, 14.01, 100)
k2=np.linspace(14.01, 16.42, 100)
k3=np.linspace(16.42, 19.4, 100)
k4=np.linspace(19.4, 22, 100)

plt.scatter(k1, np.repeat(y_mean[0],100), color="red", s=2)
plt.scatter(k2, np.repeat(y_mean[1],100), color="red", s=2)
plt.scatter(k3, np.repeat(y_mean[2],100), color="red", s=2)
plt.scatter(k4, np.repeat(y_mean[3],100), color="red", s=2)