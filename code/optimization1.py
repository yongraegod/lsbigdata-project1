import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# y = (x-2)^2 + 1 그려보기!
x = np.linspace(-4,8,100)
y = (x - 2)**2 + 1
plt.plot(x, y, color='black')
plt.xlim(-4,8)
plt.ylim(0,15)

# # y = 4x -11 빨간색으로 그려보기!
# x = np.linspace(-4,8,100)
# y = 4*x - 11
# plt.plot(x, y, color='red')

k = -2
# f'(x) = 2x-4
# k=4의 기울기
l_slope = 2*k - 4
f_k = (k-2)**2 + 1
l_intercept = f_k - l_slope * k

# y = slope*x + intercept 그래프
line_y = l_slope*x + l_intercept
plt.plot(x, line_y, color='red')

# y=x^2 경사하강법
# 초기값:10, 델타:0.9
x = 10
lstep = np.arange(100, 0, -1)*0.01
for i in range(100):
    x-=lstep[i]*(2*x)

print(x)

