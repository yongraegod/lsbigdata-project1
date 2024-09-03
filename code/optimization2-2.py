# 등고선 그래프

import numpy as np
import matplotlib.pyplot as plt

# x, y의 값을 정의합니다 (-10에서 10까지)
x = np.linspace(-10, 10, 400)
y = np.linspace(-10, 10, 400)
x, y = np.meshgrid(x, y)

# 함수 f(x, y)를 계산합니다.
z = (1 - (x+y))**2 + (4-(x+2*y))**2 +(1.5-(x+3*y))**2 +(5-(x+4*y))**2 

# 등고선 그래프를 그립니다.
plt.figure()
cp = plt.contour(x, y, z, levels=20)  # levels는 등고선의 개수를 조절합니다.
plt.colorbar(cp)  # 등고선 레벨 값에 대한 컬러바를 추가합니다.

# 특정 점 (10, 10)에 빨간색 점을 표시
plt.scatter(10, 10, color="red", s=50)

# 다음 스텝을 찍어보자
x=10; y=10
lstep=0.1
x, y = np.array([x, y]) - lstep * np.array([8*x + 20*y -23, 60*y + 20*x -47])
x
y
plt.scatter(x, y, color="red", s=50)

# 100번의 스텝 찍기
x=10; y=10
lstep=0.1
for i in range(100):
    x, y = np.array([x, y]) - lstep * np.array([8*x + 20*y -23, 60*y + 20*x -47])
    plt.scatter(x, y, color="red", s=50)
print(x,y)


# 축 레이블 및 타이틀 설정
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(-10, 10)
plt.ylim(-10, 10)

# 그래프 표시
plt.show()
# ------------------------------
!pip install sympy
from sympy import Derivative, symbols

x = symbols("x")
fx = (1 - (x + y))**2 + (4 - (x + 2*y))**2 + (1.5 - (x + 3*y))**2 + (5 - (x + 4 * y))**2
func_x = Derivative(fx, x).doit()
y = symbols("y")
fy = (1 - (x + y))**2 + (4 - (x + 2*y))**2 + (1.5 - (x + 3*y))**2 + (5 - (x + 4 * y))**2
func_y = Derivative(fy, y).doit()