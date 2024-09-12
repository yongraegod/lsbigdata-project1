import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# x와 y 범위 설정
x = np.linspace(-1.5, 1.5, 400)
y = np.linspace(-1.5, 1.5, 400)
x, y = np.meshgrid(x, y)

# 이차 함수 형태의 z값 생성
z = 4 * x**2 + 4*y**2

# 3D 그래프 그리기
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 표면 그래프를 그립니다.
ax.plot_surface(x, y, z, cmap='viridis', edgecolor='k', alpha=0.7)

# 그래프 레이블 및 타이틀 설정
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(r'$z = 4x^2 + y^2$')

# 축 범위 조정
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(0, 10)

# 그래프 표시
plt.show()
