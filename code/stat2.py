import numpy as np
import matplotlib.pyplot as plt

# 예제 넘파이 배열 생성
data = np.random.rand(10)

# 히스토그램 그리기
plt.hist(data, bins = 4, alpha = 0.7, color = 'blue')
plt.title('Histogram of Numpy Vector')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

plt.clf()


#연습
#1. 0~1사이 숫자 5개 발생
np.random.rand(5)

#2. 표본 평균 계산하기
np.random.rand(5).mean()

#3. 1,2 단계를 10,000번 반복한 결과를 벡터로 만들기
np.random.rand(50000).mean()

#4. 이를 히스토그램으로 그리기
x = np.random.rand(50000).reshape(-1, 5).mean(axis=1)
#   np.random.rand(10000, 5).mean(axis=1)

plt.hist(x, bins = 30, alpha = 0.7, color = 'blue')
plt.title('Histogram of Numpy Vector')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

------------------------------------------------------------

np.arange(33).sum()/33

x = np.arange(33)
sum(x) / 33
sum((x-16) * 1/33)
(x-16)**2

np.unique((x-16)**2 * (2/33))
sum(np.unique((x-16)**2 * (2/33)))

# E[X^2]
sum(x**2 * (1/33))

# Var(X) = E[X^2] - (E(X))^2
sum(x**2 * (1/33)) - 16**2

(114-81)/36
------------------------------------------------------------------------------
# Example
x = np.arange(4)
x
pro_x = np.array([1/6, 2/6, 2/6, 1/6])

# E[x]
Ex = sum(x * pro_x)

# E[x^2]
Exx = sum(x**2 * pro_x)

# 분산
Exx - Ex**2
sum((x - Ex)**2 * pro_x)
------------------------------------------------------------------------------
# Example 2
x = np.arange(99)
x
np.arange(1,51)
np.arange(49,0,-1)

pro_x = np.concatenate((np.arange(1,51), np.arange(49,0,-1))) / 2500

x * pro_x

# E[x]
Ex = sum(x * pro_x)

# E[x^2]
Exx = sum(x**2 * pro_x)

# 분산
Exx - Ex**2
sum((x - Ex)**2 * pro_x)
------------------------------------------------------------------------------
# Example 3
x = np.arange(4) * 2
x
pro_x = np.array([1/6, 2/6, 2/6, 1/6])

# E[x]
Ex = sum(x * pro_x)

# E[x^2]
Exx = sum(x**2 * pro_x)

# 분산
Exx - Ex**2
sum((x - Ex)**2 * pro_x)
------------------------------------------------------------------------------

np.sqrt(9.52**2 / 25)
np.sqrt(40 / 30)

np.sqrt(81/25)

