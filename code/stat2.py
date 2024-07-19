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


