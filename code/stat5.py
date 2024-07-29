import pandas as pd
import numpy as np

np.random.seed(20240729)

old_seat = np.arange(1,29)
new_seat = np.random.choice(a,28,False)

result = pd.DataFrame({"old_seat": old_seat,
                       "new_seat": new_seat})

pd.DataFrame.to_csv(result, "result.csv")

========================================================
# y = 2x 그래프 그리기
import matplotlib.pyplot as plt

x = np.linspace(0,8,2)
y = 2 * x
plt.scatter(x,y,s=2) #원래 파이썬은 이렇게 점만 보는데
plt.plot(x,y) #plot을 사용해서 두 점을 이어줌

plt.show()
plt.clf()

# y = x^2
x = np.linspace(-8,8,100)
y = x**2
# plt.scatter(x,y,s=2,color="red")
plt.plot(x,y,color="blue")

# x축, y축 범위 설정
plt.xlim(-10,10)
plt.ylim(0,40)
plt.gca().set_aspect('equal',adjustable='box')

#비율 맞추기
# plt.axis('equal')는 xlim, ylim과 같이 사용 불가

plt.show()
plt.clf()
