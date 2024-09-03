import numpy as np

# 함수 정의
def f(beta0, beta1):
    return (1 - (beta0 + beta1))**2 + (4 - (beta0 + 2*beta1))**2 + \
           (1.5 - (beta0 + 3*beta1))**2 + (5 - (beta0 + 4*beta1))**2

# f의 편미분 계산
def gradient(beta0, beta1):
    df_dbeta0 = -2*(1 - (beta0 + beta1)) - 2*(4 - (beta0 + 2*beta1)) - \
                2*(1.5 - (beta0 + 3*beta1)) - 2*(5 - (beta0 + 4*beta1))
    df_dbeta1 = -2*(1 - (beta0 + beta1))*1 - 2*(4 - (beta0 + 2*beta1))*2 - \
                2*(1.5 - (beta0 + 3*beta1))*3 - 2*(5 - (beta0 + 4*beta1))*4
    return np.array([df_dbeta0, df_dbeta1])

# 경사 하강법 설정
beta = np.array([10.0, 10.0])  # 초기값
learning_rate = 0.1
iterations = 100  # 반복 횟수

# 경사 하강법 수행
for i in range(iterations):
    grad = gradient(beta[0], beta[1])
    beta = beta - learning_rate * grad
    if i % 10 == 0:  # 10회마다 출력
        print(f"Iteration {i}: beta0 = {beta[0]:.4f}, beta1 = {beta[1]:.4f}, f(beta0, beta1) = {f(beta[0], beta[1]):.4f}")

# 결과 출력
print(f"최종 결과: beta0 = {beta[0]:.4f}, beta1 = {beta[1]:.4f}, f(beta0, beta1) = {f(beta[0], beta[1]):.4f}")
