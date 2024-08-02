# 팀별로 원하는 변수를 사용해서 회귀모델을 만들고, 제출할 것!!
GrLivArea: 지상 생활 면적 (제곱피트)

# 회귀모델을 통한 집값 예측

#필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 필요한 데이터 불러오기
house_train = pd.read_csv("data/house_price/train.csv")
house_test = pd.read_csv("data/house_price/test.csv")
sub_df = pd.read_csv("data/house_price/sample_submission.csv")

# 이상치 탐색
house_train.query("GrLivArea > 4500")

# 이상치 제외하고 house_train에 저장
house_train = house_train.query("GrLivArea <= 4500")

# 회귀분석 적합(fit)하기
x = np.array(house_train['GrLivArea']).reshape(-1,1)
y = house_train['SalePrice'].values

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y) # 자동으로 기울기, 절편 값을 구해준다

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

slope = model.coef_[0]
intercept = model.intercept_
print(f"기울기 (slope a): {slope}")
print(f"절편 (intercept b): {intercept}")

# 예측값 계산
y_pred = model.predict(x)

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='data')
plt.plot(x, y_pred, color='red', label='regression line')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([0,5000])
plt.ylim([0,900000])
plt.legend()
plt.show()
plt.clf()

test_x = np.array(house_test['GrLivArea']).reshape(-1,1)
test_x

pred_y = model.predict(test_x) #test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기
sub_df['SalePrice'] = pred_y
sub_df

#파일 바꿔치기: csv로 내보내기
sub_df.to_csv("data/house_price/sample_submission12.csv", index = False)
