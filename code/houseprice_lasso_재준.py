import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures

import os
os.getcwd()
os.chdir('..')

## 필요한 데이터 불러오기
house_train = pd.read_csv("./data/house_price/train.csv")
house_test = pd.read_csv("./data/house_price/test.csv")
sub_df = pd.read_csv("./data/house_price/sample_submission.csv")

# ID빼고 선택
house_train=house_train.iloc[:,1:]
house_test=house_test.iloc[:,1:]

# train nan값 채우기
# 각 숫자변수는 평균채우기
# 숫자형 채우기
quantitative = house_train.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    house_train[col].fillna(house_train[col].mean(), inplace=True)
house_train[quant_selected].isna().sum()


# # 각 문자변수는 unknow 채우기
 # 범주형 채우기
quantitative = house_train.select_dtypes(include = [object])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
     house_train[col].fillna("unknown", inplace=True)
house_train[quant_selected].isna().sum()

# test nan값 채우기
# 각 숫자변수는 평균채우기
# 숫자형 채우기
quantitative = house_test.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    house_test[col].fillna(house_test[col].mean(), inplace=True)
house_test[quant_selected].isna().sum()


## 각 문자변수는 unknow 채우기
 # 범주형 채우기
quantitative = house_test.select_dtypes(include = [object])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
     house_test[col].fillna("unknown", inplace=True)
house_test[quant_selected].isna().sum()

## 이상치 탐색 (여기 넣으면 안됨!)
# house_train=house_train.query("GrLivArea <= 4500")

house_train.shape
house_test.shape
train_n=len(house_train)

# 통합 df 만들기 + 더미코딩
df = pd.concat([house_train, house_test], ignore_index=True)

#df.info()

df.select_dtypes(include=[object]).columns

df = pd.get_dummies(
    df,
    columns= df.select_dtypes(include=[object]).columns,
    drop_first=True
    )
df

# train / test 데이터셋
train_df=df.iloc[:train_n,]
test_df=df.iloc[train_n:,]

## train
train_x=train_df.drop("SalePrice",axis=1)
train_y=train_df["SalePrice"]

## test
test_x=test_df.drop("SalePrice",axis=1)


# 교차 검증 설정
kf = KFold(n_splits=5, shuffle=True, random_state=2024) #셔플한 값 만들기

# 알파 값 설정 처음에는 값간격을 크게하고 범위를 넓혀서 찾은후
# 세세한 값을 찾기 위해서 값간격을 작게하고 범위를 좁혀서 세세한 값을 찾는다
alpha_values = np.arange(180,200 , 0.1)

# 각 알파 값에 대한 교차 검증 점수 저장
mean_scores = np.zeros(len(alpha_values))


def rmse(model):
    score = np.sqrt(-cross_val_score(model, train_x, train_y, cv = kf, n_jobs=-1,
                                     scoring = "neg_mean_squared_error").mean())
    return(score)



k=0
for alpha in alpha_values:
    lasso = Lasso(alpha=alpha)
    mean_scores[k] = rmse(lasso)
    k += 1

# 결과를 DataFrame으로 저장
df = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})

df

# 결과 시각화
plt.plot(df['lambda'], df['validation_error'], label='Validation Error', color='red')
plt.xlabel('Lambda')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Lasso Regression Train vs Validation Error')
plt.show()

# 최적의 alpha 값 찾기
optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
print("Optimal lambda:", optimal_alpha)


# 선형 회귀 모델 생성
model = Lasso(alpha=190.2)

# 모델 학습
model.fit(train_x, train_y)  # 자동으로 기울기, 절편 값을 구해줌

test_x.columns[test_x.isna().sum()>0]

pred_y=model.predict(test_x) # test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y
sub_df
# csv 파일로 내보내기
sub_df.to_csv("./data/house_price/sample_submission_lasso_재준.csv", index=False)