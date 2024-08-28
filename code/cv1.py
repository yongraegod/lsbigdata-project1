import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import uniform
from sklearn.linear_model import LinearRegression

np.random.seed(2024)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

import pandas as pd
df = pd.DataFrame({
    "y" : y,
    "x" : x
})
df

for i in range(2, 21):
    df[f"x{i}"] = df["x"] ** i
df

# 30개를 랜덤하게 섞음
myindex = np.random.choice(30, 30, replace=False)

# 3등분
myindex[0:10]
myindex[10:20]
myindex[20:30]

df.loc[myindex[0:10]]
df.drop(myindex[0:10])

def make_tr_val(fold_num, df, cv_num=3):
    np.random.seed(2024)
    myindex = np.random.choice(30, 30, replace=False)

    # valid index
    val_index=myindex[(10*fold_num):(10*fold_num+10)]
    
    # valid_set, train_set
    valid_set=df.loc[val_index]
    train_set=df.drop(val_index)

    train_X = train_set.iloc[:,1:]
    train_y = train_set.iloc[:,0]

    valid_X = valid_set.iloc[:,1:]
    valid_y = valid_set.iloc[:,0]


    return (train_X, train_y, valid_X, valid_y)

# --------------------------------------------

from sklearn.linear_model import Lasso

# 결과값 받기 위한 매트릭스 생성
val_result_total = np.repeat(0.0, 3000).reshape(3, -1)
tr_result_total = np.repeat(0.0, 3000).reshape(3, -1)

for j in np.arange(0, 3):
    train_X, train_y, valid_X, valid_y = make_tr_val(fold_num=j, df=df)


    # 결과 받기 위한 벡터 만들기
    val_result=np.repeat(0.0, 1000)
    tr_result=np.repeat(0.0, 1000)

    for i in np.arange(0, 1000):
        model= Lasso(alpha=i*0.01)
        model.fit(train_X, train_y)

        # 모델 성능
        y_hat_train = model.predict(train_X)
        y_hat_val = model.predict(valid_X)

        perf_train=sum((train_y - y_hat_train)**2)
        perf_val=sum((valid_y - y_hat_val)**2)
        tr_result[i]=perf_train
        val_result[i]=perf_val

    tr_result_total[j,:] = tr_result
    val_result_total[j,:] = val_result

tr_result_total
val_result_total

# --------------------------------------------

import seaborn as sns

df = pd.DataFrame({
    'lambda': np.arange(0, 10, 0.01),
    'tr': tr_result_total.mean(axis=0),
    'val': val_result_total.mean(axis=0)
})

# seaborn을 사용하여 산점도 그리기
sns.scatterplot(data=df, x='lambda', y='tr')
sns.scatterplot(data=df, x='lambda', y='val', color='red')
plt.xlim(0, 10)

np.min(val_result_total.mean(axis=0))

# alpha를 2.67로 선택!
np.argmin(val_result_total.mean(axis=0))
np.arange(0, 10, 0.01)[np.argmin(val_result_total.mean(axis=0))]