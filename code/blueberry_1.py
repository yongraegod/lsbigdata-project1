import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split

train_df=pd.read_csv("../data/blueberry/train.csv")
test_df=pd.read_csv("../data/blueberry/test.csv")
sub_df=pd.read_csv("../data/blueberry/sample_submission.csv")

train_df.isna().sum()
test_df.isna().sum()

train_x = train_df.drop(["id", "yield"], axis = 1)
train_y = train_df["yield"]

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.3, random_state=42)

test_x = test_df.drop("id", axis = 1)

# Ridge
alphas = np.arange(0, 10, 0.01)
model = RidgeCV(alphas = alphas, cv = 4) 
model.fit(train_x, train_y)

val_y_pred = model.predict(val_x)

r2 = r2_score(val_y, val_y_pred)

print('alpha: ', model.alpha_)
print(f'R^2: {r2}')

# 0, 50, 1 => lambda = 1, R^2 = 0.80808
# 0, 5, 0.1 => lambda = 1, R^2 = 0.80808
# 0, 10, 0.01 => lambda = 1.03 R^2 = 0.8080~~

test_y = model.predict(test_x)
sub_df["yield"] = test_y
sub_df.to_csv("../data/blueberry/sample_submission1.csv", index=False)