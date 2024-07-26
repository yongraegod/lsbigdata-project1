import numpy as np
import pandas as pd
house_df = pd.read_csv("data/house_price/train.csv")
house_df.shape

price_mean = house_df['SalePrice'].mean()

sub_df = pd.read_csv("data/house_price/sample_submission.csv")
sub_df['SalePrice'] = price_mean
sub_df

#파일 바꿔치기: csv로 내보내기
sub_df.to_csv("data/house_price/sample_submission.csv", index = False)
