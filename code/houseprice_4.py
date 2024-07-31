import numpy as np
import pandas as pd
house_train = pd.read_csv("data/house_price/train.csv")
# house_train = house_train[["Id","OverallQual","SalePrice"]]
# house_train.info()



MSZoning = house_train[['MSZoning']]
type(MSZoning)

MSZoning.query('MSZoning = "RL"')



Neighborhood = house_train['Neighborhood']

sns.countplot(data = house_train, x = MSZoning)
sns.countplot(data = house_train, x = Neighborhood)

sns.barplot(data = house_train, x = 'MSZoning', y = 'SalePrice')
sns.barplot(data = house_train, y = 'Neighborhood', x = 'SalePrice')

plt.show()
plt.clf()

df = house_train.dropna(subset=['MSZoning', 'Neighborhood', 'SalePrice']) \
                .groupby('Neighborhood', as_index = False) \
                .agg(mean_SP = ('SalePrice','mean')) \
                .sort_values('mean_SP', ascending = False) \
                .head(10)
df
