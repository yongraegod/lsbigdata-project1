1+1
2*3

a=1
a

b=2
c=3
d=3.5
a+b
a+b+c
4/b
5*b

var1 = [1,2,3]
var1

var2 = [4,5,6]
var2

var1 + var2

str1 = ['a']
str1

str2 = ['a','c','b']
str2

str1 + str2

x = [2, 5, 7]
sum(x)
max(x)
min(x)
x_sum = sum(x)
x_sum

import seaborn as sns
sns.__version__
df = sns.load_dataset('penguins')
sns.pairplot(df, hue='species')

var = ['a', 'a', 'b', 'c']
var

sns.countplot(x=var)

sns.palplot(sns.color_palette("tab10"))
sns.palplot(sns.color_palette("hls"))

import seaborn as sns
import matplotlib.pyplot as plt

# Titanic 데이터셋 로드
df = sns.load_dataset('titanic')

# countplot 생성
sns.countplot(data = df, x = 'class', hue = 'alive')

# 그래프 표시
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

# Titanic 데이터셋 로드
df = sns.load_dataset('titanic')

# 그래프 크기 설정
plt.figure(figsize=(10, 6))

# countplot 생성
sns.countplot(data = df, y = 'class', hue = 'alive')

# y축 레이블 설정
plt.ylabel('Count')

# 그래프 표시
plt.show()

sklearn

import pandas as pd
df = pd.read_csv('adult.csv')
df.info()

import pydataset
pydataset.data()
df = pydataset.data('mtcars')
df

score = [80,60,70,50,90]
score

sum(score)

test_score = sum(score)
print(test_score)
