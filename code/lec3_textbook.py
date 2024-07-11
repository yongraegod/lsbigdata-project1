#교재 63페이지

#!pip install seaborn

import matplotlib.pyplot as plt
 
import seaborn as sns

var = ['a','a','b','c']
var

sns.countplot(x=var)
plt.show()
plt.clf()


df = sns.load_dataset('titanic')
sns.countplot(data=df,x='sex', hue='sex')
sns.countplot(data=df,x='sex', hue='sex',orient='v')
plt.show()
plt.clf()

sns.countplot(data=df,x='class')
plt.show()
plt.clf()

sns.countplot(data=df,
              x='who',
              linewidth=3,
              edgecolor=sns.color_palette('YlGnBu'),
              orient='v')
plt.show()
plt.clf()

#69 페이지
#!pip install scikit-learn

import sklearn.metrics
sklearn.metrics.accuracy_score()

from sklearn import metrics
metrics.accuracy_score()

