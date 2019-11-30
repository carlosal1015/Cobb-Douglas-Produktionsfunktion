import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.datasets import load_boston

dataset = load_boston()
print(dataset.data)
print(dataset.feature_names)
print(dataset.DESCR)
print(dataset.target)
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df.head()
df['MEDV'] = dataset.target
df.head()
df.info()
print(df.isnull().sum())
corr = df.corr()
print(corr)

print(df.corr().abs().nlargest(3, 'MEDV').index)
print(df.corr().abs().nlargest(3, 'MEDV').values[:,13])

plt.scatter(df['LSTAT'], df['MEDV'], marker='o')
plt.xlabel('LSTAT')
plt.ylabel('MEDV')
#plt.show()
plt.savefig('plot.pdf')
