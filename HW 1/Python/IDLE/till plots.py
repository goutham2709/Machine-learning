import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv('iris.txt', sep = ',', header = None)
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
replace_names = {'Iris-setosa': '1', 'Iris-virginica': '3', 'Iris-versicolor':'2'}
#print(df.head())

for names in df.iterrows():
    df.replace(replace_names, regex = True, inplace = True)
#print(df)

x_one = df[df.columns[0]]
x_two = df[df.columns[1]]
x_three = df[df.columns[2]]
x_four = df[df.columns[3]]

plt.scatter(x_one, x_two, c='b', marker ='^')
plt.suptitle('Scatter plot')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()

plt.scatter(x_one, x_three, c='g', marker ='*')
plt.suptitle('Scatter plot')
plt.xlabel('Sepal Length')
plt.ylabel('petal Length')
plt.show()

plt.scatter(x_one, x_four, c='y', marker ='.')
plt.suptitle('Scatter plot')
plt.xlabel('Sepal Length')
plt.ylabel('Petal Width')
plt.show()

plt.scatter(x_two, x_three, c='r', marker ='+')
plt.suptitle('Scatter plot')
plt.xlabel('Sepal Width')
plt.ylabel('Petal Length')
plt.show()

plt.scatter(x_two, x_three, c='c', marker ='x')
plt.suptitle('Scatter plot')
plt.xlabel('Sepal Width')
plt.ylabel('Peatl Width')
plt.show()

plt.scatter(x_two, x_three, c='k', marker ='s', s = 10)
plt.suptitle('Scatter plot')
plt.xlabel('Petal Length')
plt.ylabel('Peatl Width')
plt.show()

