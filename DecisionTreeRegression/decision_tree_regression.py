import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.common import random_state
from sklearn.tree import DecisionTreeRegressor

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

dtr = DecisionTreeRegressor(random_state=0)
dtr.fit(x, y)
print(dtr.predict([[6.5]]))

x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, dtr.predict(x_grid), color = 'blue')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()
