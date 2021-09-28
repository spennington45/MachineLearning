import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

dataset = pd.read_csv("Position_Salaries.csv")
# Change range depending on data
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Preprocessing data here i.e. missing data or one hot incoding 
# No feacher scaling is needed for decision trees

regressor = RandomForestRegressor(n_estimators=500)
# Uses the whole data set to train 
regressor.fit(X, y)

# Uses only one indep var 
print(regressor.predict([[6.5]]))

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'green')
plt.plot(X_grid, regressor.predict(X_grid), color = 'pink')
plt.title('Decision Tree Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

plt.scatter(X, y, color = 'green')
plt.plot(X, regressor.predict(X), color = 'pink')
plt.title('Decision Tree Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

