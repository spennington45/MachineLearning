import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
y = y.reshape(len(y), 1)
# feature scaling is important because the level is much lower then salaries and not 0-1
scx = StandardScaler()
X = scx.fit_transform(X)
scy = StandardScaler()
y = scy.fit_transform(y)

svr = SVR()
svr.fit(X, y)

print(scy.inverse_transform(svr.predict(scx.transform([[6.5]]))))

plt.scatter(scx.inverse_transform(X), scy.inverse_transform(y), color = 'red')
plt.plot(scx.inverse_transform(X), scy.inverse_transform(svr.predict(X)), color = 'blue')
plt.title("Support Vector Regression")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

X_grid = np.arange(min(scx.inverse_transform(X)), max(scx.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(scx.inverse_transform(X), scy.inverse_transform(y), color = 'red')
plt.plot(X_grid, scy.inverse_transform(svr.predict(scx.transform(X_grid))), color = 'blue')
plt.title('Support Vector Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()