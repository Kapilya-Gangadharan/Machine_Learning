# Support Vector Regression (SVR)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)
y = y.reshape(len(y),1)
print(y)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
print(X)
print(y)

# Training the SVR model on the whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

# Predicting a new result
sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))

plt.rcParams["figure.figsize"] = [12,8]

fig, axes = plt.subplots(nrows=1, ncols=2)

# Visualising the SVR results
axes[0].scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
axes[0].plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X)), color = 'blue')
axes[0].set(title = 'Truth or Bluff (SVR)', xlabel = 'Position level', ylabel = 'Salary')
#plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
axes[1].scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
axes[1].plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid))), color = 'blue')
axes[1].set(title = 'Truth or Bluff (SVR), Higher resolution', xlabel = 'Position level', ylabel = 'Salary')
plt.show()