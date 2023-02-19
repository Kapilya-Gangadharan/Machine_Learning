# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Linear Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Training the Polynomial Regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


plt.rcParams["figure.figsize"] = [10,6]

fig, axes = plt.subplots(nrows=1, ncols=3)

# Visualising the Linear Regression results
axes[0].scatter(X, y, color = 'red')
axes[0].plot(X, lin_reg.predict(X), color = 'blue')
axes[0].set(title ='Truth or Bluff (Linear Regression)', xlabel = ('Position Level'), ylabel = 'Salary')
#plt.show()

# Visualising the Polynomial Regression results
axes[1].scatter(X, y, color = 'red')
axes[1].plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
axes[1].set(title= 'Truth or Bluff (Polynomial Regression)', xlabel = 'Position level', ylabel = 'Salary')
#plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
axes[2].scatter(X, y, color = 'red')
axes[2].plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
axes[2].set(title = 'Truth or Bluff (Polynomial Regression)', xlabel=  'Position level', ylabel = 'Salary')
plt.show()

# Predicting a new result with Linear Regression
lin_reg.predict([[6.5]])

# Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))