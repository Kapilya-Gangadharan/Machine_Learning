# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)


# subplots
plt.rcParams["figure.figsize"] = [12,8]

fig, axes = plt.subplots(nrows=1, ncols=2)


# Visualising the Training set results
axes[0].scatter(X_train, y_train, color = 'red')
axes[0].plot(X_train, regressor.predict(X_train), color = 'blue')
axes[0].set(title= 'Salary vs Experience (Training set)', xlabel = 'Years of Experience', ylabel = 'Salary')
#plt.show()

# Visualising the Test set results
axes[1].scatter(X_test, y_test, color = 'red')
axes[1].plot(X_train, regressor.predict(X_train), color = 'blue')
axes[1].set(title= 'Salary vs Experience (Test set)', xlabel ='Years of Experience', ylabel = 'Salary')
plt.show()