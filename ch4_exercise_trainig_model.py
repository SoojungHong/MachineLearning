#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 18:04:23 2018

@author: soojunghong
"""

import numpy as np
import matplotlib.pyplot as plt

#------------------------
# Crossed Form Solution
#------------------------

X = 2 * np.random.rand(100,1)
y = 4 + 3 * X + np.random.randn(100, 1) 
# difference : rand, randn
# randn : Return a sample (or samples) from the “standard normal” distribution.
# rand : Random values in a given shape.

X
y

tmp = np.ones((100, 1)) # Return a new array of given shape and type, filled with ones.
tmp

X_b = np.c_[np.ones((100,1)), X]
X_b
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y) # linalg : linear algebra , inv : inverse of matrix
theta_best #theta 0 and theta 1 (two values in array)

# predict using theta values 
X_new = np.array([[0],[2]])
X_new

X_new_b = np.c_[np.ones((2, 1)), X_new] 
np.ones((2, 1))
X_new_b
y_predict = X_new_b.dot(theta_best)
y_predict

plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.axis([0,2,0,15])
plt.show()

#--------------------------------
# Equivalent with SciKit-Learn
#--------------------------------
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)
lin_reg.intercept_, lin_reg.coef_
lin_reg.predict(X_new)


#-------------------------------------------
# Gradient Descent and learning rate (eta) 
#-------------------------------------------
eta = 0.1 #learning rate
n_iterations = 1000
m = 100

theta = np.random.randn(2,1) #random initialization
theta

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients
    
theta    

#-----------------------------------------------------
# Stochastic Gradient Descent & Simulated Annealing
#-----------------------------------------------------
n_epochs = 50
t0, t1 = 5, 50 #learning schedule hyperparameters

def learning_schedule(t):
    return t0 / (t + t1)

theta = np.random.randn(2,1)

for epoch in range(n_epochs):
    for i in range(m): 
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        print(xi)
        print(yi)
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
     
        
theta   



#------------------------------
# SGDRegressor in ScikitLearn
#------------------------------
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(n_iter = 50, penalty=None, eta0=0.1)
sgd_reg.fit(X, y.ravel())     
sgd_reg.intercept_, sgd_reg.coef_


#-------------------------------------------
# Using linear model to nonlinear data set 
#-------------------------------------------
m = 100
X = 6 * np.random.rand(m, 1) - 3
X
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)
y
plt.plot(X, y, "b.")
plt.show()

# model which create polynomial features by using power of feature from nonlinear data 
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)
poly_features 
X_poly = poly_features.fit_transform(X) #X_poly now contains the original feature of X plus the square of this feature.
X[0]
X_poly[0]
X
X_poly 

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
lin_reg.intercept_, lin_reg.coef_

#---------------------------------------------------
# evaluate the model's generalization performance 
#---------------------------------------------------
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split 
def plot_learning_curves(model, X, y):    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)    
    train_errors, val_errors = [], []    
    for m in range(1, len(X_train)):        
        model.fit(X_train[:m], y_train[:m])        
        y_train_predict = model.predict(X_train[:m])        
        y_val_predict = model.predict(X_val)        
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))        
        val_errors.append(mean_squared_error(y_val_predict, y_val))    
        plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")    
        plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")


lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y)  

# learning curve of 10th degree polynomial model on same data
from sklearn.pipeline import Pipeline

polynomial_regression = Pipeline([("poly_features", PolynomialFeatures(degree=10, include_bias=False)), ("lin_reg", LinearRegression()),])
plot_learning_curves(polynomial_regression, X, y)      


#Ridge Regression 
from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=1, solver="cholesky")
ridge_reg.fit(X, y)
ridge_reg.predict([[1.5]])

#Stochastic Gradient Descent 
sgd_reg = SGDRegressor(penalty="12")
sgd_reg.fit(X, y.ravel())
sgd_reg.predict([[1.5]])

#Build classifier to detect Iris-Virginia type 
from sklearn import datasets
iris = datasets.load_iris()
iris
list(iris.keys())
X = iris["data"][:, 3:] #iris["data"] returns matrix. [:,3:] all rows and column from 3 until end column
X
y = (iris["target"] == 2).astype(np.int) #astype : 
y

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X, y)

X_new = np.linspace(0, 3, 1000).reshape(-1,1)
X_new
y_proba = log_reg.predict_proba(X_new)
plt.plot(X_new, y_proba[:,1], "g-", label="Iris-Virginica")
plt.plot(X_new, y_proba[:,0], "b--", label="Not Iris-Virginica")

log_reg.predict([[1.7], [1.5]])