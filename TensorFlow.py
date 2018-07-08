#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 16:18:26 2018

@author: soojunghong
@reference: https://www.safaribooksonline.com/library/view/hands-on-machine-learning/9781491962282/ch09.html#tensorflow_chapter

"""
import tensorflow as tf

# print to check library is imported
hello = tf.constant("Hello, TensorFlow!")
print(hello)

# create computation graph, even variables are not yet initialized with following code
x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y + y + 2

# to evaluate this graph open TensorFlow session - session take care of placing operations onto devices such as CPU, GPU and rung them
sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print(result)

# instead of manually running the initializer for every single variables use following function
init = tf.global_variables_initializer() 
with tf.Session() as sess:
    init.run() #actually initialize all the variables 
print(result)

sess = tf.InteractiveSession()
init.run()
result = f.eval()
print(result)    
sess.close() 

#-----------------
# Managing Graphs
#-----------------
x1 = tf.Variable(1)
x1.graph is tf.get_default_graph()

# create graph
graph = tf.Graph()
with graph.as_default(): # with 
    x2 = tf.Variable(2)

x2.graph is graph
x2.graph is tf.get_default_graph()   

#----------------------------
# Lifecycle of a Node Value
#----------------------------
w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3

with tf.Session() as sess:
    print(y.eval())
    print(z.eval())
 
    
#------------------------------------
# Linear Regression with TensorFlow
#------------------------------------
import numpy as np
import tensorflow as tf

from sklearn.datasets import fetch_california_housing #worked after installing scipy

housing = fetch_california_housing()
m, n = housing.data.shape
housing.data
housing_data_plus_bias = np.c_[np.ones((m,1)), housing.data] #c_ is translates slice objects to concatenation along the second axis
housing_data_plus_bias

X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name="y")
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

with tf.Session() as sess:
    theta_value = theta.eval()
    
    
#-----------------------------------
# Manually computing the gradients 
#-----------------------------------
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
scaled_housing_data_plus_bias
print(scaled_housing_data_plus_bias.mean(axis=0))
print(scaled_housing_data_plus_bias.mean(axis=1))
print(scaled_housing_data_plus_bias.mean())
print(scaled_housing_data_plus_bias.shape)

tf.reset_default_graph()
n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0), name="theta")
y_pred = tf.matmul(X, theta, name="predictions") # y_prediction equals X*theta
error = y_pred - y #error is difference between predicted value and actual value
mse = tf.reduce_mean(tf.square(error), name="mse")
gradients = 2/m * tf.matmul(tf.transpose(X), error)
training_op = tf.assign(theta, theta - learning_rate * gradients)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init) 
    
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE = ", mse.eval())
        sess.run(training_op)
    
    best_theta = theta.eval()

print("Best theta:")
print(best_theta)    

#-----------------
# Using autodiff
#-----------------
import tensorflow as tf
tf.reset_default_graph()

n_epochs = 100
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n+1,1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
gradients = tf.gradients(mse, [theta])[0]
training_op = tf.assign(theta, theta - learning_rate * gradients)
init = tf.global_variables_initializer() 

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE = ", mse.eval())
        sess.run(training_op)
    
    best_theta = theta.eval()

print("Best theta:")
print(best_theta)   

#------------------------------------------
# Feeding data to the training algorithm 
#------------------------------------------
import tensorflow as tf
import numpy as np

A = tf.placeholder(tf.float32, shape=(None, 3))
B = A + 5
with tf.Session() as sess: 
    B_val_1 = B.eval(feed_dict={A: [[1,2,3]]})
    B_val_2 = B.eval(feed_dict={A: [[4,5,6], [7,8,9]]})
    
print(B_val_1)    
print(B_val_2) 

# Mini-batch Gradient Descent
X = tf.placeholder(tf.float32, shape=(None, n+1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")

batch_size = 100
n_batches = int(np.ceil(m/batch_size))

def fetch_batch(epoch, batch_index, batch_size):
    return X_batch, y_batch

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y:y_batch})
    
    best_theta = theta.eval()        


#--------------------------------
# Visualize graph within Jupyter 
#--------------------------------
from tensorflow_graph_in_jupyter import show_graph
show_graph(tf.get_default_graph())   


#-------------------------
# Using TensorBoard
#------------------------- 