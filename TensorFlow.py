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
"""
!!!!!!!!!!!!!! ERROR !!!!!!!!!!!!!!!!
import numpy as np
from sklearn.datasets import fetch_california_housing #error to import

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m,1)), housing.data]

X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name="y")
XT = tf.transposse(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)   

with tf.Session() as sess:
    theta_value = theta.eval()
"""   

#--------------------------------
# Implementing Gradient Descent
#--------------------------------    