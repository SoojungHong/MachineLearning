#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 16:18:26 2018

@author: soojunghong
@reference: https://www.safaribooksonline.com/library/view/hands-on-machine-learning/9781491962282/ch09.html#tensorflow_chapter

"""
import tensorflow as tf

hello = tf.constant("Hello, TensorFlow!")
print(hello)

x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y + y + 2

sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print(result)