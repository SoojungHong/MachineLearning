#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 16:07:28 2018

@author: soojunghong
"""

#----------------
# import modules
#----------------
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

plt.rcParams['figure.figsize'] = (10, 6)#(16, 9)

#--------------------------------------------
# Creating a sample dataset with 4 clusters
#--------------------------------------------
X, y = make_blobs(n_samples=800, n_features=3, centers=4)
X #generated sample (array of array)
y #integer labels for cluster membership

X[:,0] #X is array of array, the nested array has three element. X[:,0] means get all 0th index elements 

#--------------------
# Visualize in 3D 
#--------------------
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2])

#----------------------
# K-means clustering 
#----------------------
# Initializing KMeans
kmeans = KMeans(n_clusters=4)

# Fitting with inputs
kmeans = kmeans.fit(X)

# Predicting the clusters
labels = kmeans.predict(X)

# Getting the cluster centers
C = kmeans.cluster_centers_
C

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c='#050505', s=1000)
           
           
fruits = [["Apple", "Banana", "Mango", "Grapes", "Orange"], ['100', '200', '300', '400', '500']]
fruits[:, 1] #error - list indices must be integers, not tuple


           