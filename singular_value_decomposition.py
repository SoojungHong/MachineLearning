#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 17:08:24 2018

@author: soojunghong
"""

from numpy import array
from scipy.linalg import svd

A = array([[1,2], [3,4], [5,6]])
print(A)

#singular value decomposition
U, s, VT = svd(A)
print(U)
print(s)
print(VT)