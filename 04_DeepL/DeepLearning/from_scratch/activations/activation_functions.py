#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 30 June 2020
"""

import numpy as np

class activation_function():
        def __init__(self):
                self.inputs = None
        def __call__(self, x):
                self.inputs = x
                return self.activate(x)

        def activate(self, x):
                raise NotImplementedError()

        def derivative(self, x):
                raise NotImplementedError()

class sigmoid(activation_function):
        def activate(self, x):
                return 1 / (1 + np.exp(-x))
        def gradient(self, x):
                return np.exp(-x) / (1 + np.exp(-x))**2
class relu(activation_function):
        def activate(self, x):
                return np.maximum(x, 0, x)
        def gradient(self, x):
                return (x > 0).astype(int)

class softmax(activation_function):
        def activate(self, x):
                x = x - np.max(x, axis=-1, keepdims=True)
                expo = np.exp(x)
                return expo/np.sum(expo, axis=-1, keepdims=True)

        def gradient(self, x):
                p = self(x)
                return p * (1 - p)
