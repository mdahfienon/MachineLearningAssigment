#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux 
@date : 11 July 2020 
"""

import math
import numpy as np


available_distributions = ['uniform', 'normal']
available_modes = ['fan_in', 'fan_out', 'fan_avg']

class kernel_initializer:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError()
    def __call__(self, shape, *args, **kwargs):
        raise NotImplementedError()

class zeros(kernel_initializer):
    def __init__(self, dtype=np.float32):
        self.dtype = dtype
    def __call__(self, shape):
        return np.zeros(shape, dtype=self.dtype)

class ones(kernel_initializer):
    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def __call__(self, shape):
        return np.ones(shape, dtype=self.dtype)

class constant(kernel_initializer):
    def __init__(self, value, dtype=np.float32):
        self.value = value
        self.dtype = dtype
    @staticmethod
    def __call__(self, shape):
        return np.full(shape, self.value, dtype=self.dtype)

class random_normal(kernel_initializer):
    def __init__(self, loc=0.0, scale=1.0, seed=None, dtype=np.float32):
        self.loc = loc
        self.scale = scale
        self.dtype = dtype
    @staticmethod
    def __call__(self, shape):
        np.random.seed(self.seed)
        return np.random.normal(self.loc, self.scale, shape)

class random_uniform(kernel_initializer):
    def __init__(self, low=0.0, high=1.0, seed=None):
        self.low = low
        self.high = high
        self.seed = seed

    def __call__(self, shape):
        np.random.seed(self.seed)
        return np.random.uniform(self.low, self.high, shape)

class variance_scaling(kernel_initializer):
    def __init__(self, scale, mode, distribution, seed=None):
        self.scale = scale
        self.mode = mode
        self.distribution = distribution
        self.seed = seed

    def __call__(self, shape):
        assert self.mode in available_modes and self.distribution in available_distributions
        np.random.seed(self.seed)
        if self.mode == 'fan_in':
            n = shape[0]
        elif self.mode == 'fan_out':
            n = shape[-1]
        elif self.mode == 'fan_avg':
            n = (shape[0] + shape[-1])/2
        if self.distribution == 'normal':
            population = np.random.normal(loc=0.0, scale=math.sqrt(self.scale/n), size=shape)
        elif self.distribution == 'uniform':
            limit = math.sqrt(3*self.scale/n)
            population = np.random.uniform(low=-limit, high=limit, size=shape)
        return population

class glorot_uniform(variance_scaling):
    def __init__(self, seed=None):
       super().__init__(scale=1.0, mode='fan_avg', distribution='uniform', seed=seed)

class glorot_normal(variance_scaling):
    def __init__(self, seed=None):
       super().__init__(scale=1.0, mode='fan_avg', distribution='normal', seed=seed)

class he_uniform(variance_scaling):
    def __init__(self, seed=None):
       super().__init__(scale=2.0, mode='fan_in', distribution='uniform', seed=seed)

class he_normal(variance_scaling):
    def __init__(self, seed=None):
       super().__init__(scale=2.0, mode='fan_in', distribution='normal', seed=seed)

class lecun_uniform(variance_scaling):
    def __init__(self, seed=None):
       super().__init__(scale=1.0, mode='fan_in', distribution='uniform', seed=seed)

class lecun_normal(variance_scaling):
    def __init__(self, seed=None):
       super().__init__(scale=1.0, mode='fan_in', distribution='normal', seed=seed)


