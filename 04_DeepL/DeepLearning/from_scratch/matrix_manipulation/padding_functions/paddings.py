#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux 
@date : 12 July 2020 
"""

import math
import numpy as np

def get_pad_size(kernel_size):
    mid = (kernel_size - 1) / 2
    s1 = int(math.floor(mid))
    s2 = int(math.ceil(mid))
    return (s1, s2)

class padding:
    def __init__(self):
        self.name = self.__class__.__name__
    def __call__(self, *args, **kwargs):
        pass

class same(padding):
    def __call__(self, array, kernel_size):
        kernel_size = tuple((kernel_size,))
        if len(array.shape) == 1 and len(kernel_size) == 1:
            return self.pad_1D(array, kernel_size)
        elif len(array.shape) == 2 and len(kernel_size) == 2:
            return self.pad_2D(array, kernel_size)
        else:
            raise ValueError(f"Array with shape {array.shape} not matching padding with kernel shape {kernel_size}.")

    def initialize(self, shape):
        pass

    def pad_1D(self, array, kernel_size):
        assert isinstance(kernel_size, tuple) and len(kernel_size)==1
        assert len(array.shape) == 1
        kernel_w,  = kernel_size
        (w1, w2) = get_pad_size(kernel_w)
        return np.pad(array, (w1, w2), mode='constant', constant_values=0.0)

    def pad_2D(self, array, kernel_size):
        assert isinstance(kernel_size, tuple) and len(kernel_size) == 2
        assert len(array.shape) == 2
        kernel_h, kernel_w = kernel_size
        (h1, h2) = get_pad_size(kernel_h)
        (w1, w2) = get_pad_size(kernel_w)
        return np.pad(array, ((h1, h2), (w1, w2)), mode='constant', constant_values=0.0)

    def get_output_shape(self, input_shape, kernel_size):
        return input_shape

class valid(padding):
    def __call__(self, array, kernel_size):
        return array

    def initialize(self, shape):
        pass

    def get_output_shape(self, input_shape, kernel_size):
        output_shape = []
        for idx, (input_size, kernel) in enumerate(zip(input_shape, kernel_size)):
            out = input_size - sum(get_pad_size(kernel))
            output_shape.append(out)
        return tuple(output_shape)