#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux 
@date : 11 July 2020 
"""

from .initializers import *

def populate_kernel(kernel_initializer, shape):
    kernel_initializer = get_kernel_initializer(kernel_initializer)
    return kernel_initializer(shape=shape)

def get_kernel_initializer(value):
    if isinstance(value, kernel_initializer):
        cls =  value
    elif isinstance(value, str) and value in available_kernel_initializers:
        if value in not_string_callable_kernel_initializers:
            raise TypeError(f'Kernel initializer {value} is not string callable because it needs at least these arguments : {not_string_callable_kernel_initializers[value]}.')
        cls = available_kernel_initializers[value]()
    else:
        raise ValueError(f'Kernel initializer {value} not existing, choose within {available_kernel_initializers.keys()}')
    return cls

available_kernel_initializers = {
    'zeros':zeros,
    'ones':ones,
    'constant':constant,
    'random_uniform':random_uniform,
    'random_normal':random_normal,
    'variance_scaling':variance_scaling,
    'glorot_uniform':glorot_uniform,
    'glorot_normal':glorot_normal,
    'he_uniform': he_uniform,
    'he_normal': he_normal,
    'lecun_uniform': lecun_uniform,
    'lecun_normal': lecun_normal,
}

not_string_callable_kernel_initializers = {
    'constant': ('value'),
    'variance_scaling' : ('scale', 'mode', 'distribution')
}