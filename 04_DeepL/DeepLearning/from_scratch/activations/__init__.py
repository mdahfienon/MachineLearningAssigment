#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux 
@date : 12 July 2020 
"""

from .activation_functions import *

def get_activation_function(value):
    if isinstance(value, activation_function):
        cls =  value
    elif isinstance(value, str) and value in available_activation_functions:
        cls = available_activation_functions[value]()
    else:
        raise ValueError(f'Activation function {value} not existing, choose within {available_activation_functions.keys()}')
    return cls

available_activation_functions = {
    'sigmoid':sigmoid,
    'relu':relu,
    'softmax':softmax
}