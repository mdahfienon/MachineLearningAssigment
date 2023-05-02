#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 12 July 2020
"""

from .paddings import padding, same, valid

available_padding_functions = {
    'same':same,
    'valid':valid
}

def get_padding_function(value):
    if isinstance(value, padding):
        cls = value
    elif isinstance(value, str) and value in available_padding_functions:
        cls = available_padding_functions[value]()
    else:
        raise ValueError(f'Padding function {value} not existing, choose within {available_padding_functions.keys()}')
    return cls