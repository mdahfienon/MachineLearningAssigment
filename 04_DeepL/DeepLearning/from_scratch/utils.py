#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux 
@date : 09 July 2020 
"""
import numpy as np
from functools import wraps
from collections.abc import Iterable

def batch_iterator(X, y=None, batch_size=64):
    n = X.shape[0]
    for start_idx in np.arange(0, n, batch_size):
        end_idx = min(n, start_idx+batch_size)
        if y is not None:
            yield X[start_idx:end_idx], y[start_idx:end_idx]
        else:
            yield X[start_idx:end_idx]

def parametrized(dec):
    def layer(*args, **kwargs):
        def repl(f):
            return dec(f, *args, **kwargs)
        return repl
    return layer

@parametrized
def types(fun, *types_args):
    @wraps(fun)
    def wrapper(*args, **kwargs):
        idx = 0
        args = list(args)
        for type, arg in zip(types_args, args):
            args[idx] = caster(arg, type)
            idx += 1
        for type, (key, value) in zip(tuple(types_args[idx:]), kwargs.items()):
           kwargs[key] = caster(value, type)
        args = tuple(args)
        return fun(*args, **kwargs)
    return wrapper

def caster(arg, type):
    if type == None:
        pass
    elif type == tuple:
        if isinstance(arg, Iterable):
            arg = tuple(arg)
        else:
            arg = tuple((arg,))
    elif type == np.ndarray:
        arg = np.array(arg)
    elif not isinstance(arg, type):
        arg = type(arg)
    return arg