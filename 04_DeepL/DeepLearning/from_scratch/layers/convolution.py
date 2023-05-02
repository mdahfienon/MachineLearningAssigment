#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux 
@date : 12 July 2020 
"""
import numpy as np
import multiprocessing
from scipy import signal
from operator import mul
from functools import reduce

from from_scratch.utils import types
from from_scratch.layers.core import Layer
from from_scratch.matrix_manipulation import get_padding_function
from from_scratch.kernel_initializers import populate_kernel

class Conv1D(Layer):
    @types(None, int, tuple)
    def __init__(self, n_filters, kernel_size, strides=1, padding='valid', filters_initializer='lecun_uniform', biases_initializer='zeros', activation=None, input_shape=None):
        assert len(kernel_size) == 1
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.padding = get_padding_function(padding)
        self.activation = activation
        self.input_shape = input_shape
        self.n_channels = None
        self.weights = None
        self.biases = None
        self.filters_initializer = filters_initializer
        self.biases_initializer = biases_initializer
        self.filters_optim = None
        self.biases_optim = None
        if self.input_shape:
            self.initialize(input_shape)

    @types(None, tuple)
    def initialize(self, input_shape):
        assert len(input_shape) == 2
        self.input_shape = input_shape
        self.output_shape = (self.n_filters, *self.padding.get_output_shape(input_shape, self.kernel_size))
        self.n_channels = input_shape[-1]
        self.parameters = self.n_filters * (self.n_channels * reduce(mul, self.kernel_size) + 1)
        self.weights = populate_kernel(kernel_initializer=self.filters_initializer, shape=(self.n_filters, self.n_channels, *self.kernel_size))
        self.biases = populate_kernel(kernel_initializer=self.biases_initializer, shape=(self.n_filters,))

        self.initialized = True

    def forward(self, x):
        if not self.initialized:
            self.initialize(x[1:])
        self.inputs = x

        N, Cout, Lout = x.shape[0], self.n_filters, self.padding.get_output_shape(self.input_shape, self.kernel_size)
        output = np.zeros((N, Cout, *Lout))
        for Ni in range(N):
            for Coutj in range(Cout):
                output[Ni, Coutj] += self.biases[Coutj]
                for k in range(self.n_channels):
                    output[Ni, Coutj] += signal.convolve(x[Ni, :, k], self.weights[Coutj, k], mode=self.padding.name)
        return output


class Conv2D(Layer):
    """
    https://jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/
    """
    @types(None, int, tuple)
    def __init__(self, n_filters, kernel_size, strides=1, padding='valid', filters_initializer='lecun_uniform', biases_initializer='zeros', activation=None, input_shape=None):
        assert len(kernel_size) == 2
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.padding = get_padding_function(padding)
        self.activation = activation
        self.input_shape = input_shape
        self.n_channels = None
        self.weights = None
        self.biases = None
        self.filters_initializer = filters_initializer
        self.biases_initializer = biases_initializer
        self.filters_optim = None
        self.biases_optim = None
        if self.input_shape:
            self.initialize(input_shape)

    @types(None, tuple)
    def initialize(self, input_shape):
        assert len(input_shape) == 3
        self.input_shape = input_shape
        self.output_shape = (self.n_filters, *self.padding.get_output_shape(input_shape, self.kernel_size))
        self.n_channels = input_shape[-1]
        self.parameters = self.n_filters * (self.n_channels * reduce(mul, self.kernel_size) + 1)
        self.weights = populate_kernel(kernel_initializer=self.filters_initializer, shape=(self.n_filters, self.n_channels, *self.kernel_size))
        self.biases = populate_kernel(kernel_initializer=self.biases_initializer, shape=(self.n_filters,))

        self.initialized = True

    def forward(self, x):
        if not self.initialized:
            self.initialize(x[1:])
        self.inputs = x

        N, Cout, Lout = x.shape[0], self.n_filters, self.padding.get_output_shape(self.input_shape, self.kernel_size)
        output = np.zeros((N, Cout, *Lout))
        for Ni in range(N):
            for Coutj in range(Cout):
                output[Ni, Coutj] += self.biases[Coutj]
                for k in range(self.n_channels):
                    output[Ni, Coutj] += signal.convolve2d(x[Ni, :, :, k], self.weights[Coutj, k], mode=self.padding.name)
        return output