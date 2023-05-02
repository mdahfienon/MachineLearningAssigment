#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2020 Jun 30
@last modified : 2021 Apr 06, 17:15:11
"""

import math
import numpy as np
from copy import copy
from operator import mul
from functools import reduce
from tabulate import tabulate

from from_scratch.layers.core import Layer
from from_scratch.activations import get_activation_function
from from_scratch.kernel_initializers import get_kernel_initializer, populate_kernel


class Dense(Layer):
    def __init__(
        self,
        n_neurons,
        weights_initializer="lecun_uniform",
        biases_initializer="zeros",
        activation=None,
        input_shape=None,
    ):
        super().__init__()
        self.input_neurons = None
        self.n_neurons = n_neurons
        self.activation = activation
        self.initialized = False
        self.compiled = False
        self.inputs = None
        self.weights = None
        self.biases = None
        self.weights_initializer = weights_initializer
        self.biases_initializer = biases_initializer
        self.weights_optim = None
        self.biases_optim = None
        if input_shape is not None:
            self.initialize(input_shape)

    def initialize(self, input_shape):
        if not isinstance(input_shape, tuple):
            input_shape = tuple(input_shape)
        assert len(input_shape) == 1
        self.input_shape = input_shape
        self.output_shape = (self.n_neurons,)
        self.input_neurons = input_shape[0]

        self.weights = populate_kernel(
            kernel_initializer=self.weights_initializer,
            shape=(self.input_neurons, self.n_neurons),
        )
        self.biases = populate_kernel(
            kernel_initializer=self.biases_initializer, shape=(1, self.n_neurons)
        )

        self.parameters = reduce(mul, self.weights.shape) + reduce(
            mul, self.biases.shape
        )
        self.initialized = True

    def compile(self, optimizer):
        self.weights_optim = copy(optimizer)
        self.biases_optim = copy(optimizer)
        self.compiled = True

    def forward(self, x):
        if not self.initialized:
            self.initialize(x.shape[1])

        self.inputs = x
        self.output = np.matmul(x, self.weights) + self.biases
        return self.output

    def backward(self, grad, get_all=False):
        assert self.compiled and self.initialized

        grad_inputs = np.matmul(grad, self.weights.T)
        grad_weights = np.matmul(self.inputs.T, grad)
        grad_biases = np.sum(grad, axis=0, keepdims=True)

        self.weights = self.weights_optim.update(self.weights, grad_weights)
        self.biases = self.biases_optim.update(self.biases, grad_biases)

        if get_all:
            return dict(weights=grad_weights, biases=grad_biases, inputs=grad_inputs)
        return grad_inputs


class Flatten(Layer):
    def __init__(self, input_shape=None):
        super().__init__()
        self.compiled = True
        self.batch_size = None
        if input_shape is not None:
            self.initialize(input_shape)

    def initialize(self, input_shape):
        if not isinstance(input_shape, tuple):
            input_shape = tuple(input_shape)
        self.input_shape = input_shape
        self.output_shape = (reduce(mul, input_shape),)
        self.initialized = True

    def forward(self, x):
        self.batch_size = x.shape[0]
        if not self.initialized:
            self.initialize(x.shape[1:])

        return x.reshape((self.batch_size, -1))

    def backward(self, grad):
        return grad.reshape((self.batch_size,) + self.input_shape)


class Activation(Layer):
    def __init__(self, activation, input_shape=None):
        super().__init__()
        self.compiled = True
        self.inputs = None

        if input_shape is not None:
            self.initialize(input_shape)

        self.activation = get_activation_function(activation)

    def initialize(self, input_shape):
        if not isinstance(input_shape, tuple):
            input_shape = tuple(input_shape)
        self.input_shape = self.output_shape = input_shape

        self.parameters = 0
        self.initialized = True

    def forward(self, x):
        if not self.initialized:
            self.initialize(tuple(x[1:]))
        self.inputs = x
        return self.activation(x)

    def backward(self, grad):
        return self.activation.gradient(self.inputs) * grad

    def layer_name(self):
        return f"Activation ({self.activation.__class__.__name__})"
