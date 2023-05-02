#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux 
@date : 12 July 2020 
"""

from tabulate import tabulate

class Layer():
    """
    Define the ABC methods of a layer
    """
    def __init__(self, *args, **kwargs):
       self.input_shape = None
       self.output_shape = None
       self.parameters = None
       self.activation = None
       self.initialized = False
       self.compiled = False

    def layer_name(self):
        """
        :return: the name of the layer (default : class name)
        """
        return self.__class__.__name__

    def initialize(self, *args, **kwargs):
        """
        Set all weights, biases and shape of the layer
        :return: None
        """
        self.initialized = True

    def compile(self, *args, **kwargs):
        """
        Set the optimizer to use for setting new weights
        :return: None
        """
        self.compiled = True

    def forward(self, x):
        """
        Forward propagation for the input x
        :param x: input
        :return: output of the layer with x as input
        """
        raise NotImplementedError()

    def backward(self, grad):
        """
        Backward propagation used to compute the gradient of the layer
        :param grad: gradient of the next layer
        :return: the gradient of the layer
        """
        raise NotImplementedError()

    def _summary_table(self):
        """
        Used to summary a model
        :return: an array with the name, input shape, output shape and number of parameters of the layer
        """
        total_input_shape = (None,)+self.input_shape if self.input_shape else None
        total_output_shape = (None,)+self.output_shape if self.output_shape else None
        return [self.layer_name(), total_input_shape, total_output_shape, self.parameters]

    def __str__(self):
        return tabulate(self._summary_table(), headers=['Layer Name', 'Input Shape', 'Output Shape', 'Nb Parameters'], tablefmt='pretty')