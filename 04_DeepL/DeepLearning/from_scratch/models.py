#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux 
@date : 2020 Jul 09
@last modified : 2021 Apr 06, 19:29:24
"""
import time
import numpy as np
from tabulate import tabulate

from from_scratch.utils import batch_iterator, types
from from_scratch.layers import Activation


class Model:
    def __init__(self, name=None):
        self.name = name
        self.input_shape = None
        self.output_shape = None
        self.nb_parameters = None
        self.compiled = False
        self.initialized = False


class Sequential(Model):
    def __init__(self, layers, name=None):
        super().__init__(name=name)
        self.loss_function = None
        self.layers = layers
        self.input_layer = layers[0]
        if self.input_layer.input_shape is not None:
            self.initialize()

    def compile(self, loss, optimizer, metrics=None):
        self.loss_function = loss
        for layer in self.layers:
            layer.compile(optimizer)
        self.metrics = metrics
        self.compiled = True

    def initialize(self, inputs=None):
        self.input_shape = self.input_layer.input_shape or inputs.shape[1:]
        prev_input_shape = self.input_shape
        final_layers = []
        for layer in self.layers:
            final_layers.append(layer)
            layer.initialize(prev_input_shape)
            # If we find an activation in a layer, append as an Activation layer
            if layer.activation is not None:
                act = Activation(layer.activation)
                act.initialize(layer.output_shape)
                final_layers.append(act)
            prev_input_shape = layer.output_shape
        self.layers = final_layers
        self.initialized = True

    @types(None, np.ndarray)
    def forward(self, x):
        if not self.initialized:
            self.initialize(inputs=x)
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        assert self.compiled and self.initialized
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def batch_train_pass(self, X, y):
        y_hat = self.forward(X)
        loss = np.mean(self.loss_function.loss(y, y_hat))
        grad = self.loss_function.gradient(y, y_hat)
        self.backward(grad)
        return loss

    @types(None, np.ndarray, np.ndarray, int, int)
    def fit(self, X, y, batch_size=64, epochs=1, verbose=1):
        nb_batches = "Unknown"
        losses = []
        for epoch in range(epochs):
            batch_loss = []
            mean_batch_loss = 0.0
            idx_batch = 0
            for X_batch, y_batch in batch_iterator(X, y, batch_size):
                loss = self.batch_train_pass(X_batch, y_batch)
                batch_loss.append(loss)
                mean_batch_loss = np.mean(batch_loss)
                idx_batch += 1
                if verbose == 1:
                    print(
                        f"Epoch {epoch+1}/{epochs} : batch {idx_batch}/{nb_batches} : loss {mean_batch_loss:.3E}",
                        end="\r",
                    )
            losses.append(mean_batch_loss)
            nb_batches = idx_batch
            if verbose == 1: print()
        return {"loss": np.array(losses), "epochs": np.arange(epochs)}

    def summary(self):
        total_parameters = 0
        out_table = []
        for layer in self.layers:
            out_table.append(layer._summary_table())
            total_parameters += layer.parameters or 0
        out = (
            tabulate(
                out_table,
                headers=["Layer Name", "Input Shape", "Output Shape", "Nb Parameters"],
                tablefmt="pretty",
            )
            + "\n"
        )
        # out += tabulate([[self.name, self.input_shape, self.output_shape, self.nb_parameters]], headers= ['Network Name', 'Input Shape', 'Output Shape', 'Parameters'], tablefmt='pretty')
        out += f"Total parameters : {total_parameters}"
        return out

    def __iter__(self):
        yield self.layers
